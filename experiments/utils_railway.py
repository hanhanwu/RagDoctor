import os
import json
import yaml
import asyncio
import psycopg2
from concurrent.futures import ProcessPoolExecutor

from llama_index.llms.groq import Groq
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN"]
DATABASE_URL_PUBLIC = os.getenv("DATABASE_URL_PUBLIC_RAG_DR")
conn = psycopg2.connect(DATABASE_URL_PUBLIC)
conn.autocommit = True
db_name = conn.get_dsn_parameters()['dbname']
print(f"Connected to database: {db_name}")


class HybridRetriever(BaseRetriever):
    """
    Combine dense vector retrieval with BM25 sparse retrieval
    for queries
    """
    def __init__(self, vector_index, documents, top_k: int = 5, alpha: float = 0.5):
        """
        Args:
            vector_index: Vector store index for dense retrieval
            documents: List of documents for BM25
            top_k: Number of results to return
            alpha: Weight for vector retrieval (1-alpha for BM25)
        """
        super().__init__()
        self.vector_index = vector_index
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=documents,
            similarity_top_k=top_k
        )
        self.top_k = top_k
        self.alpha = alpha
    
    def _retrieve(self, query_bundle):
        """Retrieve using both dense and sparse methods"""
        # Dense retrieval
        vector_nodes = self.vector_index.as_retriever(
            similarity_top_k=self.top_k
        ).retrieve(query_bundle)
        
        # Sparse retrieval (BM25)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        # Merge results with weighted scoring
        node_dict = {}
        
        # Add vector results
        for i, node in enumerate(vector_nodes):
            score = (1 - i / len(vector_nodes)) * self.alpha
            node_dict[node.node_id] = {'node': node, 'score': score}
        
        # Add/merge BM25 results
        for i, node in enumerate(bm25_nodes):
            score = (1 - i / len(bm25_nodes)) * (1 - self.alpha)
            if node.node_id in node_dict:
                node_dict[node.node_id]['score'] += score
            else:
                node_dict[node.node_id] = {'node': node, 'score': score}
        
        # Sort by combined score
        sorted_results = sorted(node_dict.values(), key=lambda x: x['score'], reverse=True)
        return [r['node'] for r in sorted_results[:self.top_k]]


FINANCIAL_RAG_SYSTEM_PROMPT = """You are a finance expert.
Your role is to answer financial questions with precision and clarity.

GUIDELINES:
- If data is missing or unclear, state it explicitly - do NOT make up numbers
- Include relevant financial metrics and ratios in your analysis
- Flag any assumptions you make about the data
- For complex queries, structure responses with clear breakdowns

FINANCIAL ACCURACY IS CRITICAL. When in doubt, cite your source and indicate uncertainty.
"""

def get_query_engine(retriever, reranker=None, llm=None):
    node_postprocessors = [reranker] if reranker is not None else []
    return RetrieverQueryEngine.from_args(
        retriever,  # retrieving documents
        node_postprocessors=node_postprocessors,  # a list containing the reranker
        system_prompt=FINANCIAL_RAG_SYSTEM_PROMPT,  # guiding the answer generation
        llm=llm,  # generating the answer
    )


def get_rag_response(query_engine, question: str, print_query=False) -> Response:
    """
        Query the RAG system with optional query expansion
    """
    if print_query:
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")
        
    response = query_engine.query(question)
    retrieved_nodes = response.source_nodes
    return response, retrieved_nodes


async def _run_one(dct, query_engine):
    question = dct["question"]
    expected_answer = dct["ground_truth"]

    # run blocking call in a thread
    ai_answer, retrieved_nodes = await asyncio.to_thread(
        get_rag_response, query_engine, question
    )

    retrieved_lst = [
        {
            "metadata": n.metadata["doc_name"],
            "content": n.get_content(),
        }
        for n in retrieved_nodes
    ]

    return {
        "question": question,
        "expected_answer": expected_answer,
        "ai_answer": str(ai_answer),
        "retrieved_lst": retrieved_lst,
    }


async def run_eval_async(items, query_engine, concurrency=3):
    sem = asyncio.Semaphore(concurrency)

    async def bound_run(dct):
        async with sem:
            return await _run_one(dct, query_engine)

    tasks = [bound_run(dct) for dct in items]
    results = await asyncio.gather(*tasks)
    return results


def run_llamaindex_rag_pipeline(selected_items, documents, llm_str, embed_model_str,
                                url, retriever_params, embedding_model_settings,
                                output_file):
    embed_model = HuggingFaceEmbedding(
                model_name=embed_model_str, 
                device="cpu",
                embed_batch_size=16
            )
    llm = Groq(model=llm_str, temperature=0)
    table_name=f"data_embeddings_{embedding_model_settings['name']\
                               .split('/')[-1].replace('-', '_').replace('.', 'dot')}"
    embed_dim=embedding_model_settings['settings']['embedding_dim']

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table_name,
        embed_dim=embed_dim,
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    retriever = HybridRetriever(
            index,
            documents,
            top_k=retriever_params["top_k"],
            alpha=retriever_params["alpha"]
        )
    query_engine = get_query_engine(retriever, reranker=None, llm=llm)

    eval_lst = asyncio.run(run_eval_async(selected_items, query_engine, concurrency=3))
    print(len(eval_lst))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_lst, f, ensure_ascii=False, indent=2)


def run_one(cfg_path, selected_items, documents, url):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_str = cfg["llm_model"]
    embed_model_str = cfg["embedding_model"]
    retriever_params = cfg["retriever_params"]
    output_file = cfg["output_file"]
    embedding_model_settings = cfg["embedding_model_settings"]
    
    return run_llamaindex_rag_pipeline(
        selected_items,
        documents,
        llm_str,
        embed_model_str,
        url,
        retriever_params,
        embedding_model_settings,
        output_file,
    )


async def run_all_in_processes(cfgs, selected_items, documents, url, max_workers=2):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        tasks = [
            loop.run_in_executor(pool, run_one, cfg_path, selected_items, documents, url)
            for cfg_path in cfgs
        ]
        await asyncio.gather(*tasks)


# ============================================================================
# EVALUATION
# ============================================================================
from pydantic import BaseModel, Field
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser


def get_eval_input(json_results):
    records = []
    for item in json_results:
        record = {
            'query': item['question'],
            'ai_answer': item['ai_answer'],
            'referenced_answer': item['expected_answer'],
            'retrieved_content': ''.join(content_dct['content'] for content_dct in item['retrieved_lst']),
        }
        records.append(record)
    return pd.DataFrame(records)


# ------------------------------------------ RETRIEVAL QUALITY ------------------------------------------ #
class RetrievalQuality(BaseModel):
    score: int = Field(description="""Score with:
                - Only generate the score as -1, 0 or 1 or 2 or 3
                - Scoring as -1: if the RETRIEVED CONTENT is much more relevant to the USER QUERY than the CONTEXT
                - Scoring as 0: if the RETRIEVED CONTENT is completely irrelevant to the USER QUERY
                - If the CONTEXT is strongly relevant to the USER QUERY:
                    - Scoring as 1: if the RETRIEVED CONTENT is relevant to the USER QUERY but doesn't contain any critical information from the CONTEXT
                    - Scoring as 2: if the RETRIEVED CONTENT is relevant to the USER QUERY but only partially contains critical information from the CONTEXT
                    - Scoring as 3: if the RETRIEVED CONTENT is relevant to USER QUERY and contains all the critical information from the CONTEXT
            """)
    reasoning: str = Field(description="Reasoning for the given score.")


async def evaluate_retrieval_quality_async(llm, user_query, context, retrieved_content, rq_prompt_template):
    base_parser = PydanticOutputParser(pydantic_object=RetrievalQuality)
    output_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    prompt = PromptTemplate(
        template=rq_prompt_template,
        input_variables=["user_query", "context", "retrieved_content"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    chain = prompt | llm | output_parser
    result = await chain.ainvoke({
        "user_query": user_query,
        "context": context,
        "retrieved_content": retrieved_content
    })
    return result


async def process_retrieval_quality_record_async(llm, record, rq_prompt_template):
    eval_result = await evaluate_retrieval_quality_async(
        llm,
        record['query'],
        record['context'],
        record['retrieved_content'],
        rq_prompt_template
    )
    record['retrieval_quality_score'] = eval_result.score
    record['rq_reasoning'] = eval_result.reasoning
    return record


async def get_retrieval_quality_output_async(input_df, llm, rq_prompt_template):
    input_records = input_df.to_dict(orient='records')
    tasks = [process_retrieval_quality_record_async(llm, record, rq_prompt_template) for record in input_records]
    output_lst = await asyncio.gather(*tasks)
    output_df = pd.DataFrame(output_lst)
    return output_df
# ------------------------------------ QUERY QUALITY ------------------------------------ #


# ------------------------------------ ANSWER QUALITY ------------------------------------ #
class AnswerQuality(BaseModel):
    score: int = Field(description="""Score with:
    - Only generate the score as -1, 0 or 1 or 2 or 3 or 4
    - Scoring as -1: if the AI's ANSWER is much more relevant to the USER QUERY than the REFERENCED ANSWER
    - Scoring as 0: if the AI's ANSWER is completely irrelevant to the USER QUERY
    - If the REFERENCED ANSWER is strongly relevant to the USER QUERY:
        - Scoring as 1: if the AI's ANSWER is relevant to the USER QUERY but doesn't contain any critical information from the REFERENCED ANSWER
        - Scoring as 2: if the AI's ANSWER is relevant to the USER QUERY but only partially contains critical information from the REFERENCED ANSWER
        - Scoring as 3: if the AI's ANSWER is relevant to USER QUERY and contains all the critical information from the REFERENCED ANSWER
        - Scoring as 4: if the AI's ANSWER is relevant to USER QUERY and contains more critical information than the REFERENCED ANSWER that can help answer the USER QUERY
    """)
    reasoning: str = Field(description="Reasoning for the given score.")


async def evaluate_answer_quality_async(llm, user_query, ai_answer, referenced_answer, aq_prompt_template):
    base_parser = PydanticOutputParser(pydantic_object=AnswerQuality)
    output_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    prompt = PromptTemplate(
        template=aq_prompt_template,
        input_variables=["user_query", "ai_answer", "referenced_answer"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    chain = prompt | llm | output_parser
    result = await chain.ainvoke({
        "user_query": user_query,
        "ai_answer": ai_answer,
        "referenced_answer": referenced_answer
    })
    return result


async def process_answer_quality_record_async(llm, record, aq_prompt_template):
    eval_result = await evaluate_answer_quality_async(
        llm,
        record['query'],
        record['ai_answer'],
        record['referenced_answer'],
        aq_prompt_template
    )
    record['answer_quality_score'] = eval_result.score
    record['aq_reasoning'] = eval_result.reasoning
    return record


async def get_answer_quality_output_async(input_df, llm, aq_prompt_template):
    input_records = input_df.to_dict(orient='records')
    tasks = [process_answer_quality_record_async(llm, record, aq_prompt_template) for record in input_records]
    output_lst = await asyncio.gather(*tasks)
    output_df = pd.DataFrame(output_lst)
    return output_df
# ------------------------------------ ANSWER QUALITY ------------------------------------ #