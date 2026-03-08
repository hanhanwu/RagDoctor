import os
import json
import asyncio
import psycopg2
import hashlib
from concurrent.futures import ProcessPoolExecutor
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import RateLimitError

from llama_index.llms.groq import Groq
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============================================================================
# RAG Pipeline
# ============================================================================

embedding_map = {
    'BAAI/bge-small-en-v1.5': {'embedding_dim': 384},
}
os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN"]

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5)
)
async def _invoke_with_retry(chain, inputs):
    return await chain.ainvoke(inputs)


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

        if not vector_nodes and not bm25_nodes:
            raise ValueError(f"Both retrievers returned no results for: {query_bundle.query_str}")
        
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


async def run_rag_async(items, query_engine, concurrency=3):
    sem = asyncio.Semaphore(concurrency)  # Semaphore to throttle concurrency

    async def bound_run(dct):
        async with sem:
            return await _run_one(dct, query_engine)

    tasks = [bound_run(dct) for dct in items]
    results = await asyncio.gather(*tasks)
    return results


def run_llamaindex_rag_pipeline(selected_items, documents, llm_str, embed_model_str,
                                embed_dim, retriever_top_n, 
                                retriever_alpha, db_url, dataset):
    config_hash = hashlib.md5(
        f"{dataset}_{embed_model_str}_{retriever_top_n}_{retriever_alpha}_{llm_str}".encode()
    ).hexdigest()

    conn = psycopg2.connect(
        host=db_url.host,
        port=db_url.port,
        dbname=db_url.database,
        user=db_url.username,
        password=db_url.password,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM existing_rag_output WHERE config_hash = %s", (config_hash,))
    if cur.fetchone() is not None:
        print(f"Config {config_hash} already exists, skipping RAG run.")
        cur.close()
        conn.close()
        return config_hash

    embed_model = HuggingFaceEmbedding(
                model_name=embed_model_str, 
                device="cpu",
                embed_batch_size=16
            )
    llm = Groq(model=llm_str, temperature=0)
    table_name=f"data_embeddings_{embed_model_str\
                               .split('/')[-1].replace('-', '_').replace('.', 'dot')}"

    vector_store = PGVectorStore.from_params(
        database=db_url.database,
        host=db_url.host,
        password=db_url.password,
        port=db_url.port,
        user=db_url.username,
        table_name=table_name,
        embed_dim=embed_dim,
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = HybridRetriever(
            index,
            documents,
            top_k=retriever_top_n,
            alpha=retriever_alpha
        )
    query_engine = get_query_engine(retriever, reranker=None, llm=llm)

    rag_output_lst = asyncio.run(run_rag_async(selected_items, query_engine, concurrency=3))

    cur.execute("""
        INSERT INTO existing_rag_output
            (config_hash, dataset, embedding_model, top_n_retrieval,
                 semantic_weight, answer_gen_llm, output)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (config_hash) DO NOTHING
    """, (
        config_hash,
        dataset,
        embed_model_str,
        retriever_top_n,
        retriever_alpha,
        llm_str,
        json.dumps(rag_output_lst),
    ))
    conn.commit()
    cur.close()
    conn.close()

    return config_hash

def run_one(cfg, selected_items, documents, db_url, dataset):
    llm_str = cfg.answer_gen_llm
    embed_model_str = cfg.embedding_model
    embed_dim = embedding_map[embed_model_str]['embedding_dim']
    retriever_top_n = cfg.top_n
    retriever_alpha = cfg.semantic_weight
    
    return run_llamaindex_rag_pipeline(
        selected_items,
        documents,
        llm_str,
        embed_model_str,
        embed_dim,
        retriever_top_n,
        retriever_alpha,
        db_url,
        dataset
    )


async def run_all_in_processes(cfgs, selected_items, documents, url, dataset):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        tasks = [
            loop.run_in_executor(pool, run_one, cfg, 
                                 selected_items, documents, url, dataset)
            for cfg in cfgs
        ]
        config_hashes = await asyncio.gather(*tasks)
    return config_hashes


# ============================================================================
# EVALUATION
# ============================================================================
from pydantic import BaseModel, Field
import pandas as pd
import yaml
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, 'eval_prompts.yaml'), 'r') as file:
    prompt_versions = yaml.safe_load(file)

eval_llm = ChatGroq(
    groq_api_key=os.environ["GROQ_TOKEN"],
    model_name="openai/gpt-oss-20b", 
    temperature=0.7
)


def get_eval_input(db_url, config_hash):
    conn = psycopg2.connect(
        host=db_url.host,
        port=db_url.port,
        dbname=db_url.database,
        user=db_url.username,
        password=db_url.password,
    )
    cur = conn.cursor()
    cur.execute("SELECT output FROM existing_rag_output WHERE config_hash = %s", 
                (config_hash,))
    row = cur.fetchone()
    json_results = row[0]
    cur.close()
    conn.close()

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


async def eval_one_config(config_hash, db_url, rag_df):
    conn = psycopg2.connect(
        host=db_url.host,
        port=db_url.port,
        dbname=db_url.database,
        user=db_url.username,
        password=db_url.password,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM existing_auto_eval_output WHERE config_hash = %s", (config_hash,))
    if cur.fetchone() is not None:
        print(f"Config {config_hash} already exists, skipping Auto Eval.")
        cur.execute("SELECT retrieval_quality, answer_quality FROM existing_auto_eval_output WHERE config_hash = %s", (config_hash,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        rq_counts = {str(k): v for k, v in pd.DataFrame(row[0])\
                     ['retrieval_quality_score'].value_counts().to_dict().items()}
        aq_counts = {str(k): v for k, v in pd.DataFrame(row[1])\
                     ['answer_quality_score'].value_counts().to_dict().items()}
        return config_hash, rq_counts, aq_counts
    
    input_df = get_eval_input(db_url, config_hash)
    input_df = pd.merge(input_df, rag_df[['question', 'context']], 
                        left_on='query', right_on='question')
    input_df.drop(columns=['question'], inplace=True)

    retrieval_quality = await get_retrieval_quality_output_async(input_df, eval_llm,
                                                            prompt_versions['rq_prompt_template'])
    retrieval_quality['same_context'] = retrieval_quality['retrieved_content'] == retrieval_quality['context']
    
    answer_quality = await get_answer_quality_output_async(input_df, eval_llm,
                                                            prompt_versions['aq_prompt_template'])
    print(f"""retrieval quality shape: {retrieval_quality.shape},
           answer quality shape: {answer_quality.shape}""")

    cur.execute("""
            INSERT INTO existing_auto_eval_output
                (config_hash, retrieval_quality, answer_quality)
            VALUES (%s, %s, %s)
            ON CONFLICT (config_hash) DO NOTHING
        """, (
            config_hash,
            json.dumps(retrieval_quality.to_dict(orient='records')),
            json.dumps(answer_quality.to_dict(orient='records')),
    ))
    conn.commit()
    cur.close()
    conn.close()

    rq_counts = {str(k): v for k, v in retrieval_quality\
                 ['retrieval_quality_score'].value_counts().to_dict().items()}
    aq_counts = {str(k): v for k, v in answer_quality\
                 ['answer_quality_score'].value_counts().to_dict().items()}
    return config_hash, rq_counts, aq_counts


async def run_auto_eval(config_hashes, db_url, rag_df):
    results = await asyncio.gather(*[
        eval_one_config(config_hash, db_url, rag_df)
        for config_hash in config_hashes
    ])

    return {
         config_hash: {"retrieval_quality_counts": rq_counts,
                        "answer_quality_counts": aq_counts}
         for config_hash, rq_counts, aq_counts in results
     }


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
    result = await _invoke_with_retry(chain, {
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


async def get_retrieval_quality_output_async(input_df, llm, rq_prompt_template, concurrency=2):
    sem = asyncio.Semaphore(concurrency)  # Semaphore to throttle concurrency

    async def sem_task(record):
        async with sem:
            return await process_retrieval_quality_record_async(llm, record, rq_prompt_template)

    input_records = input_df.to_dict(orient='records')
    tasks = [sem_task(record) for record in input_records]
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
    result = await _invoke_with_retry(chain, {
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


async def get_answer_quality_output_async(input_df, llm, aq_prompt_template, concurrency=2):
    sem = asyncio.Semaphore(concurrency)  # Semaphore to throttle concurrency

    async def sem_task(record):
        async with sem:
            return await process_answer_quality_record_async(llm, record, aq_prompt_template)

    input_records = input_df.to_dict(orient='records')
    tasks = [sem_task(record) for record in input_records]
    output_lst = await asyncio.gather(*tasks)
    output_df = pd.DataFrame(output_lst)
    return output_df
# ------------------------------------ ANSWER QUALITY ------------------------------------ #