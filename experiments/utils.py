import os
import re
import pandas as pd
from typing import List, Optional
import torch
import pypdf
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Core LlamaIndex imports
from llama_index.core import (
    Document,
    Settings,
    PromptTemplate
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.response.schema import Response

# Reranking
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder
from typing import List
from pydantic import ConfigDict, Field

# Advanced retrieval components
from llama_index.retrievers.bm25 import BM25Retriever


# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================
def load_pdf_content(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    output_file = pdf_path.replace('pdfs/', 'raw_text/').replace('.pdf', '.txt')
    file_name = pdf_path.replace('pdfs/', '').replace('.pdf', '')

    # if text file already exists, load and return it
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return (file_name, f.read())
        except Exception as e:
            print(f"Error loading cached text {output_file}: {e}")

    try:
        reader = pypdf.PdfReader(pdf_path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        output = "\n".join(parts).strip()
        with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
        return (file_name, output)
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return (file_name, "")
    

def get_latest_files_per_company(pdfs_folder='pdfs/'):
    """
    Select only files matching COMPANYNAME_YEAR_ pattern and keep the latest year per company.
    """
    pdf_files = [f for f in os.listdir(pdfs_folder) if f.endswith('.pdf')]
    
    company_files = {}
    
    for filename in pdf_files:
        # Match pattern: COMPANYNAME_YEAR_
        match = re.match(r'^([A-Z_]+)_(\d{4})_', filename)
        if match:
            company = match.group(1)
            year = int(match.group(2))
            
            if company not in company_files:
                company_files[company] = (filename, year)
            else:
                # Keep the file with the latest year
                if year > company_files[company][1]:
                    company_files[company] = (filename, year)
    
    return {company: pdfs_folder + filename for company, (filename, year) in company_files.items()}


def process_item(item, selected_doc_names, loaded_pdf, selected_metadata_cols):
    """
    Process item in FinanceBench, only items from selected documents.
    """
    doc_name = item.get('doc_name', None)
    # skip items not in selected docs
    if doc_name not in selected_doc_names:
        return None
    
    doc_content = loaded_pdf[doc_name]
    metadata = {k: v for k, v in item.items() 
               if k in selected_metadata_cols}
    return Document(text=doc_content, metadata=metadata)


def process_items_parallel(dataset_items, selected_doc_names, loaded_pdf, 
                           selected_metadata_cols, max_workers=10):
    """
    Process FinanceBench items in parallel to generate documents.
    """
    # Create partial function with fixed arguments
    process_func = partial(
        process_item,
        selected_doc_names=selected_doc_names,
        loaded_pdf=loaded_pdf,
        selected_metadata_cols=selected_metadata_cols
    )
    
    # Process items in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        documents = list(executor.map(process_func, dataset_items))
    
    # Filter out None values (items not in selected docs)
    documents = [doc for doc in documents if doc is not None]
    
    return documents


# ============================================================================
# 2. CHUNKING STRATEGY
# ============================================================================
def setup_chunking_strategy(embed_model):
    """
    Configure semantic chunking for documents
    - Preserves document structure
    - Maintains context
    """
    # Use semantic splitter that understands document content
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,  # High threshold for stability,
        embed_model=embed_model,
    )
    return splitter


# ============================================================================
# 3. HYBRID RETRIEVER (Dense + Sparse)
# ============================================================================
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


# ============================================================================
# 4. Reranking
# ============================================================================
class CrossEncoderRerank(BaseNodePostprocessor):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model: CrossEncoder = Field(default=None)
    top_n: int = 3
    
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", top_n=3, device: Optional[str] = None, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(model_name, device=device)
        super().__init__(model=model, top_n=top_n, **kwargs)

    def _postprocess_nodes(self, nodes: List, query_bundle=None):
        if not nodes or query_bundle is None:
            return nodes
        query = query_bundle.query_str
        pairs = [(query, n.node.get_content()) for n in nodes]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        return [NodeWithScore(node=n.node, score=float(s)) for n, s in reranked[:self.top_n]]
    
# ============================================================================
# 4. QUERY EXPANSION
# ============================================================================
FINANCIAL_QUERY_EXPANSION_PROMPT = """
You are a financial domain expert. Expand the following financial query with 2-3 
alternative phrasings or related terms that might appear in financial documents.

Query: {query}

Return ONLY the expanded variations as a JSON array, no explanations.
Example: ["original query", "alternative phrasing 1", "alternative phrasing 2"]
"""

def expand_query(query: str, llm) -> List[str]:
    """Expand queries with synonyms and alternative phrasings"""
    prompt = PromptTemplate(FINANCIAL_QUERY_EXPANSION_PROMPT)
    query_variations= llm.complete(prompt.format(query=query))
    
    return query_variations


# ============================================================================
# 5. FINANCIAL-SPECIFIC SYSTEM PROMPT
# ============================================================================
FINANCIAL_RAG_SYSTEM_PROMPT = """You are an expert financial analyst powered by LlamaIndex.
Your role is to answer financial questions based on the FinanceBench dataset with precision and clarity.

GUIDELINES:
1. ALWAYS cite the specific document and section where you found information
2. Provide confidence levels for numerical data (High/Medium/Low confidence)
3. For comparative analysis, explicitly compare time periods or companies
4. If data is missing or unclear, state it explicitly - do NOT make up numbers
5. Include relevant financial metrics and ratios in your analysis
6. Flag any assumptions you make about the data
7. For multi-year or multi-company queries, structure responses with clear breakdowns

FINANCIAL ACCURACY IS CRITICAL. When in doubt, cite your source and indicate uncertainty.
"""

def get_query_engine(retriever, reranker=None):
    node_postprocessors = [reranker] if reranker is not None else []
    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=node_postprocessors,
        system_prompt=FINANCIAL_RAG_SYSTEM_PROMPT
    )


# ============================================================================
# 6. GET RAG OUTPUT
# ============================================================================
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
    expected_answer = dct["answer"]

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


# ============================================================================
# RAG PIPELINE
# ============================================================================
import os
import json
import yaml
import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN"]

def get_vector_index(documents, embed_model_str, indexing_storage_dir):
    if os.path.isdir(indexing_storage_dir):
        print(indexing_storage_dir)
    else:
        Settings.embed_model = HuggingFaceEmbedding(
                model_name=embed_model_str, 
                device="cpu",
                embed_batch_size=16
            )
        node_parser = setup_chunking_strategy(embed_model=Settings.embed_model)
        vector_index = VectorStoreIndex.from_documents(
            documents,
            node_parser=node_parser,
            show_progress=True
        )
        vector_index.storage_context.persist(persist_dir=indexing_storage_dir)
        print(f"Index saved to {indexing_storage_dir}")


def run_llamaindex_rag_pipeline(selected_items, documents, llm_str, embed_model_str,
                                vector_index_dir, retriever_params, 
                                output_file):
    start = time.perf_counter()
    Settings.llm = Groq(model=llm_str,temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(
                model_name=embed_model_str, 
                device="cpu",
                embed_batch_size=16
            )
    storage_context = StorageContext.from_defaults(persist_dir=vector_index_dir)
    vector_index = load_index_from_storage(storage_context)

    retriever = HybridRetriever(
            vector_index,
            documents,
            top_k=retriever_params["top_k"],
            alpha=retriever_params["alpha"]
        )
    query_engine = get_query_engine(retriever, reranker=None)

    eval_lst = asyncio.run(run_eval_async(selected_items, query_engine, concurrency=3))
    print(len(eval_lst))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_lst, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start
    print(f"Pipeline finished in {elapsed:.2f}s")


def run_one(cfg_path, selected_items, documents):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_str = cfg["llm_model"]
    embed_model_str = cfg["embedding_model"]
    retriever_params = cfg["retriever_params"]
    output_file = cfg["output_file"]
    vector_index_dir = cfg["indexing_storage_dir"]

    return run_llamaindex_rag_pipeline(
        selected_items,
        documents,
        llm_str,
        embed_model_str,
        vector_index_dir,
        retriever_params,
        output_file,
    )


async def run_all_in_processes(cfgs, selected_items, documents, max_workers=2):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        tasks = [
            loop.run_in_executor(pool, run_one, cfg_path, selected_items, documents)
            for cfg_path in cfgs
        ]
        await asyncio.gather(*tasks)


# ============================================================================
# EVALUATION
# ============================================================================
from pydantic import BaseModel, Field
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


# ------------------------------------ ANSWER USEFULNESS ------------------------------------ #
class AnswerUsefulness(BaseModel):
    score: float = Field(description="""Score with:
                                        1.0: Exceptional answer that excels in all criteria
                                        0.8: Excellent answer with minor room for improvement
                                        0.6: Good answer that adequately addresses the USER QUERY
                                        0.4: Fair answer with significant room for improvement
                                        0.2: Poor answer that barely addresses the USER QUERY
                                        0.0: Completely inadequate or irrelevant answer
                        """)
    reasoning: str = Field(description="Reasoning for the given score.")


async def evaluate_answer_usefulness_async(llm, user_query, ai_answer, referenced_answer, au_prompt_template):
    base_parser = PydanticOutputParser(pydantic_object=AnswerUsefulness)
    output_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    prompt = PromptTemplate(
        template=au_prompt_template,
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


async def process_answer_usefulness_record_async(llm, record, au_prompt_template):
    eval_result = await evaluate_answer_usefulness_async(
        llm,
        record['query'],
        record['ai_answer'],
        record['referenced_answer'],
        au_prompt_template
    )
    record['answer_usefulness_score'] = eval_result.score
    record['au_reasoning'] = eval_result.reasoning
    return record


async def get_answer_usefulness_output_async(input_df, llm, au_prompt_template):
    input_records = input_df.to_dict(orient='records')
    tasks = [process_answer_usefulness_record_async(llm, record, au_prompt_template) for record in input_records]
    output_lst = await asyncio.gather(*tasks)
    output_df = pd.DataFrame(output_lst)
    return output_df
# ------------------------------------ ANSWER USEFULNESS ------------------------------------ #