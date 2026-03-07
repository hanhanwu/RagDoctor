import os
import json
import asyncio
import psycopg2
import hashlib
from concurrent.futures import ProcessPoolExecutor

from llama_index.llms.groq import Groq
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding_map = {
    'BAAI/bge-small-en-v1.5': {'embedding_dim': 384},
}
os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN"]


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


async def run_eval_async(items, query_engine, concurrency=3):
    sem = asyncio.Semaphore(concurrency)

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
        return

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

    eval_lst = asyncio.run(run_eval_async(selected_items, query_engine, concurrency=3))
    print(len(eval_lst))

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
        json.dumps(eval_lst),
    ))
    conn.commit()
    cur.close()
    conn.close()
    

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
        await asyncio.gather(*tasks)