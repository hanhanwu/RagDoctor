import os
import json
import yaml
import asyncio
from llama_index.core import VectorStoreIndex
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

embedding_map = {
    'BAAI/bge-small-en-v1.5': {'embedding_dim': 384},
}
os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN"]


def run_llamaindex_rag_pipeline(selected_items, documents, llm_str, embed_model_str,
                                embed_dim, retriever_top_n, 
                                retriever_alpha, db_url):
    embed_model = HuggingFaceEmbedding(
                model_name=embed_model_str, 
                device="cpu",
                embed_batch_size=16
            )
    llm = Groq(model=llm_str, temperature=0)
    table_name=f"data_embeddings_{embed_model_str\
                               .split('/')[-1].replace('-', '_').replace('.', 'dot')}"

    vector_store = PGVectorStore.from_params(
        database=db_url,
        host=db_url.host,
        password=db_url.password,
        port=db_url.port,
        user=db_url.username,
        table_name=table_name,
        embed_dim=embed_dim,
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    print(index)  # TEST ONLY
    return None # TEST ONLY
    

def run_one(cfg, selected_items, documents, db_url):
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
        db_url
    )


async def run_all_in_processes(cfgs, selected_items, documents, url):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        tasks = [
            loop.run_in_executor(pool, run_one, cfg, selected_items, documents, url)
            for cfg in cfgs
        ]
        await asyncio.gather(*tasks)