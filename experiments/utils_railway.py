import os
import asyncio
import psycopg2
from concurrent.futures import ProcessPoolExecutor

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN"]
DATABASE_URL_PUBLIC = os.getenv("DATABASE_URL_PUBLIC_RAG_DR")
conn = psycopg2.connect(DATABASE_URL_PUBLIC)
conn.autocommit = True
db_name = conn.get_dsn_parameters()['dbname']
print(f"Connected to database: {db_name}")

embedding_map = {
    'llama-3.1-8b-instant': 
}


def run_llamaindex_rag_pipeline(selected_items, documents, llm_str, embed_model_str,
                                vector_index_dir, retriever_params, 
                                output_file):
    embed_model = HuggingFaceEmbedding(
                model_name=embed_model_str, 
                device="cpu",
                embed_batch_size=16
            )
    llm = Groq(model=llm_str, temperature=0)
    

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