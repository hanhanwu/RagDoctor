import asyncio
import json
import traceback
import os
import uuid
import time
import psycopg2
import pandas as pd
from contextlib import asynccontextmanager
from llama_index.core import Document
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import make_url
from typing import Optional

from .utils import run_all_in_processes, run_auto_eval, run_rca, run_agg_rag_review, FINANCIAL_RAG_SYSTEM_PROMPT, rca_llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(process_job_queue())
    asyncio.create_task(cleanup_old_jobs())
    yield
app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGConfig(BaseModel):
    embedding_model: str
    top_n: int
    semantic_weight: float
    keyword_weight: float
    answer_gen_llm: str

class DataOverride(BaseModel):
    index: int
    context: Optional[str] = None
    referenced_answer: Optional[str] = None

class PreprocessRequest(BaseModel):
    dataset_name: str

class DatasetRequest(BaseModel):
    dataset: str
    rag1: RAGConfig
    rag2: RAGConfig
    data_overrides_rag2: Optional[list[DataOverride]] = None

preprocessing_status = {"status": "idle", "message": ""}
rag_data = {"rag_lst": [], "documents": [], "rag_df": None}

_job_queue: asyncio.Queue = asyncio.Queue()
_job_results: dict = {}   # job_id -> result dict
_queue_order: list = []   # job_ids waiting, in order
_rca_results: dict = {}   # rca_job_id -> result dict
_job_to_rca: dict = {}    # job_id -> rca_job_id, prevents duplicate RCA tasks

DATABASE_URL = os.getenv("DATABASE_URL_PRIVATE")
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
db_url = make_url(DATABASE_URL)


# ── RCA DB cache helpers ──────────────────────────────────────────────────────

def _db_fetch_cached_rca(config_hashes: list) -> dict:
    """Return {config_hash: {rca_records, agg_review}} for any hashes found in DB."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT config_hash, rca_records, agg_review FROM existing_rca_output WHERE config_hash = ANY(%s)",
            (config_hashes,)
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return {row[0]: {"rca_records": row[1], "agg_review": row[2]} for row in rows}


def _db_save_rca(entries: list) -> None:
    """Save [(config_hash, rca_records, agg_review), ...] — skips on conflict."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    try:
        cur = conn.cursor()
        for config_hash, rca_records, agg_review in entries:
            cur.execute(
                """
                INSERT INTO existing_rca_output (config_hash, rca_records, agg_review)
                VALUES (%s, %s, %s)
                ON CONFLICT (config_hash) DO NOTHING
                """,
                (config_hash, json.dumps(rca_records), json.dumps(agg_review))
            )
        cur.close()
    finally:
        conn.close()


def fetch_raw_data(dataset_name: str):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT data
        FROM raw_datasets
        WHERE dataset_name = %s
        ORDER BY record_index
    """, (dataset_name,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def run_fiqa_preprocessing(dataset_name: str):
    global preprocessing_status, rag_data
    try:
        preprocessing_status = {"status": "running", "message": "Preprocessing the data ..."}
        fiqa_eval = fetch_raw_data(dataset_name)
        
        rag_lst = []
        documents = []
        for idx, record in enumerate(fiqa_eval):
            record = record[0] 
            context = ''.join(record['contexts'])
            gt = ''.join(record['ground_truths'])
            if 'answer' in record.keys():
                ai0_answer = record['answer'].strip()
            else:
                ai0_answer = None

            rag_lst.append({
                 'question': record['question'],
                 'context': context,
                 'context_ct': len(record['contexts']),
                 'ground_truth': gt,
                 'ai0_answer': ai0_answer
             })
            doc = Document(
                 text=context,
                 metadata={"doc_name": idx}
             )
            documents.append(doc)
 
        rag_df = pd.DataFrame(rag_lst)
        rag_data["rag_lst"] = rag_lst
        rag_data["documents"] = documents
        rag_data["rag_df"] = rag_df
        preprocessing_status = {"status": "done", "message": "Finished data preprocessing ✅"}
        print(len(rag_lst), len(documents), rag_df.shape)
        print(rag_df.head())
    except Exception as e:
        traceback.print_exc()  # prints full error in backend terminal
        preprocessing_status = {"status": "error", "message": f"Error: {str(e)}"}


@app.post("/load-fiqa")
async def load_fiqa(request: PreprocessRequest, background_tasks: BackgroundTasks):
    # rag_data is global, only 1 run when multiple users chose the same dataset simultaneously
    if preprocessing_status["status"] == "running":
        return {"message": "Preprocessing already in progress"}
    preprocessing_status["status"] = "running"
    preprocessing_status["message"] = "Preprocessing the data ..."
    background_tasks.add_task(run_fiqa_preprocessing, request.dataset_name)
    return {"message": f"{request.dataset_name} preprocessing started"}


async def cleanup_old_jobs(max_age_seconds=14400, interval_seconds=7200):
    """Remove done/error jobs older than max_age_seconds. Runs every interval_seconds."""
    while True:
        await asyncio.sleep(interval_seconds)
        now = time.time()
        to_delete = [
            job_id for job_id, result in list(_job_results.items())
            if result.get("status") in ("done", "error")
            and now - result.get("completed_at", now) > max_age_seconds
        ]
        for job_id in to_delete:
            del _job_results[job_id]
            rca_job_id = _job_to_rca.pop(job_id, None)
            if rca_job_id:
                _rca_results.pop(rca_job_id, None)
        if to_delete:
            print(f"Cleaned up {len(to_delete)} old jobs")


async def process_job_queue():
    while True:
        job_id, request, snapshot = await _job_queue.get()
        if job_id in _queue_order:
            _queue_order.remove(job_id)
        _job_results[job_id] = {"status": "running"}
        try:
            cfgs = [request.rag1, request.rag2]
            overrides_rag2 = request.data_overrides_rag2 or []
            config_hashes = await run_all_in_processes(
                cfgs, snapshot['rag_lst'], snapshot['documents'], db_url, request.dataset,
                data_overrides_rag2=overrides_rag2
            )
            eval_results = await run_auto_eval(
                config_hashes, db_url, snapshot['rag_df'],
                data_overrides_rag2=overrides_rag2
            )
            _job_results[job_id] = {
                "status": "done",
                "completed_at": time.time(),
                "config_hashes": config_hashes,
                "rag1_config": request.rag1.model_dump(),
                "rag2_config": request.rag2.model_dump(),
                "rag1": eval_results.get(config_hashes[0], {}),
                "rag2": eval_results.get(config_hashes[1], {}),
                "eval_records_1": eval_results.get(config_hashes[0], {}).get("eval_records", []),
                "eval_records_2": eval_results.get(config_hashes[1], {}).get("eval_records", []),
            }
        except Exception as e:
            traceback.print_exc()
            _job_results[job_id] = {"status": "error", "message": str(e),
                                    "completed_at": time.time()}
        finally:
            _job_queue.task_done()


@app.get("/preprocessing-status")
async def get_preprocessing_status():
    return preprocessing_status


@app.post("/run-rags")
async def run_rags(request: DatasetRequest):
    job_id = str(uuid.uuid4())
    snapshot = {
         "rag_lst": list(rag_data["rag_lst"]),
         "documents": list(rag_data["documents"]),
         "rag_df": rag_data["rag_df"],
     }
    is_running = any(v["status"] == "running" for v in _job_results.values())
    position = len(_queue_order) + (1 if is_running else 0)
    _job_results[job_id] = {"status": "queued", "position": position}
    _queue_order.append(job_id)
    await _job_queue.put((job_id, request, snapshot))
    print(f"Job {job_id} queued at position {position}. Dataset: {request.dataset}")
    return {"status": "queued", "job_id": job_id, "position": position}


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in _job_results:
        return {"status": "not_found"}
    result = {k: v for k, v in _job_results[job_id].items() if not k.startswith("eval_records")}
    if result["status"] == "queued" and job_id in _queue_order:
        is_running = any(v["status"] == "running" for v in _job_results.values())
        result["position"] = _queue_order.index(job_id) + (1 if is_running else 0)
    return result


async def _run_rca_task(rca_job_id: str, job_id: str):
    try:
        job = _job_results[job_id]
        records_1 = job["eval_records_1"]
        records_2 = job["eval_records_2"]
        config_hashes = job.get("config_hashes", [])
        rag1_config = job.get("rag1_config", {})
        rag2_config = job.get("rag2_config", {})
        same_config = len(config_hashes) >= 2 and config_hashes[0] == config_hashes[1]

        def _to_dict(r):
            return {"root_cause_analysis": r.root_cause_analysis,
                    "improvement_suggestions": r.improvement_suggestions} if r else None

        def build_agg_df(rca_records, cfg):
            df = pd.DataFrame(list(rca_records))
            df['embedding_model'] = cfg.get('embedding_model', '')
            df['top_n_retrieval']  = cfg.get('top_n', 0)
            df['semantic_weight']  = cfg.get('semantic_weight', 0.0)
            df['answer_gen_llm']   = cfg.get('answer_gen_llm', '')
            return df[df['needs_re_eval'] == 0]

        # ── 1. Check DB cache ─────────────────────────────────────────────────
        unique_hashes = list(dict.fromkeys(config_hashes))  # deduplicated, order preserved
        cached = await asyncio.to_thread(_db_fetch_cached_rca, unique_hashes)
        print(f"RCA cache hit for: {list(cached.keys())}")

        hash_1_cached = config_hashes[0] in cached
        # If same_config, config 2 is always satisfied once config 1 is resolved
        hash_2_cached = same_config or config_hashes[1] in cached

        # ── 2. Generate missing results ───────────────────────────────────────
        to_save = []

        if not hash_1_cached and not hash_2_cached:
            # Both missing — run concurrently to preserve original performance
            rca_raw = await asyncio.gather(
                asyncio.gather(*[run_rca(r) for r in records_1]),
                asyncio.gather(*[run_rca(r) for r in records_2]),
            )
            rca_1, rca_2 = list(rca_raw[0]), list(rca_raw[1])

            agg_df_1 = build_agg_df(rca_1, rag1_config)
            agg_df_2 = build_agg_df(rca_2, rag2_config)
            agg_tasks = []
            if len(agg_df_1) > 0:
                agg_tasks.append(run_agg_rag_review(agg_df_1, rca_llm, FINANCIAL_RAG_SYSTEM_PROMPT))
            if len(agg_df_2) > 0:
                agg_tasks.append(run_agg_rag_review(agg_df_2, rca_llm, FINANCIAL_RAG_SYSTEM_PROMPT))
            agg_results = list(await asyncio.gather(*agg_tasks))
            result_iter = iter(agg_results)
            agg_review_1_dict = _to_dict(next(result_iter) if len(agg_df_1) > 0 else None)
            agg_review_2_dict = _to_dict(next(result_iter) if len(agg_df_2) > 0 else None)
            to_save = [
                (config_hashes[0], rca_1, agg_review_1_dict),
                (config_hashes[1], rca_2, agg_review_2_dict),
            ]
        else:
            # At least one config is cached — handle individually
            if hash_1_cached:
                rca_1 = cached[config_hashes[0]]["rca_records"]
                agg_review_1_dict = cached[config_hashes[0]]["agg_review"]
            else:
                rca_1 = list(await asyncio.gather(*[run_rca(r) for r in records_1]))
                agg_df_1 = build_agg_df(rca_1, rag1_config)
                agg_review_1 = await run_agg_rag_review(agg_df_1, rca_llm, FINANCIAL_RAG_SYSTEM_PROMPT) if len(agg_df_1) > 0 else None
                agg_review_1_dict = _to_dict(agg_review_1)
                to_save.append((config_hashes[0], rca_1, agg_review_1_dict))

            if same_config:
                rca_2 = rca_1
                agg_review_2_dict = agg_review_1_dict
            elif hash_2_cached:
                rca_2 = cached[config_hashes[1]]["rca_records"]
                agg_review_2_dict = cached[config_hashes[1]]["agg_review"]
            else:
                rca_2 = list(await asyncio.gather(*[run_rca(r) for r in records_2]))
                agg_df_2 = build_agg_df(rca_2, rag2_config)
                agg_review_2 = await run_agg_rag_review(agg_df_2, rca_llm, FINANCIAL_RAG_SYSTEM_PROMPT) if len(agg_df_2) > 0 else None
                agg_review_2_dict = _to_dict(agg_review_2)
                to_save.append((config_hashes[1], rca_2, agg_review_2_dict))

        # ── 3. Persist new results to DB ──────────────────────────────────────
        if to_save:
            await asyncio.to_thread(_db_save_rca, to_save)
            print(f"RCA results saved for config hashes: {[e[0] for e in to_save]}")

        # ── 4. Store in memory for polling ────────────────────────────────────
        _rca_results[rca_job_id] = {
            "status": "done",
            "completed_at": time.time(),
            "rca_records_1": list(rca_1),
            "rca_records_2": list(rca_2),
            "agg_review_1": agg_review_1_dict,
            "agg_review_2": agg_review_2_dict,
        }
    except Exception as e:
        traceback.print_exc()
        _rca_results[rca_job_id] = {"status": "error", "message": str(e),
                                    "completed_at": time.time()}
        

@app.post("/run-rca/{job_id}")
async def run_rca_endpoint(job_id: str, background_tasks: BackgroundTasks):
    job = _job_results.get(job_id)
    if not job or job.get("status") != "done":
        return {"status": "error", "message": "RAG job not done or not found"}
    config_hashes = job.get("config_hashes")
    if not config_hashes or len(config_hashes) < 2:
        return {"status": "error", "message": "Config hashes not found in job"}
    if job_id in _job_to_rca:
         existing_rca_job_id = _job_to_rca[job_id]
         existing_status = _rca_results.get(existing_rca_job_id, {}).get("status", "unknown")
         return {"status": existing_status, "rca_job_id": existing_rca_job_id}
    rca_job_id = str(uuid.uuid4())
    _job_to_rca[job_id] = rca_job_id
    _rca_results[rca_job_id] = {"status": "running"}
    background_tasks.add_task(_run_rca_task, rca_job_id, job_id)
    return {"status": "running", "rca_job_id": rca_job_id}


@app.get("/rca-status/{rca_job_id}")
async def get_rca_status(rca_job_id: str):
    if rca_job_id not in _rca_results:
        return {"status": "not_found"}
    return _rca_results[rca_job_id]