import traceback
import os
import psycopg2
import pandas as pd
from llama_index.core import Document
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import make_url

from .utils import run_all_in_processes


app = FastAPI()

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

class PreprocessRequest(BaseModel):
    dataset_name: str

class DatasetRequest(BaseModel):
    dataset: str
    rag1: RAGConfig
    rag2: RAGConfig

preprocessing_status = {"status": "idle", "message": ""}
rag_data = {"rag_lst": [], "documents": [], "rag_df": None}

DATABASE_URL = os.getenv("DATABASE_URL_PRIVATE")
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
db_url = make_url(DATABASE_URL)


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
    preprocessing_status["status"] = "running"
    preprocessing_status["message"] = "Preprocessing the data ..."
    background_tasks.add_task(run_fiqa_preprocessing, request.dataset_name)
    return {"message": f"{request.dataset_name} preprocessing started"}


@app.get("/preprocessing-status")
async def get_preprocessing_status():
    return preprocessing_status


@app.post("/run-rags")
async def run_rags(request: DatasetRequest):
    print(f"Selected Dataset: {request.dataset}")
    print(f"RAG1 Settings: {request.rag1}")
    print(f"RAG2 Settings: {request.rag2}")

    cfgs = [request.rag1, request.rag2]
    config_hashes = await run_all_in_processes(cfgs, rag_data['rag_lst'],
                                rag_data['documents'], db_url, 
                                request.dataset)
    print("RAG Config Hashes:", config_hashes)

    return {"status": "success"}