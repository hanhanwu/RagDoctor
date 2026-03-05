import traceback
import os
import pandas as pd
from llama_index.core import Document
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine


app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DatasetRequest(BaseModel):
    dataset: str

@app.post("/run-rags")
async def run_rags(request: DatasetRequest):
    print(f"Selected Dataset: {request.dataset}")
    return {"status": "success", "dataset": request.dataset}

# preprocessing_status = {"status": "idle", "message": ""}
# rag_data = {"rag_lst": [], "documents": [], "rag_df": None}

# DATABASE_URL = os.getenv("DATABASE_URL_PRIVATE")
# DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://")
# engine = create_async_engine(DATABASE_URL)

# def run_fiqa_preprocessing():
#     global preprocessing_status, rag_data
#     try:
#         preprocessing_status = {"status": "running", "message": "Preprocessing the data ..."}
#         fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")['baseline']
        
#         output_dir = "fiqa_raw_text"
#         os.makedirs(output_dir, exist_ok=True)

#         rag_lst = []
#         documents = []
#         for idx, record in enumerate(fiqa_eval):
#             context = ''.join(record['contexts'])
#             gt = ''.join(record['ground_truths'])
#             if 'answer' in record.keys():
#                 ai0_answer = record['answer'].strip()
#             else:
#                 ai0_answer = None

#             rag_lst.append({
#                  'question': record['question'],
#                  'context': context,
#                  'context_ct': len(record['contexts']),
#                  'ground_truth': gt,
#                  'ai0_answer': ai0_answer
#              })
#             doc = Document(
#                  text=context,
#                  metadata={"doc_name": idx}
#              )
#             documents.append(doc)
 
#         rag_df = pd.DataFrame(rag_lst)
#         rag_data["rag_lst"] = rag_lst
#         rag_data["documents"] = documents
#         rag_data["rag_df"] = rag_df
#         preprocessing_status = {"status": "done", "message": "Finished data preprocessing ✅"}
#         print(len(rag_lst), len(documents), rag_df.shape)
#         print(rag_df.head())
#     except Exception as e:
#         traceback.print_exc()  # prints full error in backend terminal
#         preprocessing_status = {"status": "error", "message": f"Error: {str(e)}"}


# @app.post("/load-fiqa")
# async def load_fiqa(background_tasks: BackgroundTasks):
#     preprocessing_status["status"] = "running"
#     preprocessing_status["message"] = "Preprocessing the data ..."
#     background_tasks.add_task(run_fiqa_preprocessing)
#     return {"message": "FIQA preprocessing started"}


# @app.get("/preprocessing-status")
# async def get_preprocessing_status():
#     return preprocessing_status
