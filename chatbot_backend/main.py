from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
import os

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


DATABASE_URL = os.getenv("DATABASE_URL_PRIVATE")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )
DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+asyncpg://",
    1
)
engine = create_async_engine(DATABASE_URL)


@app.get("/debug/embeddings")
async def check_embeddings():
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT * FROM rag_dr_embeddings LIMIT 5")
        )
        rows = result.fetchall()

    safe_rows = []
    for row in rows:
        data = dict(row._mapping)

        # Convert embedding vector to string
        if "embedding" in data:
            data["embedding"] = str(data["embedding"])[:200]

        safe_rows.append(data)

    return {"rows": safe_rows}