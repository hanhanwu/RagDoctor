from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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