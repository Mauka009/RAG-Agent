from fastapi import FastAPI
from pydantic import BaseModel
from day7_day3plusMax import hybrid_search_with_rerank


app = FastAPI()

@app.get("/")
def root():
    return {"message":"RAG服务已启动"}

@app.get("/health")
def health():
    return {"status":"ok"}

class SearchRequset(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search(req:SearchRequset):
    results = hybrid_search_with_rerank(req.query,req.top_k)
    return {"code":0,"data":results,"message":"success"}