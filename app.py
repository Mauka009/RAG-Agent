from fastapi import FastAPI
from pydantic import BaseModel
from day7_day3plusMax import hybrid_search_with_rerank
from agent import run_agent


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

class AgentRequest(BaseModel):
    question: str

from agent import run_agent  # 在文件顶部导入

@app.post("/agent")
def agent_endpoint(req: dict):
    try:
        question = req.get("question", "")
        if not question:
            return {"code": 400, "data": None, "message": "question 不能为空"}
        result = run_agent(question)
        return {"code": 0, "data": result, "message": "success"}
    except Exception as e:
        return {"code": 500, "data": None, "message": str(e)}