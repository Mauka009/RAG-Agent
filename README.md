# RAG+Agent 智能问答系统

基于 LangChain 和 FastAPI 构建的 RAG 知识库问答系统，支持多文档检索与多工具 Agent 调用。

## 技术栈
- Python + FastAPI + Docker
- LangChain (ReAct Agent)
- FAISS + BM25 混合检索
- 多工具 Agent（RAG检索、计算器、时间、天气）

## 快速启动
\`\`\`bash
docker build -t rag-agent .
docker run -p 8000:8000 rag-agent
\`\`\`

## 接口示例
\`\`\`bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "A类人才有什么要求？"}'
\`\`\`

## 核心功能
- RAG 混合检索：FAISS（语义）+ BM25（关键词），权重 0.7/0.3
- Rerank 重排序：CrossEncoder 精排，Top-1 命中率 80%
- 多工具 Agent：四个工具，基于 ReAct 模式自动调用

## 评估结果
- 50个测试样本，四组对比实验
- 分块参数调优：chunk_size 400 → 600，overlap 50 → 150，Top-1命中率从52%+提升至80%
- 当前最佳配置：chunk_size=600，overlap=150，FAISS:BM25 = 0.7:0.3