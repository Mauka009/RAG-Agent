import os
# 设置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import config
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import pickle
import json

# ----- 文档加载和切分 -----
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_core.documents import Document

# ----- 检索库 -----
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import jieba
from sentence_transformers import CrossEncoder


# ============ 1. 加载配置和文档 ============
API_KEY = config.API_KEY
BASE_URL = config.BASE_URL

# 从 docs 文件夹加载文档（复用 smart_loader）
from day7_day3plusMax import smart_loader

documents = []
docs_folder = "./docs/"
supported_extensions = ('.txt', '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.docx', '.xlsx', '.xls')
for filename in os.listdir(docs_folder):
    if filename.lower().endswith(supported_extensions):
        file_path = os.path.join(docs_folder, filename)
        try:
            docs = smart_loader(file_path)
            documents.extend(docs)
            print(f"✅ 已加载: {filename}")
        except Exception as e:
            print(f"❌ 加载失败 {filename}: {e}")

if not documents:
    raise ValueError("❌ 未能加载到任何有效文档")

# 切分文档块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
print(f"📚 切分成 {len(chunks)} 个文档块")

# ============ 2. 初始化 Embeddings ============
embeddings = SiliconFlowEmbeddings(
    model="BAAI/bge-m3",
    api_key=API_KEY,
    base_url=BASE_URL
)

# ============ 3. 构建 FAISS 向量库 ============
print("🔄 构建 FAISS 索引...")
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"✅ FAISS 索引构建完成，包含 {vectorstore.index.ntotal} 个向量")

# ============ 4. 构建 BM25 检索器 ============
all_docs = chunks  # 使用切分后的文档
chunk_texts = [doc.page_content for doc in all_docs]

def tokenize(text):
    return list(jieba.cut(text))

corpus_tokens = [tokenize(doc) for doc in chunk_texts]
bm25 = BM25Okapi(corpus_tokens)
print("✅ BM25 检索器构建完成")

# ============ 5. 加载 Rerank 模型 ============
use_rerank = True
if use_rerank:
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
    print("✅ Rerank 模型加载完成")

# ============ 6. 加载测试集 ============
testset_path = "testset_llm_generated.csv"
if not os.path.exists(testset_path):
    raise FileNotFoundError(f"测试集文件 {testset_path} 未找到。")
df_test = pd.read_csv(testset_path)
questions = df_test["question"].tolist()
ground_truth_contexts = df_test["context"].tolist()
print(f"📊 加载测试集，共 {len(questions)} 个问题")

# ============ 7. 定义检索函数（与之前相同）============
def retrieve_faiss(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    return docs_and_scores

def retrieve_bm25(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[-k:][::-1]
    results = [(all_docs[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
    return results

def retrieve_hybrid(query: str, k: int = 10, alpha: float = 0.6) -> List[Tuple[Document, float]]:
    """
    混合检索：使用加权 Reciprocal Rank Fusion (RRF)
    alpha: FAISS 部分的权重,BM25 部分为 (1-alpha)
    若 alpha=0.5 则等价于标准 RRF
    """
    faiss_results = retrieve_faiss(query, k=30)
    bm25_results = retrieve_bm25(query, k=30)
    
    scores = {}
    K = 60
    # FAISS 排名贡献
    for rank, (doc, _) in enumerate(faiss_results, start=1):
        content = doc.page_content
        scores[content] = scores.get(content, 0) + alpha / (rank + K)
    # BM25 排名贡献
    for rank, (doc, _) in enumerate(bm25_results, start=1):
        content = doc.page_content
        scores[content] = scores.get(content, 0) + (1 - alpha) / (rank + K)
    
    # 按融合分数排序取 top-k
    sorted_contents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    results = []
    for content, score in sorted_contents:
        doc = next(d for d in all_docs if d.page_content == content)
        results.append((doc, score))
    return results

def retrieve_hybrid_rerank(query: str, k: int = 10, alpha: float = 0.7, rerank_top_n: int = 30) -> List[Tuple[Document, float]]:
    """
    混合检索 + Rerank
    先使用混合检索取 rerank_top_n 个候选，再用 cross-encoder 重排序
    """
    candidates = retrieve_hybrid(query, k=rerank_top_n, alpha=alpha)
    if not candidates:
        return []
    pairs = [(query, doc.page_content) for doc, _ in candidates]
    scores = reranker.predict(pairs)
    scored = [(doc, score) for (doc, _), score in zip(candidates, scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

# ============ 8. 评估函数 ============
def evaluate_retrieval(retrieve_func, k_values=[1, 3, 5, 10], name="Retriever"):
    hits = {k: 0 for k in k_values}
    mrr = {k: 0.0 for k in k_values}
    
    for idx, (query, truth) in enumerate(tqdm(zip(questions, ground_truth_contexts), total=len(questions), desc=f"Evaluating {name}")):
        retrieved = retrieve_func(query, k=max(k_values))
        retrieved_docs = [doc.page_content for doc, _ in retrieved]
        
        # 检查是否命中（包含关系）
        hit_flags = [truth in doc or doc in truth for doc in retrieved_docs]
        
        for k in k_values:
            if any(hit_flags[:k]):
                hits[k] += 1
                try:
                    rank = hit_flags.index(True) + 1
                    mrr[k] += 1.0 / rank
                except ValueError:
                    pass
    
    print(f"\n{name} 评估结果:")
    for k in k_values:
        hit_rate = hits[k] / len(questions)
        mrr_score = mrr[k] / len(questions)
        print(f"  Top-{k}: Hit Rate = {hit_rate:.4f}, MRR = {mrr_score:.4f}")

# ============ 9. 运行评估 ============
if __name__ == "__main__":
    print("\n开始评估...")
    evaluate_retrieval(retrieve_faiss, k_values=[1, 3, 5, 10], name="FAISS")
    evaluate_retrieval(retrieve_bm25, k_values=[1, 3, 5, 10], name="BM25")
    evaluate_retrieval(retrieve_hybrid, k_values=[1, 3, 5, 10], name="Hybrid (FAISS+BM25)")
    if use_rerank:
        evaluate_retrieval(retrieve_hybrid_rerank, k_values=[1, 3, 5, 10], name="Hybrid+Rerank")
    
    print("\n✅ 评估完成！")