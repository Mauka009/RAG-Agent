# ============================================================
# config.py - 所有配置的统一管理
# ============================================================

import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# ---------- 1. 敏感信息（从 .env 读取，无默认值，缺失就报错） ----------
API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not API_KEY:
    raise ValueError("❌ 请在 .env 中设置 SILICONFLOW_API_KEY")

# ---------- 2. 路径配置（提供默认值，也可从 .env 覆盖） ----------
TEXT_FILE_PATH = os.getenv("TEXT_FILE_PATH", "./docs/")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")

# ---------- 3. 模型配置 ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------- 4. API 服务配置 ----------
BASE_URL = os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")
API_MODEL = os.getenv("API_MODEL", "deepseek-ai/DeepSeek-V3")

# ---------- 5. 检索参数 ----------
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "0.4"))
WEIGHT_VECTOR = float(os.getenv("WEIGHT_VECTOR", "0.6"))
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ---------- 6. 是否开启调试 ----------
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# 打印一下配置（方便调试）
if DEBUG:
    print("📋 当前配置:")
    print(f"  文档路径: {TEXT_FILE_PATH}")
    print(f"  向量库路径: {INDEX_PATH}")
    print(f"  嵌入模型: {EMBEDDING_MODEL}")
    print(f"  Rerank模型: {RERANK_MODEL}")
    print(f"  混合权重: BM25={WEIGHT_BM25}, Vector={WEIGHT_VECTOR}")

FORCE_REBUILD = os.getenv("FORCE_REBUILD", "False").lower() == "true"