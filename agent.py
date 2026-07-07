from dotenv import load_dotenv
load_dotenv()                       # 加载 .env 中的环境变量

import config

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pypdf import PdfReader
from PIL import Image
import re
import json
import requests
import numpy as np
import gradio as gr
import jieba
import pytesseract
import time
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# LangChain 相关
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document   # 用于构造临时文档

def 假装查资料(问题):
    print("假装在查："+问题)
    return "这是假的文档内容"

def 假装调用AI(指令):
    return '{"工具":"查资料","参数":"随便"}'

用户说 = "帮我查XG-2000"

AI决定 = 假装调用AI(用户说)
AI决定 = json.loads(AI决定)

print(type(AI决定))
print(AI决定)

if AI决定["工具"] == "查资料":
    结果 = 假装查资料(AI决定["参数"])
    print("查到的结果是：" + 结果)