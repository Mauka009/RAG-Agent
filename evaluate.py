import config
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import asyncio
from typing import List

# LangChain 相关
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 复用你的 smart_loader
from day7_day3plusMax import smart_loader

# -------- 1. 固定随机种子 --------
random.seed(42)
np.random.seed(42)

# -------- 2. 加载文档 --------
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
    raise ValueError("❌ 未能加载到任何有效文档，请检查 docs 文件夹")

print(f"📚 共加载 {len(documents)} 个文档块")

# -------- 3. 切分文本块 --------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,        # 适当增大，确保每个块包含足够信息
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
print(f"✂️ 切分成 {len(chunks)} 个文本块")

# -------- 4. 初始化 LLM（使用 DeepSeek-V3.2，也支持其他模型）--------
generator_llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.2",  # 或者换成 "Qwen/Qwen2.5-72B-Instruct" 等
    openai_api_key=config.API_KEY,
    base_url=config.BASE_URL,
    temperature=0.3,          # 适当随机性，使问题多样化
    max_tokens=1024,          # 输出无需太长，节省时间
)

# -------- 5. 准备生成问题的 Prompt --------
prompt_template = PromptTemplate.from_template(
    """你是一个测试数据生成专家。请根据以下提供的文本片段，生成一个**事实性问题**，问题的答案必须**明确**出现在文本中。
问题应该清晰、简洁，并且与文档内容相关。

文本：
{context}

请直接输出问题，不要包含多余的解释。问题："""
)

chain = prompt_template | generator_llm | StrOutputParser()

# -------- 6. 随机抽样需要生成问题的块 --------
# 我们希望生成 testset_size 个问题，这里设为 50
testset_size = 50
# 如果总块数少于 testset_size，则全部使用；否则随机抽样
if len(chunks) < testset_size:
    sampled_chunks = chunks
else:
    sampled_chunks = random.sample(chunks, testset_size)

print(f"🎯 将从 {len(sampled_chunks)} 个块中生成问题")

# -------- 7. 批量生成问题（带进度条和错误处理）--------
questions_data = []
for idx, chunk in enumerate(tqdm(sampled_chunks, desc="生成问题")):
    try:
        # 只取前 1000 字符，避免超出模型上下文（但 DeepSeek 上下文很大，稳妥起见）
        context = chunk.page_content[:2000]  # 足够
        # 调用链生成问题
        question = chain.invoke({"context": context})
        # 去除可能的空白和引号
        question = question.strip().strip('"')
        # 保存问题、答案（留空后续可补充）、上下文
        questions_data.append({
            "question": question,
            "answer": "",  # 留空，可以后续人工补或由另一个模型生成
            "context": context,
            "chunk_id": idx
        })
    except Exception as e:
        print(f"❌ 生成第 {idx} 个问题时出错: {e}")
        # 可选：跳过此块或记录错误
        continue

print(f"✅ 成功生成 {len(questions_data)} 个问题")

# -------- 8. 保存为 CSV --------
df = pd.DataFrame(questions_data)
output_path = "testset_llm_generated.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"📁 测试集已保存到 {output_path}")

# -------- 9. 打印样例 --------
print("\n📋 样例问题：")
for i in range(min(3, len(df))):
    print(f"{i+1}. {df.iloc[i]['question']}")