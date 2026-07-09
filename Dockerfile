FROM python:3.12-slim

ENV HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir langchain==0.3.0 langchain-community==0.3.0 langchain-core==0.3.0 langchain-openai==0.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]