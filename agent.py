import config
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools import rag_search, calculator, get_current_time, get_weather

# ---------- 1. 初始化大模型 ----------
llm = ChatOpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.7
)

# ---------- 2. 工具列表 ----------
tools = [rag_search, calculator, get_current_time, get_weather]   

# ---------- 3. ReAct 提示词 ----------
prompt = PromptTemplate.from_template("""
你是一个智能助手，可以使用工具来回答问题。
**重要：所有涉及知识、政策、定义的问题，都必须使用 RAG 检索工具，而不是凭自己的知识回答。**

你有以下工具可用：
{tools}

工具名称：{tool_names}

请使用以下格式回答：
Question: 用户的问题
Thought: 你需要思考应该用什么工具
Action: 工具名称
Action Input: 给工具的输入
Observation: 工具返回的结果
...（可以重复 Thought/Action/Observation 多次）
Thought: 我现在有足够的信息来回答
Final Answer: 给用户的最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

# ---------- 4. 创建 Agent ----------
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

def run_agent(question: str) -> str:
    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Agent执行失败：{str(e)}"

if __name__ == "__main__":
    test_questions = [
        "A类人才和B类人才的补贴区别大吗？",
        "123 + 456 * 7 等于多少？",
        "现在几点了？",
        "成都今天天气怎么样？",
    ]
    for q in test_questions:
        print(f"\n用户：{q}")
        print(f"助手：{run_agent(q)}")