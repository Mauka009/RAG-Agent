from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# 1. 定义状态的类型
class State(TypedDict):
    text: str

# 2. 定义两个简单的工作节点
def node_a(state: State) -> dict:
    return {"text": state["text"] + "a"}

def node_b(state: State) -> dict:
    return {"text": state["text"] + "b"}

# 3. 构建图
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")

# 4. 编译并运行
app = graph.compile()
result = app.invoke({"text": ""})
print(result)  # 输出: {'text': 'ab'}