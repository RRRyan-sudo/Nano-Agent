"""
ReAct Agent with LangGraph
包含：工具调用、记忆管理、多步推理、最大迭代控制
"""
import os
import json
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # 记忆管理

load_dotenv()

# ===== 1. 定义 State =====
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int

# ===== 2. 定义工具 =====
@tool
def search_web(query: str) -> str:
    """搜索网络获取信息。当需要查找实时信息时使用此工具。"""
    # 这里用模拟数据，实际可接入搜索API
    mock_results = {
        "天气": "深圳今天 28°C，多云",
        "新闻": "2025年AI Agent市场规模预计达500亿美元",
    }
    for key, value in mock_results.items():
        if key in query:
            return value
    return f"搜索结果：关于'{query}'的最新信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当需要进行数学计算时使用此工具。"""
    try:
        result = eval(expression)  # 生产环境请用 sympy 或 ast.literal_eval
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def code_interpreter(code: str) -> str:
    """执行Python代码。当需要运行代码来解决问题时使用此工具。"""
    import io, contextlib
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__})
        return f"代码执行成功，输出：\n{output.getvalue()}"
    except Exception as e:
        return f"代码执行错误：{str(e)}"

# 工具列表
tools = [search_web, calculator, code_interpreter]
tool_map = {t.name: t for t in tools}

# ===== 3. 初始化 LLM（绑定工具） =====
# 支持任何兼容 OpenAI API 的模型
llm = ChatOpenAI(
    model="Qwen2.5-3B-Instruct",  # 或用本地模型如 vLLM serve 的 Qwen
    temperature=0,
    base_url="http://localhost:8000/v1",  # vLLM 本地服务地址
    api_key="EMPTY",                      # vLLM 不校验 key，但 SDK 要求非空
).bind_tools(tools)

# System Prompt - 定义 Agent 行为
SYSTEM_PROMPT = """你是一个强大的AI助手，能够使用工具来解决问题。

## 工作流程（ReAct 范式）
1. **思考 (Reason)**：分析用户的问题，思考需要哪些信息或操作
2. **行动 (Act)**：调用合适的工具获取信息或执行操作
3. **观察 (Observe)**：分析工具返回的结果
4. 重复以上步骤直到能给出完整回答

## 原则
- 如果一个工具调用不够，可以多次调用不同工具
- 给出最终回答时要综合所有工具调用的结果
- 如果工具调用失败，尝试换一种方式
"""

# ===== 4. 定义节点 =====
MAX_ITERATIONS = 10  # 防止无限循环

def call_model(state: AgentState) -> dict:
    """LLM 推理节点"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def call_tools(state: AgentState) -> dict:
    """工具执行节点"""
    last_message = state["messages"][-1]
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name in tool_map:
            # 执行工具
            result = tool_map[tool_name].invoke(tool_args)
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"未知工具：{tool_name}",
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            )
    
    return {"messages": tool_messages}

# ===== 5. 定义条件路由 =====
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    条件路由逻辑：
    1. 如果达到最大迭代次数 → 强制结束
    2. 如果 LLM 返回了 tool_calls → 继续调用工具
    3. 否则 → 结束（LLM 已给出最终回答）
    """
    # 安全阀：防止无限循环（类似 RL 训练中的 max_episode_steps）
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"
    
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ===== 6. 构建 Graph =====
def build_agent():
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    
    # 设置入口
    graph.set_entry_point("agent")
    
    # 添加条件边：agent 之后根据条件决定去 tools 还是结束
    graph.add_conditional_edges(
        "agent",           # 从哪个节点出发
        should_continue,   # 路由函数
        {
            "tools": "tools",   # 如果返回 "tools"，去 tools 节点
            "end": END          # 如果返回 "end"，结束
        }
    )
    
    # 普通边：tools 执行完后无条件回到 agent
    graph.add_edge("tools", "agent")
    
    # ===== 记忆管理 =====
    # MemorySaver 基于 thread_id 持久化对话状态
    # 生产环境可换成 SqliteSaver / PostgresSaver
    memory = MemorySaver()
    
    return graph.compile(checkpointer=memory)

# ===== 7. 运行 Agent =====
def chat(agent, user_input: str, thread_id: str = "default"):
    """单轮对话"""
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)], "iteration_count": 0},
        config=config
    )
    
    # 返回最后一条 AI 消息
    return result["messages"][-1].content

def main():
    agent = build_agent()
    
    # 可视化图结构（可选）
    # print(agent.get_graph().draw_ascii())
    
    print("=== ReAct Agent 已启动 ===")
    print("输入 'quit' 退出, 'new' 开始新对话\n")
    
    thread_id = "session_1"
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "new":
            thread_id = f"session_{hash(os.urandom(8))}"
            print(f"[新对话已创建: {thread_id}]")
            continue
        
        response = chat(agent, user_input, thread_id)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()