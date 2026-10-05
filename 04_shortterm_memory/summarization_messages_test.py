# 从 Langchain.agents 中导入创建Agent的函数 create_agent,以及默认的 AgentState 状态类型
from langchain.agents import create_agent, AgentState
# 从 langchain.agetns.middleware 中导入 before_model 装饰器，用于声明 “模型调用前”的中间件
from langchain.agents.middleware import before_model, SummarizationMiddleware
# 从 langchain_core.messages 中导入 RemoveMessage ,用于在状态中删除消息
from langchain_core.messages import RemoveMessage
# 从 langgraph.graph.message 中导入特殊常量 REMOVE_ALL_MESSAGES, 表示删除全部消息的占位ID？
from langgraph.graph.message import REMOVE_ALL_MESSAGES
# 从 langgraph.runtime 中导入Runtime, 表示运行时上下文（包含状态，配置等）
from langgraph.runtime import Runtime
# 从typing 中导入 Any 类型，用于类型注解中表示“任意类型”
from typing import Any
# 从 LangGraph 导入内存检查点存储器，用于短期记忆与会话状态持久化
from langgraph.checkpoint.memory import InMemorySaver

# 从自定义配置模块导入 config 类
from utils.config import Config
from utils.llms import get_llm
from utils.logger import LoggerManager


logger = LoggerManager.get_logger()

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
checkpointer = InMemorySaver()

# 使用 LangChain 的 create_agent 创建一个 Agent 实例
agent = create_agent(
    model = llm_chat,
    # 使用一个摘要中间件来管理对话历史，避免上下文时长
    # 用于生成摘要的小模型，一般比主模型更便宜，专门负责“压缩历史”
    # 触发摘要的条件：当累计 token 数超过 4000 时，启动一次“对历史消息做摘要”
    # 摘要之后仍然“保留”的原始内容：这里表示保留最近 3 条 messages的完全记录（不做摘要），keep = ("messages", 3), 只是说明 “在做摘要时保留最近3条原文消息
    middleware=[SummarizationMiddleware(model=llm_chat, trigger=("tokens", 40), keep=("messages", 3))],
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content":"我的名字叫 Patrick"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")


# 调用 Agent 进行第二次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于冬天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")

# 调用 Agent 进行第三次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于秋天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")

# 调用 Agent 进行第四次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于夏天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")

# 调用 Agent 进行第五次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于春天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")


# 调用 Agent 进行第六次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")


