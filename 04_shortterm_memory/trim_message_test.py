from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import  before_model
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from utils.config import Config
from utils.llms import get_llm
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

checkpointer = InMemorySaver()

# 定义 trim_messages 函数， 用于每次调 LLM 前 对消息进行修剪
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str,Any] | None:
    # 从当前 AgentState 中获取消息列表（也就是：对话历史）
    messages = state["messages"]
    # 打印修剪前的消息列表，方便调试查看原始的上下文
    logger.info(f'############修剪前的消息:{messages}')

    # 如果消息数量不超过 3 条， 说明上下文还很短，无需修剪
    # 返回 None 表示不对 state 做任何个性
    if len(messages) <= 3:
        print(f'#### 消息数量不超过 3 条')
        return None

    # 根据消息总数的奇偶性决定保留最后 3 条或 4 条消息
    # 目的是尽量保留完整的 user/assistant 轮次结构
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    logger.info(f'############修剪后的消息:{recent_messages}')

    # 返回一个用于更新 state 的字典， 只修改 ”messages" 这个字段
    return {
        "messages": [
            # 先插入一个 RemoveMessage 指令， 表示清空当前所有历史消息
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            # 再把刚才选出的最近几条消息追加进去，形成新的精简上下文
            *recent_messages
        ]
    }

agent = create_agent(
    model=llm_chat,
    middleware=[trim_messages],
    checkpointer=checkpointer
)

# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id": "1"}}

# 调用 Agent 进行第一次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字叫 南哥AGI研习社。"}]},
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

# 调用 Agent 进行第二次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于冬天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")


# 调用 Agent 进行第三次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")
