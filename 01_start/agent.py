import os
# 从Langchain导入create_agent方法，用于创建智能休（agent)
from langchain.agents import create_agent
# 从langgraph中导入内存检查点存储器，用于短期的记忆与会话状态持久化
from langgraph.checkpoint.memory import InMemorySaver
# 从langchain导入 ToolStrategy,用于指定代理使用“工具调用”的结构化输出格式
from langchain.agents.structured_output import ToolStrategy
# 从自定义配置模块导入config类，用于读取模型类型等配置
from utils.config import Config

from utils.llms import get_llm

from utils.tools import get_tools
from utils.models import Context, ResponseFormat

from utils.logger import LoggerManager

#设置langsmith相关环境变量，开户Langchain V2版链路追踪，用于观测与设计agent 执行过程
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 获取当前智能体可用的工具列表
tools = get_tools()

# 定义系统提示词，指定 Agent 的角色和行为约束
SYSTEM_PROMPT = """你是一名擅长讲冷笑话的专业天气预报员。

你可以使用两个工具：

get_weather_for_location：用于获取某个具体地点的天气

get_user_location：用于获取用户当前所在位置

如果用户向你询问天气，一定要先确认地点。
如果从问题中可以判断出用户指的是“我所在的地方”的天气，就使用 get_user_location 工具来获取用户的位置。

得到天气信息后，必须调用 ResponseFormat 工具给出最终答案，不要直接用普通文本回复。
ResponseFormat 参数说明：
- punny_response: 带冷笑话风格的完整回复
- weather_conditions: 天气情况，未知时可为 null"""

#创建一个基于内存的检查点存储器，用于保存对话状态，实现短期记忆与多轮会话关联
checkpointer = InMemorySaver()

# 使用Langchain的create_agent创建一个agent 实例
agent = create_agent(
    model=llm_chat,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer)

# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id":"1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)



# ToolStrategy 只有在模型调用 ResponseFormat 工具时才会写入 structured_response。
# 若模型用普通文本结束对话，该字段不会出现，需要回退到最后一条消息。
logger.info(
    f"raw response: {response}"
)
structured = response.get("structured_response")
if structured is not None:
    final_reply = structured
else:
    last_message = response["messages"][-1]
    final_reply = getattr(last_message, "content", last_message)

print(f"Agent最终回复是: {final_reply} \n")
logger.info(f"Agent最终回复是: {final_reply}")
