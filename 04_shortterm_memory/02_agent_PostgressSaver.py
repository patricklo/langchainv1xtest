import os
from langchain.agents import create_agent
# 从 langchain.agents.middleware 模块中导入 SummarizationMiddleware
# 这是一个“摘要中间件”, 用于在对话过长时，
# 自动用聊天模型对早期消息做摘要并替换原始消息
# 以此控制上下文长度、同时尽量保留历史关键信息
from langchain.agents.middleware import  SummarizationMiddleware

from langchain_core.prompts import  PromptTemplate, ChatPromptTemplate
# 从 Langgraph导入内存检查点存储器，用于短期记忆与会话持久化
from langgraph.checkpoint.postgres import PostgresSaver

from langchain.agents.structured_output import ToolStrategy

from utils.config import Config
from utils.llms import get_llm
from utils.tools import get_tools
from utils.models import Context, ResponseFormat
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

tools = get_tools()

system_prompt = PromptTemplate.from_file(
    template_file=Config.SYSTEM_PROMPT_TMPL,
    encoding="utf-8",
).template

human_prompt = PromptTemplate.from_file(
    template_file=Config.HUMAN_PROMPT_TMPL,
    encoding="utf-8",

).template

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", human_prompt)]
)

# 创建一个基于数据库的检查点存储器，用于保存会话状态和记录，实现短期记忆与多轮会话关联
with PostgresSaver.from_conn_string(Config.DB_URI) as checkpointer:
    # 初始化检查点保存器的数据库表结构 - 初始化
    checkpointer.setup()

    # 使用 Langchain 的 create_agent 创建一个 Agent 实例
    # - model: 指定使用的对话 LLM 模型
    # - system_prompt:
    # - tools:
    # - context_schema:
    # - response_format:
    # - checkpoint: 传入 PostgresSaver, 使 Agent 具备按线程维度存储和恢复对话的能力
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        # 使用一个摘要中间件来管理对话历史，避免上下文过长
        # 用于生成摘要的小模型，一般比主模型更便宜，专门负责“压缩历史”
        # 触发摘要的条件：当累计 token 数超过4000 时，启动一次“对历史消息做摘要"
        # 摘要之后仍然”保留“在原始内容：在这里表示保留最近3条 message 不做摘要， keep=("messages":3) 只要说明”在做摘要时，保留最近3条原文消息”
        middleware=[SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3))],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
    )

    # （1） 第一次问答
    #  定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
    #  不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
    config = {"configurable": {"thread_id": "1"}}
    raw_question = "杭州的天气怎么样？"
    name = "patrick"

    # 使用 chat_prompt.format_messages 方法，将消息模板中的{question} {name} 等点位符
    # 替换为实际变量，生成一组完整的对话消息列表（messages)
    messages = chat_prompt.format_messages(question=raw_question, name=name)

    human_msg = messages[-1]

    # 调用 agent 进行对话
    # - messages: 传入用户消息列表，这里用户问“外面天气怎么样？”
    # - config: 传入包含 thread_id 的配置，用于绑定会话上下文
    # - context: 传入自定义的 Context 对象，如包含 user_id 等业务想着的信息）
    response = agent.invoke(
        {"messages": [{"role": "user", "content": human_msg.content}]},
        config=config,
        context=Context(user_id="1")
    )

    print(f"agent replay: {response['structured_response']}")

    # （2）第二次问答，相同的会话ID
    #  定义调用配置，其中configurable.thread_id 用于标识一段对话的唯一 “线程ID"
    #  不同 thread_id 之间的状态是隔离的，相同 thread_id 则共享对话上下文与短期记忆
    config = {"configurable": {"thread_id": "1"}}
    raw_question = "我刚才问的是哪个城市的天气？"
    name = "patrick"

    messages = chat_prompt.format_messages(question=raw_question, name=name)
    human_msg = messages[-1]
    # 调用 Agent 进行对话
    # - messages: 传入用户消息列表，这里用户问”我刚才问的是哪个城市的天气？
    # - config: 传入包含 thread_id 的配置， 用于绑定会话上下文
    # - context: 传入自定义的 Context 对象（如包含 user_id 等业务相关信息）
    response = agent.invoke(
        {"messages": [{"role": "user", "content": human_msg.content}]},
        config=config,
        context=Context(user_id="1")
    )

    print(f"agent replay: {response['structured_response']}")

    #  (3)第三次问答，不同的会话ID
    #  定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一 “线程ID”
    #  不同 thread_id 之间状态隔离，相同 的thread_id 则共享对话上下文与短期记忆
    config = {"configurable": {"thread_id": "2"}}
    raw_question = "我刚才问的是哪个城市的天气？"
    name = "patrick"
    messages = chat_prompt.format_messages(question=raw_question, name=name)
    human_msg = messages[-1]

    response = agent.invoke(
        {"messages": [{"role": "user", "content": human_msg.content}]},
        config=config,
        context=Context(user_id="1")
    )
    print(f"agent replay: {response['structured_response']}")