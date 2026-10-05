import os
import uuid
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tracers import context
from langgraph.checkpoint.postgres import PostgresSaver
#从 langraph 中的 Postgres 存储模块中导入 PostgreStore 类，用于把长期记忆/状态持久化到 PostgreSQL 数据库中
from langgraph.store.postgres import PostgresStore

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
    template_file= Config.SYSTEM_PROMPT_TMPL,
    encoding="utf-8",
).template

human_prompt = PromptTemplate.from_file(
    template_file= Config.HUMAN_PROMPT_TMPL,
    encoding="utf-8",
).template

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# 创建一个基于数据库检查点的存储器，用于保存对话状态，实现短期记与多轮会议关联
with(
    PostgresStore.from_conn_string(Config.DB_URI) as store,
    PostgresSaver.from_conn_string(Config.DB_URI) as checpointer,
):
    checpointer.setup()
    store.setup()

    # 使用 LangChain 的 create_agent 创建一个 Agent 实例
    # - model
    # - system_prompt
    # - tools
    # - middleware
    # - context_schema
    # - response_format: 使用ToolStrategy + ResponseFormat 定义结构化输出格式 ，支持从 Agent 状态中读取 structured_response 字段
    # - checkpointer： 传入 PostgreSaver, 使 Agent 具备按线程维度存储和恢复对话状态的能力
    # - store: 传入 PostgreStore, 使 Agent 具备按用户维度存储和查找长期记忆（如用户偏好设置）的能力
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3))],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checpointer,
        store=store,
    )

    # 读取长期记忆内容
    def read_long_term_info(user_id: str):
        # 定义用于查询的命名空间，通常用（类别，用户ID）这种形式做分区
        namespace = ("memories", user_id)

        # 在指定命名空间下搜索所有记忆数据，这里 query="" 意味着不带语言过滤，取全部 或者 由实现决定
        memories = store.search(namespace, query="")

        # 如果没有查到任何结果 返回None
        if memories is None:
            pass

        # 如果有memories , 则从每个条目中提取 value["data"] 字段，并用空格拼接成一个长字符串：
        # 条件判断确保 d.value 是字典且包含“data"键；
        # 如果 memories 为 None 或空列表，则整体结果为 ”“（空字符串）
        long_term_info = " ".join(
            [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
        ) if memories else ""

        logger.info(f"成功获取用户ID: {user_id} 的长期记忆， 内容长度：{len(long_term_info)}, 内容：{long_term_info}")

        return long_term_info


    def write_long_term_info(user_id: str, memory_info: str):
        namespace = ("memories", user_id)

        memory_id = str(uuid.uuid4())

        # 调用存储接口，将记忆写入存储：
        # - namespace: 命名空间，用于分层组织数据
        # - key: 该命名空间下唯一的主键（这里用随机UUID）
        # - value: 实际存储内容，这里用 dict 包一层，字段名为 “data"
        result = store.put(
            namespace=namespace,
            key=memory_id,
            value={"data": memory_info}
        )

        logger.info(f"成功为用户ID:{user_id}, 存储记忆，记忆ID：{memory_id}")
        return "记忆存储 成功"

    #写入长期记忆
    write_long_term_info("user_001", "patrikc_001")
    write_long_term_info("user_002", "patrikc_002")

    # (1) 第一次问答
    # 定义调用配置,其中 configurable.thread_id 用于标识一段对话的唯一 线程ID
    # configurable.user_id用于标识唯一”用户ID“
    # 不同的 thraed_id 之间状态隔离，相同 的thread_id则共享对话上下文与短期记忆
    # 不同的 user_id 之间数据隔离， 相同 的user_id 则共享长期记忆
    config = {
        "configurable": {
            "thread_id": "01",
            "user_id": "user_001",
        }
    }

    # 定义原始用户问题
    raw_question = "杭州的天气怎么样？"
    # 获取长期记忆内容 ，用户偏好设置（用户名称）
    name = read_long_term_info("user_001")

    print(f"readed long term info: {name}")

    # 使用 chat_prompt.format_messages 方法，将模板中的{question} {name}等占位符
    # 替换为实际变量，生成一组完整的对话消息列表 （messages)
    messages = chat_prompt.format_messages(question=raw_question, name=name)

    human_msg = messages[-1]

    response = agent.invoke(
        {"messages": [{"role": "user", "content": human_msg.content}]},
        config=config,
        context=Context(user_id="user_001")
    )
    print(f"Agent最终回复是: {response['structured_response']} \n\n")






























































