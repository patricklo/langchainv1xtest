from langchain.tools import  tool, ToolRuntime
from langgraph.config import get_stream_writer
# 从 langchain_chroma 包中导入 Chroma, 用于构建和使用基于 Chroma 的向量数据库、向量存储
from langchain_chroma import Chroma
from langchain.agents.middleware import HumanInTheLoopMiddleware

from .config import Config
from .models import Context
from .llms import get_llm
from .logger import  LoggerManager

logger = LoggerManager.get_logger()

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

def get_tools():
    @tool("get_weather_for_location", description="为指定的城市获取天气。")
    def get_weather_for_location(city: str) -> str:
        # 从当前 LangGraph 执行上下文中获取一个流式写入器,用于发送自定义流数据
        writer = get_stream_writer()
        # 通过流式写入器发送自定义日志: 表示正在查找该城市的数据
        writer(f"正在查找城市数据: {city}")
        # 再次通过流式写入器发送自定义日志: 表示已成功获取该城市的数据
        writer(f"已获取城市数据: {city}")
        # 根据传入的城市名返回一个固定的晴天描述（此处为示例逻辑，未实际调用天气 API）
        return f"{city}的天气是晴天!"

    # 使用 @tool 装饰器注册第二个工具，工具名为 "get_user_location"，描述为“根据用户 ID 检索用户信息。”
    # 该工具通过 ToolRuntime 获取上下文中的用户信息，从而推断用户所在城市
    @tool("get_user_location", description="根据用户 ID 检索用户信息。")
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        # 从运行时上下文中读取 user_id，用于根据用户 ID 判断所属城市
        user_id = runtime.context.user_id
        # 简单的示例映射：user_id 为 "1" 时返回“北京”，否则返回“上海”
        return "北京" if user_id == "user_001" else "上海"

    # 使用 @tool 装饰器注册第三个工具，工具名为 “retrieve_context"， 描述为根据查询内容在向量数据库中进行相似度搜索
    # 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=llm_embedding,
        persist_directory="./chroma_langchain_db"
    )

    # 使用 @tool 装饰器注册第二个工具，工具名为 "retrieve_context" 描述为”根据查询内容在向量数据库中进行相似度搜索“
    @tool("retrieve_context", description="根据查询内容在向量数据库中进行相似度搜索。", response_format="content_and_artifact")
    def retrieve_context(query: str):
        # 根据查询内容在向量数据库中进行相似度搜索，返回最相关的2个文档
        # k: number of result
        retrieved_docs = vector_store.similarity_search(query, k=2)
        # 将检索到的文档序列化成字符串格式 ， 包含来源和内容信息
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
        )
        logger.info(f"######检索到的文本块：{serialized}")
        # 返回序列化的字符串和原始文档列表
        return serialized, retrieved_docs

    tools = [
        get_weather_for_location,
        get_user_location,
        retrieve_context,
    ]

    ###################### 2. Human-in-the-loop 策略配置 ##########
    interrupt_on = {
        'get_weather_for_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '调用 list_refund_reasons 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        },
        'get_user_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '调用 get_user_location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        },
        'retrieve_context': False,
    }
    logger.info(f"需要人工审批的工具有：{interrupt_on}")

    # 创建一个人工介入循环 human-in-the-loop 中间件实例
    # 该中间件用于在工具调用时拦截并等待人工审核
    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on=interrupt_on,
        description_prefix="工具调用需人工审核"
    )

    return tools, hitl_middleware



