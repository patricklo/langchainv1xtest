from utils.config import Config
from utils.llms import get_llm
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

'''
2‑Step RAG
2‑Step RAG 是最经典、最简单的一种 RAG 架构：永远是“先检索，再生成”，检索到的文档固定作为上下文喂给 LLM，一般每个请求只需要一次模型调用，延迟和成本都比较好控制

特点

结构简单、行为可预测，适合把“查资料”视为前置条件的应用，如 FAQ、文档机器人等
控制力高：最大 LLM 调用次数是预先固定的，一般是一次
延迟较快且可预估，但仍会受检索 API、网络和数据库性能影响
典型处理流程拆解

以 LangChain 文档里的描述为基础，一个 2‑Step RAG 请求大致会经历这些步骤：

(1) 用户提出问题或任务

(2) 用问题去查询知识库（向量库、SQL、搜索引擎等），检索一批最相关的文档片段

(3) 将原始问题 + 若干检索到的文档组合成 prompt，作为上下文发给 LLM

(4) LLM 在这些上下文基础上生成答案，实现“有依据”的回复
'''


llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=llm_embedding,
    persist_directory="./chroma_langchain_db"
)

# 使用@dynamic_prompt 装饰器 定义动态提示词函数，该函数会在每次模型调用 前执行
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    # 从请求状态中获取最后一条消息的文本内容作为查询
    last_query = request.state["messages"][-1].text
    # 在向量数据库中搜索与查询相关的文档
    retrieved_docs = vector_store.similarity_search(last_query)

    # 将检索到的所有文档内容用双换行符连接成一个字符串
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 构建系统消息，将检索到的上下文信息注入到提示词中
    system_prompt = (
        "你是一个乐于助人的AI助手。在你的回复中使用以下上下文："
        f"\n\n {docs_content}"
    )

    return system_prompt

# 创建 Agent 实例，传入对话模型、空工具列表和动态提示词中间件
agent = create_agent(llm_chat, middleware=[prompt_with_context])

# 定义用户查询内容，包含两个问题
query = (
    "张三九基本信息？"
)

response = agent.invoke(
    # 将用户输入包装成消息列表，角色为 user, 内容为 query
    {"messages": [{"role": "user", "content": query}]}
)

# 从 response 字典中取出 "messages“ 列表的最后一条消息的内容 ，作为代理 (agent) 的最终回复结果
result = response["messages"][-1].content

print(f"####agent 最后回复是：{result} \n")