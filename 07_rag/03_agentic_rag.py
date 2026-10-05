
from utils.config import Config
from utils.llms import get_llm
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=llm_embedding,
    persist_directory="./chroma_langchain_db"
)

@tool("retrieve_context", description="根据查询内容在向量数据库中进行相似度搜索",
      response_format="content_and_artifact")
def retrieve_context(query: str):
    retrieved_docs = vector_store.similarity_search(query, k = 2)
    # 将检索到的文档序列化成字符串形式，包含来源和内容信息
    serialized = "\n\n".join(
        (f"Source: {doc.metadata} \n Content: {doc.page_content}") for doc in retrieved_docs
    )
    print(f"######检索到的文本块:{serialized} \n")
    return serialized, retrieved_docs

tools = [retrieve_context]

prompt = (
    "你可以使用一个工具来从文档中检索上下文。"
    "使用该工具来帮助回答用户的问题。"
)

agent = create_agent(llm_chat, tools, system_prompt=prompt)

query = (
    "张三九的基本信息？"
)

response = agent.invoke(
    {"messages":[{"role": "user", "content": query}]}
)


# 从 response 字典中取出 "messages" 列表的最后一条消息的内容，作为代理（Agent）的最终回复结果
result = response["messages"][-1].content
# 在控制台打印代理最终回复内容，方便调试和查看
print(f"######Agent最终回复是: {result} \n")

