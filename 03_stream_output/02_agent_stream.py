import os

from langchain.agents import create_agent

# 从langchain_core.messages模块中导入3种消息类型：
# AIMessageChunk: 表示LLM流式输出的增量消息分片（用于 messages 流式模式）
# AIMessage:表示完整的一条AI消息（包含最终文本、工具调用等完整信息）
# ToolMessage: 表示工具执行完成后返回给代理的消息（包含工具返回内容等）
from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.checkpoint.memory import  InMemorySaver
from langchain.agents.structured_output import ToolStrategy

from utils.config import Config
from utils.llms import get_llm
from utils.tools import get_tools
from utils.models import Context, ResponseFormat
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

tools = get_tools()

# 使用PromptTemplate.from_file从外部文件加载系统提示词模板
# template_file指定模板文件路径（从配置中读取），encoding指定文件编码为UTF-8
# .template 属性返回模板的原始字符串内容 （尚未进行变量格式化）
system_prompt = PromptTemplate.from_file(
    template_file= Config.SYSTEM_PROMPT_TMPL,
    encoding="utf-8"
).template

human_prompt = PromptTemplate.from_file(
    template_file= Config.HUMAN_PROMPT_TMPL,
    encoding="utf-8"
).template

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm_chat,
    system_prompt=system_prompt,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# 使用 ChatPromptTemplate.from_messages 构建聊天提示模板
# 其中包含一条system和human 消息
# - system: 使用上面加载的系统提示词模板，用于定义agent的角色与规则
# - human: 使用上面加载的用户提示词模板，用于定义用户提问的表达方式
chat_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt),("human", human_prompt)]
)
raw_question = "外面的天气怎么样？"
name = "patrick"

messages = chat_prompt.format_messages(question=raw_question, name=name)

human_msg = messages[-1]
config = {"configurable": {"thread_id": "1"}}

logger.info(f"用户的问题是：: {human_msg.content}")

# 调用 Agent 进行对话
# - messages: 传入用户消息列表，这里用户问“外面的天气怎么样？”
# - config: 传入包含 thread_id 的配置，用于绑定会话上下文
# - context: 传入自定义的Context 对象（如包含 user_id 等业务相关信息）
# - stream_mode: 传入模式

# （1）使用updates模式
#    遍历 agent 的流式结果，每次循环拿到当前一步的状态增量(chunk)
# for chunk in agent.stream(
#     # 传入对话消息，这里 human_msg.content 是用户输入内容
#     {"messages": [{"role": "user","content": human_msg.content}]},
#     # 传入运行配置，例如线程id, 检查点等。
#     config=config,
#     # 传入上下文对象，可用于在图中读取 user_id 等信息
#     context=Context(user_id="1"),
#     # 使用“updates"模式，按代理步骤（stream agent progress)推送状态更新
#     stream_mode="updates",
# ):
#     # chunk 是一个dict: key 是步骤名(step), value 是该步骤的状态数据
#     for step, data in chunk.items():
#         # 打印当着步骤名，便于在控制台观察执行流程
#         print(f"current step: {step} \n\n")
#         # 打印当前步骤最后一条消息的 content_blocks，通常是本步骤的主要输出内容
#         print(f"data: {data['messages'][-1].content_blocks} \n\n")

#  （2）使用 messages 模式
# for token, metadata in agent.stream(
#     {"messages": [{"role": "user","content": human_msg.content}]},
#     config=config,
#     context=Context(user_id="1"),
#     stream_mode="messages",
# ):
#     # 打印当前产生 token 的 LangGraph 节点名（例如“model" 或 ”tools")
#     print(f"current token: {metadata['langgraph_node']}  \n\n")
#     print(f"data: {token.content_blocks} \n\n")


#  （3）使用custom模式
# for chunk in agent.stream(
#         {"messages": [{"role": "user","content": human_msg.content}]},
#     config=config,
#         context=Context(user_id="1"),
#     stream_mode="custom",
# ):
#     print(f"data: {chunk} \n\n")


#  (4)使用updates和custom组合模式
#  在同一个流中同时启用 “updates" 和 “custom" 两种流式模式
# for stream_mode, chunk in agent.stream(
#     {"messages": [{"role": "user","content": human_msg.content}]},
#     config=config,
#     context=Context(user_id="1"),
#     stream_mode=["updates","custom"],
# ):
#     print(f"current stream_mode: {stream_mode} \n\n")
#
#     # 当流式模式为 custom 时， chunk 是工具等节点通过  get_stream_writer 写出的自定义数据
#     if stream_mode == "custom":
#         print(f"data: {chunk} \n\n")
#     elif stream_mode == "updates":
#         for step, data in chunk.items():
#             print(f"step: {step} \n\n")
#             print(f"data: {data['messages'][-1].content_blocks} \n\n")

#  (5)使用messages 和 custom 组合模式
#  在同一个流中同时启用 ”messages" 和 “custom" 2种模式
#  好处： 能收到 ”messages" 和 “custom” 模式下的所有消息，适合同时需要2种模式消息的用户

for stream_node, payload in agent.stream(
    {"messages":[{"role": "user", "content": human_msg.content}]},
    config=config,
    context=Context(user_id="1"),
    stream_mode=["messages","custom"],
):
    print(f"current stream_mode: {stream_node} \n\n")

    # 当模式为 custom 时， payload是通过 get_stream_writer 写出的自定义数据
    if stream_node == "custom":
        print(f"data: {payload} \n\n")
    #当模式为 messages 时， payload 是 (token, metadata）二元组
    elif stream_node == "messages":
        token, metadata = payload
        print(f"当前节点：{metadata['langgraph_node']}")
        print(f"当前节点内容： {token.content_blocks} \n\n")


# （6）使用messages和updates组合模式
# 只有在涉及工具调用,且想拿到“完整解析好”的工具调用或完整消息时使用messages和custom组合模式
# 在 stream_mode="messages" 时,LLM 输出的是一连串增量的message chunk
# 对于工具调用会先以多个tool_call_chunk的形式逐步吐出JSON片段
# 比如先输出 {", 再输出 "city", 再输出 :"Boston" 等
# 在同一个流里同时启用 "messages" 和 "updates" 两种流式模式
for stream_mode, payload in agent.stream(
    # 传入当前轮对话消息,其中 human_msg.content 是用户输入内容
    {"messages": [{"role": "user", "content": human_msg.content}]},
    # 运行配置,例如线程 id、检查点等信息
    config=config,
    # 上下文信息,在图中可以通过 Context 获取 user_id 等
    context=Context(user_id="1"),
    # 指定要开启的流式模式列表: LLM 消息分片 + 代理步骤更新
    stream_mode=["messages", "updates"]
):
    # 打印当前返回的数据属于哪种流式模式(messages/updates)
    print(f"当前流式模式: {stream_mode}")
    # 将当前流式模式写入日志,便于观测和排查
    logger.info(f"当前流式模式: {stream_mode}")

    # 当模式为 messages 时,payload 是 (token, metadata) 二元组
    if stream_mode == "messages":
        # 解包出当前的消息分片 token 以及其元数据(包含节点名等信息)
        token, metadata = payload
        # 如果是 LLM 的增量输出(AIMessageChunk),分别处理文本和工具调用参数
        if isinstance(token, AIMessageChunk):
            # 若当前 chunk 中包含文本内容,则流式打印出来(结尾不换行,用于打字机效果)
            if token.text:
                print(f"流式文本数据：{token.text} ", end="|")
                logger.info(f"流式文本数据：{token.text} ")
            # 若当前 chunk 中包含工具调用参数增量(tool_call_chunks),则一次性打印
            if token.tool_call_chunks:
                print(f"流式工具参数数据：{token.tool_call_chunks}")
                logger.info(f"流式工具参数数据：{token.tool_call_chunks}")

    # 当模式为 updates 时,payload 是一次代理步骤(state)更新的字典
    elif stream_mode == "updates":
        # 遍历本次更新中所有来源(source),通常是节点名,如 "model"、"tools"
        for source, update in payload.items():
            # 只关心来自模型节点或工具节点的更新
            if source in ("model", "tools"):
                # 取出该节点最新的一条消息,通常是本步骤的核心输出
                message = update["messages"][-1]
                # 如果是 AIMessage 且带有 tool_calls,说明这里有完整的工具调用参数
                if isinstance(message, AIMessage) and message.tool_calls:
                    print(f"非流式工具调用完整参数: {message.tool_calls}")
                    logger.info(f"非流式工具调用完整参数: {message.tool_calls}")
                # 如果是 ToolMessage,则说明这是工具执行后的完整返回内容
                if isinstance(message, ToolMessage):
                    print(f"非流式工具调用完整返回内容: {message.content_blocks}")
                    logger.info(f"非流式工具调用完整返回内容: {message.content_blocks}")
