import os
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

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
    [("system", system_prompt), ("human", human_prompt)],
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm_chat,
    system_prompt=system_prompt,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

raw_question_1 = "外面的天气怎么样？"
name = "Patrick"
messages_1 = chat_prompt.format_messages(question=raw_question_1, name=name)
human_msg_1 = messages_1[-1]

raw_question_2 = "杭州的天气怎么样？"
messages_2 = chat_prompt.format_messages(question=raw_question_2, name=name)
human_msg_2 = messages_2[-1]

# 调用 agent 进行对话
# - messages: 传入用户消息列5表，这里用户问“外面的天气怎么样”
# - config: 传入包含 thread_id 的配置，用于绑定会话的上下文
# - context: 传入自定义的 Context 对象 (包含 user_id 等业务相关信息）

# (1) 返回整个批次的最终输出 （2条消息）
#   - 一次性并行提交两条用户消息给 Agent 处理
responses = agent.batch(
    [
        {"messages":[{"role": "user", "content": human_msg_1.content}]},
        {"messages":[{"role": "user", "content": human_msg_2.content}]},
    ],
    config=config,
    context=Context(user_id="1"),
)

# batch() 返回的是一个【响应列表】,每个元素对应一条输入的最终结果
for res in responses:
    print(f"agent reply: {res['structured_response']} \n")


# # （2）在每个输入生成完成时接收输出
# for response in agent.batch_as_completed(
#     [
#         {"messages": [{"role": "user", "content": human_msg_1.content}]},
#         {"messages": [{"role": "user", "content": human_msg_2.content}]}
#     ],
#     config=config,
#     context=Context(user_id="1")
# ):
#     # batch_as_completed 每次返回的通常是一个元组 (idx, result)
#     # idx：当前这条结果对应的是第几个输入（下标索引）
#     # result：当前这条输入对应的真正 Agent 执行结果
#     idx, result = response
#     # 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
#     print(f"Agent最终回复是: {result['structured_response']} \n")
#     # 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
#     logger.info(f"Agent最终回复是: {result['structured_response']}")
#





















