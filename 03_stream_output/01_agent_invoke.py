import os
# 从 LangChain 导入 create_agent 方法，用于创建智能体（Agent）
from langchain.agents import create_agent
# 从 langchain_core.prompts 模块中导入
# PromptTemplate 用于构建单条文本提示模板,通过占位符+format 的方式动态生成提示词
# ChatPromptTemplate 用于构建多轮对话风格的提示模板,支持 system/human 等多种角色消息组合
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 从 LangGraph 导入内存检查点存储器，用于短期记忆与会话状态持久化
from langgraph.checkpoint.memory import InMemorySaver
# 从 LangChain 导入 ToolStrategy，用于指定代理使用“工具调用”的结构化输出格式
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

from agent import llm_embedding
# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from utils.config import Config
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 从自定义工具模块导入 get_tools 方法，用于获取可供 Agent 调用的工具列表
from utils.tools import get_tools
# 从自定义模型定义模块导入上下文 Context 和结构化响应模型 ResponseFormat
from utils.models import Context, ResponseFormat
# 从自定义日志模块导入 LoggerManager，用于获取日志记录器实例
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()

#根据配置中指定的LLM类型，获取对话模型 llm_chat 和 嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 获取可以用的工具列表
tools = get_tools()

# 使用PromptTemple.from_file 从外部文件中加载系统提示词模板
# template_file 指定模板文件路径（从配置中读取），endcoding为UTF-8
# .template 属性返回模板的原始字符串内容（此时，未进行变量的格式化）
system_prompt = PromptTemplate.from_file(
    template_file=Config.SYSTEM_PROMPT_TMPL,
    encoding="utf-8",
).template

human_prompt = PromptTemplate.from_file(
    template_file=Config.HUMAN_PROMPT_TMPL,
    encoding="utf-8",
).template

# 使用ChatPromptTemplate.from_messages 构建一个聊天提示模板
# 其中包含system和human消息
# - system消息：用于定义Agent的角色与规则
# -  human消息：用于定义用户提问的表达方式
chat_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
     ("human", human_prompt)],
)

checkpointer = InMemorySaver()


# 使用Langchain create_agent创建一个Agent实例
# - model:
# - system_prompt
# - tools
# - context_schema
# - response_format
# - checkpointer
agent = create_agent(
    model = llm_chat,
    system_prompt=system_prompt,
    tools=tools,
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {"configurable":{"thread_id": "1"}}

raw_question = "外面的天气怎么样？"
name = "patrick"

messages = chat_prompt.format_messages(question=raw_question, name=name)

human_msg = messages[-1]

response = agent.invoke(
    {"messages": [{"role": "user", "content": human_msg.content}]},
    config=config,
    context =Context(user_id="1"),
)



