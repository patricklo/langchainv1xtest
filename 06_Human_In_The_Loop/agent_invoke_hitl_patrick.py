import os
import json
import uuid
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, before_model, after_agent
# 从 langchain_core.prompts 模块中导入
# PromptTemplate 用于构建单条文本提示模板,通过占位符+format 的方式动态生成提示词
# ChatPromptTemplate 用于构建多轮对话风格的提示模板,支持 system/human 等多种角色消息组合
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 从 LangGraph 导入内存检查点存储器，用于短期记忆与会话状态持久化
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.runtime import Runtime
# 从 langgraph 的 Postgres 存储模块中导入 PostgresStore 类，用于把长期记忆/状态持久化到 PostgreSQL 数据库中
from langgraph.store.postgres import PostgresStore
# 从 LangChain 导入 ToolStrategy，用于指定代理使用“工具调用”的结构化输出格式
from langchain.agents.structured_output import ToolStrategy
# Command 用于在中断后携带决策恢复执行
from langgraph.types import Command


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

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

tools, hitl_middleware = get_tools()

system_prompt = PromptTemplate.from_file(
    template_file=Config.SYSTEM_PROMPT_TMPL,
    encoding="utf-8"
).template

human_prompt = PromptTemplate.from_file(
    template_file= Config.HUMAN_PROMPT_TMPL,
    encoding="utf-8"
).template

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", human_prompt)]
)
from langgraph.config import get_config
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id")
    print(f"#############user={user_id}, msgs={len(state['messages'])}")
    logger.info(f"############user={user_id}, state={state}")

    return None  # 不改 state

@after_agent
def test_struc_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    sr = state.get("structured_response")
    if sr is None:
        return None
    # 改字段，不要整段拼成 str
    new_sr = ResponseFormat(
        punny_response="this is structured response```" + sr.punny_response + "```",
        weather_conditions=sr.weather_conditions,
    )
    #return {"structured_response": new_sr}
    return state

# 封装一个带 HITL 审核的调用函数，问答统一用它
# 定义一个带有人在环路的（HITL）机制的运行函数
# agent: 代理对象；
# user_content: 用户输入内容
# config: 运行配置字典
# context: 上下文对象
def run_with_hitl_invoke(agent, user_content: str, config: dict, context: Context):
    # 第一次调用代理，发送用户消息，发起对话/执行
    result = agent.invoke(
        # 将用户输入包装为消息列表，角色为 user，内容为 user_content
        {"messages": [{"role": "user", "content": user_content}]},
        # 传入配置，用于控制代理行为
        config=config,
        # 传入上下文对象，维持对话或执行环境
        context=context,
    )

    # 使用循环持续处理所有中断情况
    # 只要返回结果中包含 “__interrupt__"字段，就说明有人工审核步骤，需要处理
    # 支持同时有多个工具需要审批，也支持工具链式调用 ，如先 get_location 再 get_weather
    while "__interrupt__" in result:
        # 从结果中取出所有中断请求列表
        hitl_requests = result["__interrupt__"]
        # 简化处理： 只取第一个的中断请求
        hitl_req = hitl_requests[0]

        # 从中断请求中取出所有需要人工审核的工具调用请求
        action_requests = hitl_req.value["action_requests"]
        # 从中断请求中取出每个工具调用对应的审核配置
        review_configs = hitl_req.value["review_configs"]

        # 初始化一个决策列表，用来收集对每个工具调用的人工决策
        decisions = []
        # 遍历所有工具调用 请求，i 为索引， ar 为单个请求
        for i, ar in enumerate(action_requests):
            # 从请求中取得工具名称
            name = ar.get("name")
            # 兼容两种参数的写法：优先从 "args" 取， 如果没有再从 “arguments" 取
            args = ar.get("args", ar.get("arguments"))

            # 从对应的审核配置中获取当前工具允许的决策类型
            allowed = review_configs[i]["allowed_decisions"]

            # 在控制台提示检测到需要人工审核的工具调用
            print("\n=== 检测到工具调用 需要审核 ****")
            print(f"工具名：{name}")
            # 打印工具参数
            print(f"参数：{args}")
            # 打印允许的决策类型列表
            print(f"允许的决策类型：{allowed}")

            # 提示人工输入决策类型，并去掉前后空格
            decision_type = input("请输入决策（approve/edit/reject): ").strip().lower()
            # 如果输入不在允许的决策列表中，则循环要求重新输入
            while decision_type not in allowed:
                decision_type = input(f"非法决策，请重新输入({', '.join(allowed)}): ").strip()

            # 如果决策为 edit, 表示需要修改工具调用的参数
            if decision_type == "edit":
                new_args_str = input("请输入修改后的 args(JSON 字符串）：").strip()
                new_args = json.loads(new_args_str)

                # 构造编辑后的工具调用动作，只允许个性 args, 保持 name 不变
                edited_action = {
                    "name": name,
                    "args": new_args,
                }
                # 将编辑类型的决策及编辑后的动作加入决策列表
                decisions.append({
                    "type": "edit",
                    "edited_action": edited_action,
                })
            else:
                # 对于 approve 或 reject 等非编辑类决策，仅记录决策类型
                decisions.append({"type": decision_type})

        # 带着上一步收集的 decisions 信息恢复代理执行
        # Commnad(resume=...) 告诉agent根据这些人工决策继续往下跑
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            context=context,
        )

    # 跳出 while 循环，说明已经没有新的中断需要审核
    # 此时 result 即为整个流程的最终结果，直接返回
    return result

# 创建一个基于数据库的检查点存储器，用于保存对话状态，实现短期记忆与多轮会话关联
with(
    PostgresStore.from_conn_string(Config.DB_URI) as store,
    PostgresSaver.from_conn_string(Config.DB_URI) as checkpointer
):
    #初始化saver和store
    checkpointer.setup()
    store.setup()

    # 使用 LangChain 的 create_agent 创建一个 agent 实例
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[
            log_before_model,
            test_struc_response,
            SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3)),
            hitl_middleware
        ],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
        store=store,
    )

    # 读取长期记忆内容
    def read_long_term_info(user_id: str):
        # 定义用于查询的命名空间，通常用（类别，用户ID)这种形式做分区
        namespace = ("memories", user_id)

        # 在指定命名空间下搜索所有记忆数据
        memories = store.search(namespace, query="")
        if memories is None:
            pass

        # 如果有 memories, 则从每个条目中提取 value["data"] 字段，并用空格拼接成一个长字符
        logger.info(f"##########get_long_term_info: {memories}")

        long_term_info = " ".join(
            [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
        ) if memories else ""

        return long_term_info

    # 写入指定用户的长期记忆
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
            value={"data": memory_info},
        )
        logger.info(f"##########write_long_term_info: {result}")

        return "store saved"

    config = {
        "configurable": {
            "thread_id": "97",
            "user_id": "user_001",
        }
    }
    user_id = config["configurable"]["user_id"]

    # 写入长期记忆，这里是写入用户偏好的名称
    write_long_term_info(user_id, "PatrickNickName")

    raw_question = "今天天气怎么样？"
    # 从长期记忆中，把 name 拿出来（因为写入的时候只写入 name）
    name = read_long_term_info(user_id)

    messages = chat_prompt.format_messages(question=raw_question, name=name)
    # 取出消息列表中的最后一条消息,通常对应 user(用户)消息,作为本轮要发送给 Agent 的用户提示
    human_msg = messages[-1]
    # 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
    print(f'用户的问题是: {human_msg.content} \n')
    # 将用户提示内容写入日志,方便后续排查问题或重现对话
    logger.info(f"用户的问题是: {human_msg.content}")

    response = run_with_hitl_invoke(
        agent=agent,
        user_content=human_msg.content,
        config=config,
        context=Context(user_id="user_001"),
    )

    # 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
    print(f"Agent最终回复是: {response['structured_response']} \n\n")
    # 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
    logger.info(f"Agent最终回复是: {response['structured_response']}")





