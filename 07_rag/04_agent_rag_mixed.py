import json
import os
import uuid
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain.agents.structured_output import ToolStrategy
from langgraph.types import Command

from utils.config import Config
from utils.llms import get_llm
from utils.tools import get_tools
from utils.models import Context, ResponseFormat
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

tools, hitl_middleware = get_tools()

system_prompt = PromptTemplate.from_file(
    template_file=Config.SYSTEM_PROMPT_TMPL,
    encoding="utf-8",
).template
human_prompt = PromptTemplate.from_file(
    template_file=Config.HUMAN_PROMPT_TMPL,
    encoding="utf-8",
).template

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
     ("human", human_prompt)]
)

# 封装一个带 HITL 审核的调用函数，问答统一用它
# 定义一个带有人在环路（HITL）机制的运行函数
# agent: 代理对象
# user_content: 用户输入内容
# config: 运行配置字典
# context: 上下文对象
def run_with_hitl_invoke(agent, user_content: str, config: dict, context: Context):
    # 第一次调用代理agent, 发送用户消息，发起对话/执行
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
        context=context,
    )

    # 使用循环持续处理所有中断情况
    # 只要返回结果中包含 “__interrupt__" 字段，就说明人工审核步骤需要处理
    # 支持同时有多个工具需要审批，也支持工具链式调用，如先 get_location 再 get_weather
    while "__interrupt__" in result:
        # 从结果中取出所有中断请求列表
        hitl_requests = result["__interrupt__"]
        # 简化处理： 只取第一个中断请求
        first_hitl_req = hitl_requests[0]
        # 从中断请求中取出所有需要人工审核的工具调用请求
        action_requests = first_hitl_req.value["action_requests"]
        # 从中断请求中取出每个工具调用对应的审核配置
        review_configs = first_hitl_req.value["review_configs"]

        # 初始化一个决策列表，用来收集对每个工具调用的人工决策
        decisions = []
        # i 为索引， ar 为单个请求
        for i, ar in enumerate(action_requests):
            name = ar["name"]
            args = ar.get("args",ar.get("arguments"))

            # 从对应的审核配置中获取当前工具允许的决策类型
            allowed = review_configs[i]["allowed_decisions"]

            # 在控制台提示检测到需要人工审核的工具调用
            print("\n=== 检测到工具调用需要审核 ===")
            # 打印工具名称
            print(f"工具名：{name}")
            # 打印工具调用参数
            print(f"参数：{args}")
            # 打印允许的决策类型列表
            print(f"允许的决策类型：{allowed}")

            # 提示人工输入决策类型，并去掉前后空格
            decision_type = input("请输入决策(approve/edit/reject)：").strip()

            # 如果输入不在允许的决策列表中，则循环要求重新输入
            while decision_type not in allowed:
                decision_type = input(f"非法决策，请重新输入({','.join(allowed)})：").strip()

            if decision_type == "edit":
                # 让用户输入修改后的参数，要求为 JSON 字符串格式
                new_args_str = input("请输入修改后的 args(JSON): ").strip()
                # 将新 JSON 字符串解析为 Python 字典对象
                new_args = json.loads(new_args_str)

                # 构造编辑后的工具调用动作，只允许修改args，保持 name 不变
                edited_action = {
                    "name": name,
                    "args": new_args,
                }
                decisions.append({
                    "type": "edit",
                    "edited_action": edited_action,
                })
            else:
                decisions.append({"type": decision_type})

        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            context=context,
        )

    # 当跳出 while 循环时，说明已经没有新的中断需要审核
    # 此时 result 即为整个流程的最终结果，直接返回
    return result

with(
    PostgresSaver.from_conn_string(Config.DB_URI) as checkpointer,
    PostgresStore.from_conn_string(Config.DB_URI) as store
):
    checkpointer.setup()
    store.setup()
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[
            SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3)),
            hitl_middleware
        ],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
        store=store,
    )

    def read_long_term_info(user_id: str):
        namespace = ("memories", user_id)
        memories = store.search(namespace, query="")
        if memories is None:
            pass

        long_term_info = " ".join(
            [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
        ) if memories else ""

        return long_term_info

    def write_long_term_info(user_id: str, memory_info: str):
        # 定义命名空间，用于把某个用户的记忆归到 ("memories", user_id) 这个层级路径下
        namespace = ("memories", user_id)

        # 生成一个全局唯一的记忆ID，作为该条记忆在命名空间内的 key
        memory_id = str(uuid.uuid4())

        # 调用存储接口，将记忆写入存储：
        # - namespace: 命名空间，用于分层组织数据
        # - key: 该命名空间下唯一的主键（这里用随机UUID）
        # - value: 实际存储内容，这里用 dict 包一层，字段名为 "data"
        result = store.put(
            namespace=namespace,
            key=memory_id,
            value={"data": memory_info}
        )

        # 记录日志，说明为该用户成功写入了一条长期记忆，并打印记忆ID，便于排查
        logger.info(f"成功为用户ID: {user_id} 存储记忆，记忆ID: {memory_id}")

        # 返回给上层一个简单的成功提示文案
        return "记忆存储成功"

    write_long_term_info("user_001", "patrick")

    config = {
        "configurable":{
            "thread_id": "199",
            "user_id": "user_001",
        }
    }

    raw_question = "调用健康档案查询工具查询张三九的基本信息？"
    name = read_long_term_info("user_001")
    print(f"#####读取长期记忆的名字：{name}")

    messages = chat_prompt.format_messages(question=raw_question, name=name)
    human_msg = messages[-1]
    response = run_with_hitl_invoke(
        agent=agent,
        user_content=human_msg.content,
        config=config,
        context=Context(user_id="user_001"),
    )

    # 从 response 字典中取出 "messages" 列表的最后一条消息的内容，作为代理（Agent）的最终回复结果
    result = response["messages"][-1].content
    # 在控制台打印代理最终回复内容，方便调试和查看
    print(f"Agent最终回复是: {result} \n")
    # 通过日志记录器写入一条信息级别的日志，内容为代理最终回复，方便后续排查和追踪
    logger.info(f"Agent最终回复是: {result}")
























































