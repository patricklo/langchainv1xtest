import os
import asyncio
import json
import uuid
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import  AsyncPostgresStore
from langchain.agents.structured_output import ToolStrategy
from langgraph.types import Command

from utils.config import Config

from utils.llms import get_llm
from utils.tools import get_tools
from utils.models import Context,ResponseFormat
from utils.logger import LoggerManager

async def main():
    logger = LoggerManager.get_logger()
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    tools, hitl_middleware = await get_tools()
    system_prompt = PromptTemplate.from_file(
        template_file=Config.SYSTEM_PROMPT_TMPL,
        encoding="utf-8",
    ).template
    human_prompt = PromptTemplate.from_file(
        template_file=Config.HUMAN_PROMPT_TMPL,
        encoding="utf-8",
    ).template
    # chat_prompt = ChatPromptTemplate.from_messages(
    #     [
    #     SystemMessagePromptTemplate.from_template(system_prompt),
    #     HumanMessagePromptTemplate.from_template(human_prompt),
    #     ]
    # )
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    async def run_with_hitl_invoke(agent, user_content: str, config: dict, context: Context):
        # 第一次调用代理，发送用户消息，发起对话/执行
        result = await agent.ainvoke(
            # 将用户输入包装为消息列表，角色为 user，内容为 user_content
            {"messages": [{"role": "user", "content": user_content}]},
            # 传入配置，用于控制代理行为
            config=config,
            # 传入上下文对象，维持对话或执行环境
            context=context,
        )

        # 使用循环持续处理所有中断情况
        # 只要返回结果中包含 "__interrupt__" 字段，就说明有人工审核步骤需要处理
        # 支持同时有多个工具需要审批，也支持工具链式调用，如先 get_location 再 get_weather
        while "__interrupt__" in result:
            # 从结果中取出所有中断请求列表
            hitl_requests = result["__interrupt__"]
            # 简化处理：只取第一个中断请求
            hitl_req = hitl_requests[0]
            # 从中断请求中取出所有需要人工审核的工具调用请求
            action_requests = hitl_req.value["action_requests"]
            # 从中断请求中取出每个工具调用对应的审核配置
            review_configs = hitl_req.value["review_configs"]

            # 初始化一个决策列表，用来收集对每个工具调用的人工决策
            decisions = []
            # 遍历所有工具调用请求，i 为索引，ar 为单个请求
            for i, ar in enumerate(action_requests):
                # 从请求中取得工具名称
                name = ar.get("name")
                # 兼容两种参数字段写法：优先从 "args" 取，如果没有再从 "arguments" 取
                args = ar.get("args", ar.get("arguments"))

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

                # 如果决策为 edit，表示需要修改工具调用的参数
                if decision_type == "edit":
                    # 让用户输入修改后的参数，要求为 JSON 字符串格式
                    new_args_str = input("请输入修改后的 args(JSON 字符串)：").strip()
                    # 将 JSON 字符串解析为 Python 字典对象
                    new_args = json.loads(new_args_str)

                    # 构造编辑后的工具调用动作，只允许修改 args，保持 name 不变
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
            # Command(resume=...) 告诉代理根据这些人工决策继续往下跑
            result = await agent.ainvoke(
                Command(resume={"decisions": decisions}),
                config=config,
                context=context,
            )

        # 当跳出 while 循环时，说明已经没有新的中断需要审核
        # 此时 result 即为整个流程的最终结果，直接返回
        return result

    async with AsyncConnectionPool(
        conninfo=Config.DB_URI,
        min_size=Config.MIN_SIZE,
        max_size=Config.MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        # 延迟打开连接池->open=False
        open=False
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("saver inited")
        store = AsyncPostgresStore(pool)
        await store.setup()
        logger.info("store inited")

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

        # 读取长期记忆内容
        async def read_long_term_info(user_id: str):
            # 定义用于查询的命名空间，通常用 (类别, 用户ID) 这种形式做分区
            namespace = ("memories", user_id)

            # 在指定命名空间下搜索所有记忆数据，这里 query="" 意味着不带语义过滤，取全部或由实现决定
            memories = await store.asearch(namespace, query="")

            # 如果没有查到任何结果（返回 None），这里先占位，不做额外处理
            if memories is None:
                pass

            # 如果有 memories，则从每个条目中提取 value["data"] 字段，并用空格拼接成一个长字符串；
            # 条件判断确保 d.value 是字典且包含 "data" 键；
            # 如果 memories 为 None 或空列表，则整体结果为 ""（空字符串）
            long_term_info = " ".join(
                [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
            ) if memories else ""

            # 打日志，记录成功获取到的用户长期记忆，以及拼接后的文本长度，方便排查与监控
            logger.info(
                f"成功获取用户ID: {user_id} 的长期记忆，内容长度: {len(long_term_info)} 字符，内容: {long_term_info} 字符")

            # 返回拼接好的长期记忆文本
            return long_term_info

        # 写入指定用户的长期记忆
        async def write_long_term_info(user_id: str, memory_info: str):
            # 定义命名空间，用于把某个用户的记忆归到 ("memories", user_id) 这个层级路径下
            namespace = ("memories", user_id)

            # 生成一个全局唯一的记忆ID，作为该条记忆在命名空间内的 key
            memory_id = str(uuid.uuid4())

            # 调用存储接口，将记忆写入存储：
            # - namespace: 命名空间，用于分层组织数据
            # - key: 该命名空间下唯一的主键（这里用随机UUID）
            # - value: 实际存储内容，这里用 dict 包一层，字段名为 "data"
            result = await store.aput(
                namespace=namespace,
                key=memory_id,
                value={"data": memory_info}
            )

            # 记录日志，说明为该用户成功写入了一条长期记忆，并打印记忆ID，便于排查
            logger.info(f"成功为用户ID: {user_id} 存储记忆，记忆ID: {memory_id}")

            # 返回给上层一个简单的成功提示文案
            return "记忆存储成功"

        # 写入长期记忆
        await write_long_term_info("user_001", "南哥")

        config = {
            "configurable": {
                "thread_id": "210",
                "user_id": "user_001",
            }
        }
        raw_question = "智能体UI-TARS-2是哪一家发布的？返回2篇文章并给出文章的标题、链接、发布者"
        # 获取长期记忆内容，用户偏好设置(用户名称)
        name = await read_long_term_info("user_001")
        print(f"name=!!!{name}")
        messages = chat_prompt.format_messages(question=raw_question, name=name)
        human_msg = messages[-1]
        print(f"human_msg#######={human_msg}")
        response = await run_with_hitl_invoke(
            agent=agent,
            user_content=human_msg.content,
            config=config,
            context=Context(user_id="user_001")
        )

        # 从 response 字典中取出 "messages" 列表的最后一条消息的内容，作为代理（Agent）的最终回复结果
        result = response["messages"][-1].content
        # 在控制台打印代理最终回复内容，方便调试和查看
        print(f"Agent最终回复!!!!是: {result} \n")
        # 通过日志记录器写入一条信息级别的日志，内容为代理最终回复，方便后续排查和追踪
        logger.info(f"Agent最终回复!!!!!是: {result}")
if __name__ == "__main__":
    asyncio.run(main())
