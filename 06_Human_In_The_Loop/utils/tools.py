from langchain.tools import tool, ToolRuntime
from langgraph.config import get_stream_writer
# Human-in-the-loop 中间件，用于在工具调用前做人工审查
from langchain.agents.middleware import HumanInTheLoopMiddleware
# 从当前包中导入 Context 模型，用于在工具运行时携带用户等上下文信息
from .models import Context
# 从当前包中导入 LoggerManager，用于获取日志记录器实例
from .logger import LoggerManager

logger = LoggerManager.get_logger()

def get_tools():

    ###################### 1、定义工具 ######################

    # 使用 @tool 装饰器注册一个工具，工具名为 "list_refund_reasons"，描述为“为指定的城市获取天气。”
    # 该工具接收城市名称，并返回该城市的天气描述字符串
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
        logger.info(f"#####config: {runtime.config}")
        #user_id = runtime.config.user_id #从config中取user_id也行
        logger.info(f"###########get_user_location: {user_id}")
        # 简单的示例映射：user_id 为 "1" 时返回“北京”，否则返回“上海”
        return "北京" if user_id == "user_001" else "上海"

    # 将定义好的两个工具函数封装到列表中，作为 Agent 可调用的工具集合
    tools = [
        get_weather_for_location,
        get_user_location
    ]
    # 记录当前获取到的工具列表，方便在调试或运行中查看已注册的工具
    logger.info(f"获取并提供的工具列表: {tools} ")


    ##################### 2. Human-in-the-loop 策略配置 #####################

    #构建一个工具中断配置字典，键为工具名称，值为该工具的审核配置信息
    interrupt_on = {
        'get_weather_for_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '调用get_weather_for_location 工具需要人工审批。请输入 approve/reject/edit'
        },
        'get_user_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '调用 get_user _location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        }
    }
    logger.info(f"需要人工审批的工具有：: {interrupt_on} ")

    # 创建一个人工介入循环 （Human-in-the-loop)中间件的实例
    # 该中间件用于在工具调用时拦截并等待人工审核
    hitl_middleware = HumanInTheLoopMiddleware(
        # 传入中断配置dict, 指定哪些工具需要人工审核以及审核规则
        # interrupt_on 包含每个工具的审核配置 （allowed_decisions 和 description)
        interrupt_on=interrupt_on,
        # 设置描述信息的前缀文本
        # 当触发人工审核时，会在提示信息前添加此前缀
        description_prefix="工具调用需要人工审批"
    )

    return tools, hitl_middleware

