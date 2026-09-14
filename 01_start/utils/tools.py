from langchain.tools import tool, ToolRuntime
from .models import Context
from .logger import LoggerManager

logger = LoggerManager.get_logger()


def get_tools():

    @tool("get_weather_for_location", description="为指定的城市获取天气")
    def get_weather_for_location(city: str) -> str:
        return f"{city}的天气是晴天！"


    #使用@tool 注册第二个工具，工具名为 “get_user_location" 描述为”根据用户ID检索用户信息
    # 该工具通过 ToolRuntime获取上下文中的用户信息，从而推断用户所在城市
    @tool("get_user_location", description="根据用户ID检索用户信息。")
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        user_id = runtime.context.user_id
        return "北京" if user_id == "1" else "上海"

    tools = [get_weather_for_location, get_user_location]

    logger.info(f"获取可提供的工具列表：{tools}")
    return tools