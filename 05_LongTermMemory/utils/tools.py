from langchain.tools import tool, ToolRuntime
from langgraph.config import get_stream_writer
from .models import Context
from .logger import LoggerManager

logger = LoggerManager.get_logger()

def get_tools():

    @tool("get_weather_for_location", description="为指定的城市获取天气。")
    def get_weather_for_location(city: str) -> str:
        writer = get_stream_writer()
        writer(f"正在查找城市数据：{city}")
        writer(f"已获取城市数据：{city}")
        return f"{city}的天气是晴天"

    @tool("get_user_location", description="根据用户 ID 检索用户信息。")
    #工具通过 ToolRuntime 获取上下文中的用户信息，从而推断用户所在城市
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        user_id = runtime.context.user_id
        return "北京" if user_id == "user_001" else "上海"

    tools = [
        get_weather_for_location,
        get_user_location
    ]

    return tools