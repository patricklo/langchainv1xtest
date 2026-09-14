# 从 dataclasses模块导入 dataclass 装饰器，用于简化数据类的定义
from dataclasses import dataclass

# 使用 @dataclass 定义运行时上下文数据模型，用于在agent/工具执行时传递用户相关信息

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    """最终回复给用户的结构化结果。完成天气查询后必须调用此工具输出答案。"""

    punny_response: str
    weather_conditions: str | None = None