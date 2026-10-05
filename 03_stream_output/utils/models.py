from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class Context:
    user_id: str

# 使用 @dataclass 定义 Agent 的结构化响应数据模型
@dataclass
class ResponseFormat:
    """查完天气后必须调用此工具给出最终答案，不要用普通文本结束。
    punny_response: 带冷笑话的主要回复。
    weather_conditions: 天气情况；没有天气信息时填 null。
    """
    # punny_response 为必填字段，用于存放包含谐音梗 / 冷笑话的主要回复内容
    # 该字段通常会由 LLM 根据系统提示词和用户问题生成
    punny_response: str
    # weather_conditions 为可选字段，用于补充与天气相关的有趣信息
    # 类型注解为 str | None，表示可以是字符串或 None
    weather_conditions: str | None = None
