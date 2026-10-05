#从dataclasses模块 导入 dadtaclass装饰器，用于简化数据类的定义

from dataclasses import dataclass
from typing import NotRequired

from langchain.agents import AgentState
from pydantic import BaseModel


@dataclass
class Context:
    user_id: str

#@dataclass
class ResponseFormat(BaseModel):
    punny_response: str
    weather_response: str

class MyState(AgentState):
    call_count: NotRequired[int]
    preferred_city: NotRequired[str]