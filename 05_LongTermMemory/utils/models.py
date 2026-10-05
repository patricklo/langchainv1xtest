from dataclasses import dataclass


@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    punny_response: str
    weather_conditions: str | None = None