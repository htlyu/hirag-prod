from typing import Literal

from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM configuration"""

    service_type: Literal["openai", "local"]

    base_url: str
    api_key: str

    model_name: str = "gpt-4o"
    max_tokens: int = 16000
    timeout: float = 30.0

    entry_point: str = "/v1/chat/completions"

    class Config:
        alias_generator = lambda x: f"llm_{x}".upper()
        populate_by_name = True
        extra = "ignore"
