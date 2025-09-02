from typing import Dict, Literal, Optional

from pydantic_settings import BaseSettings


class Envs(BaseSettings):
    ENV: Literal["dev", "prod"] = "dev"
    HI_RAG_LANGUAGE: str = "en"
    POSTGRES_URL_NO_SSL: str
    POSTGRES_URL_NO_SSL_DEV: str
    POSTGRES_TABLE_NAME: str = "KnowledgeBaseJobs"
    POSTGRES_SCHEMA: str = "public"
    REDIS_URL: str = "redis://redis:6379/2"
    REDIS_KEY_PREFIX: str = "hirag"
    REDIS_EXPIRE_TTL: int = 3600 * 24
    EMBEDDING_DIMENSION: int

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

    def __init__(self, values: Optional[Dict]):
        if values:
            super().__init__(**values)
        else:
            super().__init__()
