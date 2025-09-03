import threading
from typing import Dict, List, Optional

from hirag_prod.configs.embedding_config import EmbeddingConfig
from hirag_prod.configs.envs import Envs
from hirag_prod.configs.hi_rag_config import HiRAGConfig
from hirag_prod.configs.llm_config import LLMConfig


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "ConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dict: Optional[Dict] = None):
        if getattr(self, "_created", False):
            return

        self.envs: Envs = Envs(**config_dict if config_dict else {})
        self.hi_rag_config = HiRAGConfig(**self.envs.model_dump())
        self.embedding_config: EmbeddingConfig = EmbeddingConfig(
            **self.envs.model_dump()
        )
        self.llm_config: LLMConfig = LLMConfig(**self.envs.model_dump())

        self.supported_languages: List[str] = ["en", "cn-s", "cn-t"]
        self.language: str = (
            config_dict["language"]
            if (config_dict and "language" in config_dict)
            else self.envs.HI_RAG_LANGUAGE
        )
        if self.language not in self.supported_languages:
            raise ValueError(
                f"Unsupported language {self.language}. Supported languages are {self.supported_languages}"
            )

        self.postgres_url_env: Optional[str] = (
            self.envs.POSTGRES_URL_NO_SSL_DEV
            if self.envs.ENV == "dev"
            else self.envs.POSTGRES_URL_NO_SSL
        )
        self.postgres_url_async: Optional[str] = self.postgres_url_env
        self.postgres_url_sync: Optional[str] = self.postgres_url_env
        # Replace postgres:// with postgresql+asyncpg:// for async connections
        if self.postgres_url_env.startswith("postgres://"):
            self.postgres_url_async = self.postgres_url_env.replace(
                "postgres://", "postgresql+asyncpg://", 1
            )
            self.postgres_url_sync = self.postgres_url_env.replace(
                "postgres://", "postgresql://", 1
            )
        elif self.postgres_url_env.startswith("postgresql://"):
            self.postgres_url_async = self.postgres_url_env.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        elif self.postgres_url_env.startswith("postgres+asyncpg://"):
            self.postgres_url_async = self.postgres_url_env.replace(
                "postgres+asyncpg://", "postgresql+asyncpg://", 1
            )
            self.postgres_url_sync = self.postgres_url_env.replace(
                "postgres+asyncpg://", "postgresql://", 1
            )
        elif self.postgres_url_env.startswith("postgresql+asyncpg://"):
            self.postgres_url_sync = self.postgres_url_env.replace(
                "postgresql+asyncpg://", "postgresql://", 1
            )

        self._created: bool = True
