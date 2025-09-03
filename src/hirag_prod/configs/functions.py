from typing import Dict, Optional

from hirag_prod.configs.config_manager import ConfigManager
from hirag_prod.configs.envs import Envs
from hirag_prod.configs.hi_rag_config import HiRAGConfig


def initialize_config_manager(config_dict: Optional[Dict] = None) -> None:
    ConfigManager(config_dict)


def get_config_manager() -> ConfigManager:
    return ConfigManager()


def get_hi_rag_config() -> HiRAGConfig:
    return ConfigManager().hi_rag_config


def get_envs() -> Envs:
    return ConfigManager().envs
