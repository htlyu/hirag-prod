import pytest
from dotenv import load_dotenv

from hirag_prod.configs.functions import initialize_config_manager
from hirag_prod.resources.functions import (
    get_resource_manager,
    initialize_resource_manager,
)

load_dotenv("../.env", override=True)


@pytest.mark.asyncio
async def test_initialization_and_cleanup():
    initialize_config_manager()
    await initialize_resource_manager()
    print("✅ Test initialization successfully!")

    await get_resource_manager().cleanup()
    print("✅ Test cleanup successfully!")
