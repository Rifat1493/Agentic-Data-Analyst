from langchain.chat_models import init_chat_model
from langchain_qwq import ChatQwen

from src import config


def get_foundation_model_v2():
    """Initialize and return the foundation model for the agent."""

    llm = init_chat_model(
        model=config.FOUNDATION_MODEL,
        base_url=config.DASHSCOPE_BASE_URL,
        api_key=config.DASHSCOPE_API_KEY,
    )
    return llm


def get_foundation_model():
    """Initialize and return the foundation model for the agent."""

    llm = ChatQwen(
        model=config.FOUNDATION_MODEL,
        api_key=config.DASHSCOPE_API_KEY,
        # other params...
    )
    return llm