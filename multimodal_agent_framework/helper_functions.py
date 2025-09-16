import os
from openai import OpenAI
import anthropic
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()
from .logging_config import get_logger
from openai import AzureOpenAI

logger = get_logger()


def get_openai_client():
    """Create and return an OpenAI client."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_azure_opensource_client():
    """Create and return an Azure OpenSource client."""
    return ChatCompletionsClient(
        endpoint=os.getenv("AZURE_OPENSOURCE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_OPENSOURCE_API_KEY")),
    )


def get_claude_client():
    """Create and return an Anthropic Claude client."""
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_openai_azure_client():
    """Create and return an Azure OpenAI client."""
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
    )
    return client


def get_openai_azure_dalle_client():
    """Create and return an Azure OpenAI DALL-E client."""
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_DALLE_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_DALLE_API_KEY"),
        api_version="2024-02-01",
    )
    return client
