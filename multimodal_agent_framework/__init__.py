"""
Multi-modal Agent Framework

A framework that provides an abstraction layer for different AI model connectors
(OpenAI, Claude, Azure) with unified interfaces for chat interactions, tool calling,
and token management.
"""

from .connectors import (
    Connector,
    OpenAIConnector,
    ClaudeConnector,
    AzureOpenSourceConnector,
)
from .multimodal_agent import MultiModalAgent, Reviewer, NoTokensAvailableError
from .helper_functions import (
    get_openai_client,
    get_claude_client,
    get_azure_opensource_client,
    get_openai_azure_client,
    get_openai_azure_dalle_client,
)
from .function_schema_generator import generate_function_schema

__all__ = [
    "Connector",
    "OpenAIConnector",
    "ClaudeConnector",
    "AzureOpenSourceConnector",
    "MultiModalAgent",
    "Reviewer",
    "NoTokensAvailableError",
    "get_openai_client",
    "get_claude_client",
    "get_azure_opensource_client",
    "get_openai_azure_client",
    "get_openai_azure_dalle_client",
    "generate_function_schema",
]
