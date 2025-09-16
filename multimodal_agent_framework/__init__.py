"""
Multi-modal Agent Framework

A framework that provides an abstraction layer for different AI model connectors
(OpenAI, Claude, Azure) with unified interfaces for chat interactions, tool calling,
and token management.
"""

from .connectors import Connector, OpenAIConnector, ClaudeConnector, AzureConnector
from .multimodal_agent import MultiModalAgent
from .helper_functions import get_openai_client
from .function_schema_generator import generate_schema_from_function

__version__ = "0.1.0"
__all__ = [
    "Connector",
    "OpenAIConnector",
    "ClaudeConnector",
    "AzureConnector",
    "MultiModalAgent",
    "get_openai_client",
    "generate_schema_from_function",
]
