# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-modal agent framework that provides an abstraction layer for different AI model connectors (OpenAI, Claude, Azure) with unified interfaces for chat interactions, tool calling, and token management. The framework is designed to be published as a standalone Python package.

## Architecture

### Core Components

- **`connectors.py`** - Base `Connector` class and implementations (`OpenAIConnector`, `ClaudeConnector`, `AzureOpenSourceConnector`) for different AI model providers
- **`multimodal_agent.py`** - `MultiModalAgent` class that orchestrates chat interactions with support for text/image inputs, tool calling, and conversation review
- **`helper_functions.py`** - Client factory functions for different AI providers (OpenAI, Claude, Azure)
- **`function_schema_generator.py`** - Generates JSON schemas from Python function objects for tool calling
- **`logging_config.py`** - Simple logging configuration for the package
- **`token_tracker.py`** - Token usage tracking implementation

### Key Design Patterns

1. **Connector Pattern**: Each AI provider has its own connector that implements a common interface (`create_message`, `get_response`, `_adapt_chat_history`, etc.)

2. **Agent Abstraction**: `MultiModalAgent` provides a high-level interface that works with any connector, handling system prompts, chat history, and optional reviewers

3. **Tool Integration**: Functions can be automatically converted to tool schemas and executed through the `make_tool_calls` method

4. **Token Management**: Built-in cost tracking and token usage monitoring across all providers using internal `TokenUsageTracker`

### Package Structure

This is a self-contained Python package with no external `app.*` dependencies. All utilities are contained within the `multimodal_agent_framework` module.

## Development Commands

### Package Development

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run a specific test file
python -m pytest tests/framework_test.py

# Build the package
python -m build

# Format code
black multimodal_agent_framework/

# Type checking
mypy multimodal_agent_framework/
```

## Key Implementation Notes

1. **Model Support**: The OpenAI connector supports various models including GPT-4o, o1, o3, and search-enabled models with different parameter handling

2. **Multimodal Input**: Agents support both text and base64-encoded images through the `create_message` method

3. **Tool Calling**: Functions are automatically registered and can be called by AI models through the tool calling interface

4. **Error Handling**: Custom exception `NoTokensAvailableError` for token limit scenarios

5. **Cost Tracking**: Real-time cost calculation based on model-specific token pricing

## Working with Connectors

When adding new AI provider support:
1. Inherit from the base `Connector` class
2. Implement all abstract methods (`create_message_internal`, `get_response`, `_adapt_chat_history`, etc.)
3. Handle provider-specific message formats and API parameters
4. Update token cost calculations for the new provider

## Working with Agents

When creating new agent types:
1. Instantiate `MultiModalAgent` with appropriate connector
2. Provide system prompts and optional reviewer for conversation quality control
3. Use `filter_chat_history` to control context scope
4. Implement token callbacks for usage monitoring