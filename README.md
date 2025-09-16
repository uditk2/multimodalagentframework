# Multimodal Agent Framework

A Python framework that provides an abstraction layer for different AI model connectors (OpenAI, Claude, Azure) with unified interfaces for chat interactions, tool calling, and token management.

## Features

- **Unified Interface**: Work with OpenAI, Claude, and Azure AI models through a consistent API
- **Multimodal Support**: Handle both text and image inputs seamlessly
- **Tool Calling**: Automatic function schema generation and execution
- **Token Management**: Built-in cost tracking and token usage monitoring
- **Conversation Review**: Optional reviewer system for conversation quality control

## Installation

```bash
pip install multimodal-agent-framework
```

## Quick Start

```python
from multimodal_agent_framework import MultiModalAgent, OpenAIConnector, get_openai_client

# Initialize OpenAI connector
client = get_openai_client()  # Requires OPENAI_API_KEY environment variable
connector = OpenAIConnector(client)

# Create agent
agent = MultiModalAgent(
    connector=connector,
    system_prompt="You are a helpful assistant."
)

# Start conversation
response = agent.start_conversation("Hello, how are you?")
print(response)

# Continue conversation
response = agent.add_message("What's the weather like?")
print(response)
```

## Supported Models

### OpenAI
- GPT-4o, GPT-4o-mini
- o1, o1-mini, o1-preview
- o3-mini
- Search-enabled models

### Claude (Anthropic)
- Claude 3.5 Sonnet
- Claude 3 Opus, Sonnet, Haiku

### Azure AI
- Azure OpenAI models
- Azure AI Inference

## Architecture

### Core Components

- **Connectors** (`connectors.py`): Base `Connector` class and provider-specific implementations
- **MultiModalAgent** (`multimodal_agent.py`): High-level agent orchestration with conversation management
- **Helper Functions** (`helper_functions.py`): Utilities for file operations and content management
- **Function Schema Generator** (`function_schema_generator.py`): Automatic JSON schema generation for tool calling

## Advanced Usage

### Tool Calling

```python
def get_weather(location: str) -> str:
    \"\"\"Get weather information for a location\"\"\"
    return f"The weather in {location} is sunny and 75°F"

# Register function with agent
agent.register_function(get_weather)

# Agent can now call the function
response = agent.add_message("What's the weather in New York?")
```

### Image Processing

```python
# Send image with text
image_data = {"data": base64_image, "img_fmt": "png"}
response = agent.add_message(
    text="Describe this image",
    base64_image=image_data
)
```

### Token Usage Monitoring

```python
def token_callback(tokens):
    print(f"Tokens used: {tokens}")

agent = MultiModalAgent(
    connector=connector,
    system_prompt="You are a helpful assistant.",
    token_callback=token_callback
)
```

## Development

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Installation for Development

```bash
git clone <repository-url>
cd multimodal-agent-framework
pip install -e .
```

### Running Examples

```python
from examples.response_summary_agent import ResponseSummaryAgent

summary_agent = ResponseSummaryAgent()
summary = summary_agent.generate_summary("Your text to summarize here")
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.