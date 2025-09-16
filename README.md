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

## Environment Variables

The framework requires API keys and configuration for different AI providers. Create a `.env` file in your project root or set these environment variables:

### Required for OpenAI
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Required for Claude (Anthropic)
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Required for Azure OpenAI
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
```

### Required for Azure OpenAI DALL-E
```bash
AZURE_OPENAI_DALLE_ENDPOINT=https://your-dalle-resource.openai.azure.com/
AZURE_OPENAI_DALLE_API_KEY=your_azure_dalle_api_key_here
```

### Required for Azure AI Inference (Open Source Models)
```bash
AZURE_OPENSOURCE_ENDPOINT=https://your-inference-endpoint.inference.ai.azure.com
AZURE_OPENSOURCE_API_KEY=your_azure_inference_api_key_here
```

### Example .env file
```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-...

# Anthropic Claude Configuration
ANTHROPIC_API_KEY=sk-ant-...

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com/
AZURE_OPENAI_API_KEY=abc123...

# Azure OpenAI DALL-E Configuration (optional, for image generation)
AZURE_OPENAI_DALLE_ENDPOINT=https://mydalle.openai.azure.com/
AZURE_OPENAI_DALLE_API_KEY=xyz789...

# Azure AI Inference Configuration (optional, for open source models)
AZURE_OPENSOURCE_ENDPOINT=https://myinference.inference.ai.azure.com
AZURE_OPENSOURCE_API_KEY=def456...
```

**Important:**
- Only set the environment variables for the providers you plan to use
- Keep your API keys secure and never commit them to version control
- The framework uses `python-dotenv` to automatically load `.env` files

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

- Python 3.9+
- Dependencies are managed via `pyproject.toml`

### Installation for Development

```bash
git clone <repository-url>
cd AgentFramework

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

**Note:** If you already have the package installed from PyPI (test or production), `pip install -e .` will replace it with your local development version. You can switch back by running `pip install multimodal-agent-framework` or uninstalling and reinstalling from PyPI.

### Development Commands

```bash
# Run tests
pytest tests/

# Run a specific test
python -m pytest tests/framework_test.py

# Format code
black multimodal_agent_framework/

# Type checking
mypy multimodal_agent_framework/

# Build package
python -m build
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