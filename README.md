# Multimodal Agent Framework

A Python framework providing unified interfaces for AI model providers (OpenAI, Claude, Azure) with conversation management, tool calling, and persistent storage.

## Features

- **Unified Interface**: Consistent API across OpenAI, Claude, and Azure AI models
- **Multimodal Support**: Text and image inputs with seamless processing
- **Tool Calling**: Automatic function schema generation and execution
- **Conversation Persistence**: Save and restore conversations with multiple storage backends (File, AWS S3)
- **Agent Handoff**: Continue conversations between different AI models
- **Token Management**: Built-in cost tracking and usage monitoring

## Installation

```bash
pip install multimodal-agent-framework
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# For OpenAI GPT models (GPT-4o, o1, GPT-5, etc.)
OPENAI_API_KEY=your_openai_api_key_here

# For Claude models (Claude 4 Sonnet, Claude 3.5, etc.)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Azure OpenAI services
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_api_key_here

# For AWS S3 conversation storage
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AGENT_CONVERSATIONS_BUCKET=your-s3-bucket-name
AGENT_CONVERSATIONS_FOLDER=conversations
```

## Quick Start

```python
from multimodal_agent_framework import MultiModalAgent, OpenAIConnector, get_openai_client

# Create agent
agent = MultiModalAgent(
    connector=OpenAIConnector(get_openai_client()),
    system_prompt="You are a helpful assistant."
)

# Start conversation
response = agent.start_conversation("Hello, how are you?")
print(response)
```

## Conversation Persistence

Save and restore agent conversations with multiple storage backends:

### File Storage (Local)
```python
from multimodal_agent_framework.conversation_manager import AgentConversationManager, AgentConversation
from multimodal_agent_framework.conversation_manager.storage import FileStorage

# Create storage and manager
storage = FileStorage(base_path='./conversations')
manager = AgentConversationManager(storage=storage)

# Create and save conversation
conversation = AgentConversation(
    agent_name='my_agent',
    chat_history=[
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi there!'}
    ],
    metadata={'topic': 'greeting'}
)

manager.save_conversation('user123', 'my_agent', conversation, 'chat456')

# Load conversation later
loaded = manager.load_conversation('user123', 'my_agent', 'chat456')
```

### AWS S3 Storage
```python
from multimodal_agent_framework.conversation_manager.storage import S3Storage

# Create S3 storage (requires AWS credentials in environment)
storage = S3Storage(
    bucket_name='my-conversations',
    conversations_folder='agent_chats'
)
manager = AgentConversationManager(storage=storage)

# Same API as file storage
manager.save_conversation('user123', 'my_agent', conversation, 'chat456')
```

### Agent Handoff with Persistence
```python
# Start with OpenAI agent
openai_agent = MultiModalAgent(
    connector=OpenAIConnector(get_openai_client()),
    system_prompt="You are a technical advisor."
)

response, chat_history = openai_agent.execute_user_ask("Explain microservices")

# Save conversation
conversation = AgentConversation(
    agent_name='technical_advisor',
    chat_history=chat_history,
    metadata={'topic': 'microservices'}
)
manager.save_conversation('user123', 'technical_advisor', conversation, 'session1')

# Load and continue with Claude
loaded = manager.load_conversation('user123', 'technical_advisor', 'session1')
claude_agent = MultiModalAgent(
    connector=ClaudeConnector(get_claude_client()),
    system_prompt="You are a code reviewer."
)

# Continue conversation with loaded history
response, updated_history = claude_agent.execute_user_ask(
    "Review the microservices approach",
    chat_history=loaded.chat_history
)
```

## Advanced Features

### Tool Calling
Use `generate_function_schema` to convert Python callables into the tool schema expected by the connectors and pass the resulting list through the `tools` argument of `execute_user_ask`.
```python
from multimodal_agent_framework import generate_function_schema

def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return {"text": f"The weather in {location} is sunny and 75°F"}

tools = [generate_function_schema(get_weather)]

response, updated_history = agent.execute_user_ask(
    user_input="What's the weather in New York?",
    tools=tools,
    model="gpt-4o-mini"
)
print(response)
```

### Multimodal Input
```python
# Process image with text
response = agent.add_message(
    text="Describe this image",
    base64_image={"data": base64_image, "img_fmt": "png"}
)
```

### Token Monitoring
```python
agent = MultiModalAgent(
    connector=connector,
    system_prompt="You are a helpful assistant.",
    token_callback=lambda tokens: print(f"Used: {tokens} tokens")
)
```

## Supported Models

**OpenAI**: GPT-4o, o1/o3 series, GPT-5 series, search-enabled models
**Claude**: Claude 4 Sonnet, Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
**Azure**: Azure OpenAI models, Azure AI Inference

## Development

### Setup
```bash
git clone <repository-url>
cd multimodalagentframework
pip install -r requirements.txt
pip install -e .[dev]  # Install with development dependencies
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=multimodal_agent_framework

# Run specific test file
pytest tests/test_function_schema_generator.py
```

### Code Quality
```bash
# Format code
black .

# Check formatting (run before committing)
black --check .

# Type checking
mypy multimodal_agent_framework/
```

## License

MIT License
