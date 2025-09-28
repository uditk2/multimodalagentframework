import pytest
from unittest.mock import Mock
from multimodal_agent_framework.connectors import (
    Connector,
    OpenAIConnector,
    ClaudeConnector,
    AzureOpenSourceConnector,
)
from multimodal_agent_framework.token_tracker import (
    DefaultTokenUsageTracker,
    BaseTokenUsageTracker,
)


class MockClient:
    """Mock client for testing connector initialization."""

    def __init__(self, client_type="openai"):
        self.client_type = client_type


class TestBaseConnector:
    """Test cases for the base Connector class."""

    def test_connector_initialization_success(self):
        """Test successful connector initialization."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        assert connector.client == mock_client
        assert isinstance(connector.token_tracker, DefaultTokenUsageTracker)
        assert connector.get_cost() == 0
        assert connector._tokens == {"input_tokens": 0, "output_tokens": 0}

    def test_connector_initialization_with_custom_tracker(self):
        """Test connector initialization with custom token tracker."""
        mock_client = MockClient()
        custom_tracker = Mock(spec=BaseTokenUsageTracker)
        connector = Connector(mock_client, token_tracker=custom_tracker)

        assert connector.token_tracker == custom_tracker

    def test_connector_initialization_fails_without_client(self):
        """Test that connector initialization fails without client."""
        with pytest.raises(ValueError, match="Client is required"):
            Connector(None)

    def test_validate_arguments_text_only(self):
        """Test argument validation with text only."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        # Should not raise
        connector.validate_arguments("Hello", None)

    def test_validate_arguments_image_only(self):
        """Test argument validation with image only."""
        mock_client = MockClient()
        connector = Connector(mock_client)
        image_data = {"data": "base64data", "img_fmt": "png"}

        # Should not raise
        connector.validate_arguments(None, image_data)

    def test_validate_arguments_both_text_and_image(self):
        """Test argument validation with both text and image."""
        mock_client = MockClient()
        connector = Connector(mock_client)
        image_data = {"data": "base64data", "img_fmt": "png"}

        # Should not raise
        connector.validate_arguments("Hello", image_data)

    def test_validate_arguments_neither_text_nor_image(self):
        """Test that validation fails when neither text nor image provided."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        with pytest.raises(ValueError, match="Either text or image is required"):
            connector.validate_arguments(None, None)

    def test_validate_arguments_text_as_bytes(self):
        """Test that validation fails when text is bytes."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        with pytest.raises(ValueError, match="Text should be a string"):
            connector.validate_arguments(b"bytes text", None)

    def test_validate_arguments_invalid_image_format(self):
        """Test that validation fails with invalid image format."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        with pytest.raises(
            ValueError,
            match="Image should be a dict object containing the fields",
        ):
            connector.validate_arguments("Hello", "not_a_dict")

    def test_execute_function_with_dict_args(self):
        """Test function execution with dictionary arguments."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        def test_func(name, age):
            return {"text": f"{name} is {age} years old"}

        args = {"name": "Alice", "age": 30}
        result = connector._execute_function(test_func, args)

        assert result == {"text": "Alice is 30 years old"}

    def test_execute_function_with_json_string_args(self):
        """Test function execution with JSON string arguments."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        def test_func(x, y):
            return {"result": x + y}

        args = '{"x": 5, "y": 3}'
        result = connector._execute_function(test_func, args)

        assert result == {"result": 8}

    def test_execute_function_with_error(self):
        """Test function execution when function raises an error."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        def error_func():
            raise ValueError("Something went wrong")

        result = connector._execute_function(error_func, {})

        assert "Error executing function error_func" in result["text"]
        assert "Something went wrong" in result["text"]

    def test_execute_function_with_image_response(self):
        """Test function execution that returns image data."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        def image_func():
            return {"image": {"data": "base64imagedata"}}

        result = connector._execute_function(image_func, {})

        # Should replace image data with UUID and store in context
        assert "image" in result
        assert result["image"]["data"] != "base64imagedata"
        # UUID should be stored in context
        uuid_key = result["image"]["data"]
        assert uuid_key in connector._context
        assert connector._context[uuid_key] == "base64imagedata"

    def test_execute_function_with_context_substitution(self):
        """Test function execution with context value substitution."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        # Set up context
        context_key = "test_uuid_123"
        connector._context[context_key] = "actual_image_data"

        def test_func(image_data):
            return {"text": f"Processed: {image_data}"}

        args = {"image_data": context_key}
        result = connector._execute_function(test_func, args)

        assert result == {"text": "Processed: actual_image_data"}

    def test_get_chat_text_content_string(self):
        """Test extracting text content from string message."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        result = connector.get_chat_text_content("Hello world")
        assert result == "Hello world"

    def test_get_chat_text_content_array(self):
        """Test extracting text content from array message."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        message = [{"type": "text", "text": "Hello world"}]
        result = connector.get_chat_text_content(message)
        assert result == "Hello world"

    def test_set_chat_text_content_string(self):
        """Test setting text content for string message."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        result = connector.set_chat_text_content("old text", "new text")
        assert result == "new text"

    def test_set_chat_text_content_array(self):
        """Test setting text content for array message."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        message = [{"type": "text", "text": "old text"}]
        result = connector.set_chat_text_content(message, "new text")
        assert result[0]["text"] == "new text"

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        mock_client = MockClient()
        connector = Connector(mock_client)

        with pytest.raises(NotImplementedError):
            connector.create_message_internal()

        with pytest.raises(NotImplementedError):
            connector.get_response()

        with pytest.raises(NotImplementedError):
            connector._adapt_chat_history([])

        with pytest.raises(NotImplementedError):
            connector._adapt_functions([])

        with pytest.raises(NotImplementedError):
            connector.get_system_message("", "")

        with pytest.raises(NotImplementedError):
            connector.get_agent_response({}, "")

        with pytest.raises(NotImplementedError):
            connector.make_tool_calls([])

        with pytest.raises(NotImplementedError):
            connector.update_chat_history_with_toolcall_response({}, [])

    def test_set_default_token_tracker(self):
        """Test setting default token tracker class method."""
        mock_client = MockClient()
        custom_tracker = Mock(spec=BaseTokenUsageTracker)

        # Set default tracker
        Connector.set_default_token_tracker(custom_tracker)

        # New connector should use the custom tracker
        connector = Connector(mock_client)
        assert connector.token_tracker == custom_tracker

        # Reset to original default
        Connector.set_default_token_tracker(DefaultTokenUsageTracker())


class TestOpenAIConnector:
    """Test cases for the OpenAI connector."""

    def test_openai_connector_initialization(self):
        """Test OpenAI connector initialization."""
        mock_client = MockClient("openai")
        connector = OpenAIConnector(mock_client)

        assert connector.client == mock_client
        assert connector.config is not None
        assert connector.supported_roles == [
            "system",
            "assistant",
            "user",
            "function",
            "tool",
            "developer",
        ]
        assert connector.reasoning == ["low", "medium", "high"]

    def test_create_message_text_only(self):
        """Test creating message with text only."""
        mock_client = MockClient("openai")
        connector = OpenAIConnector(mock_client)

        result = connector.create_message_internal("Hello world")

        expected = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello world"}],
                "name": "user",
            }
        ]
        assert result == expected

    def test_create_message_with_image(self):
        """Test creating message with text and image."""
        mock_client = MockClient("openai")
        connector = OpenAIConnector(mock_client)

        image_data = {"data": "base64data", "img_fmt": "png"}
        result = connector.create_message_internal("Describe this image", image_data)

        assert len(result) == 1
        message = result[0]
        assert message["role"] == "user"
        assert len(message["content"]) == 2
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image_url"
        assert (
            "data:image/png;base64,base64data"
            in message["content"][1]["image_url"]["url"]
        )

    def test_create_message_image_only(self):
        """Test creating message with image only."""
        mock_client = MockClient("openai")
        connector = OpenAIConnector(mock_client)

        image_data = {"data": "base64data", "img_fmt": "jpeg"}
        result = connector.create_message_internal(None, image_data)

        assert len(result) == 1
        message = result[0]
        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "image_url"

    def test_adapt_functions_single_dict(self):
        """Test adapting a single function dictionary for OpenAI."""
        mock_client = MockClient("openai")
        connector = OpenAIConnector(mock_client)

        def test_func():
            return "test"

        function_schema = {
            "name": "test_function",
            "description": "A test function",
            "arguments": {
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
            "func_obj": test_func,
        }

        result = connector._adapt_functions(function_schema)

        assert len(result) == 1
        adapted = result[0]
        assert adapted["type"] == "function"
        assert "function" in adapted
        func = adapted["function"]
        assert func["name"] == "test_function"
        assert "parameters" in func
        assert "arguments" not in func
        assert "func_obj" not in func
        assert connector._func_obj_map["test_function"] == test_func

    def test_adapt_functions_list(self):
        """Test adapting a list of functions for OpenAI."""
        mock_client = MockClient("openai")
        connector = OpenAIConnector(mock_client)

        def func1():
            return "func1"

        def func2():
            return "func2"

        functions = [
            {
                "name": "function1",
                "description": "First function",
                "arguments": {"type": "object", "properties": {}},
                "func_obj": func1,
            },
            {
                "name": "function2",
                "description": "Second function",
                "arguments": {"type": "object", "properties": {}},
                "func_obj": func2,
            },
        ]

        result = connector._adapt_functions(functions)

        assert len(result) == 2
        assert all(item["type"] == "function" for item in result)
        assert connector._func_obj_map["function1"] == func1
        assert connector._func_obj_map["function2"] == func2


class TestClaudeConnector:
    """Test cases for the Claude connector."""

    def test_claude_connector_initialization(self):
        """Test Claude connector initialization."""
        mock_client = MockClient("claude")
        connector = ClaudeConnector(mock_client)

        assert connector.client == mock_client
        assert connector.config is not None
        assert connector.supported_roles == ["system", "assistant", "user"]
        assert connector.convert_image_fmt == {"jpeg": "png", "jpg": "png"}

    def test_create_message_text_only(self):
        """Test creating message with text only."""
        mock_client = MockClient("claude")
        connector = ClaudeConnector(mock_client)

        result = connector.create_message_internal("Hello Claude")

        expected = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello Claude"}],
            }
        ]
        assert result == expected

    def test_create_message_with_image(self):
        """Test creating message with text and image."""
        mock_client = MockClient("claude")
        connector = ClaudeConnector(mock_client)

        image_data = {"data": "base64data", "img_fmt": "png"}
        result = connector.create_message_internal("Analyze this", image_data)

        assert len(result) == 1
        message = result[0]
        assert message["role"] == "user"
        assert len(message["content"]) == 2
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image"
        assert message["content"][1]["source"]["type"] == "base64"
        assert message["content"][1]["source"]["media_type"] == "image/png"

    def test_create_message_image_format_conversion(self):
        """Test image format conversion for Claude."""
        mock_client = MockClient("claude")
        connector = ClaudeConnector(mock_client)

        # Test JPEG to PNG conversion
        image_data = {"data": "base64data", "img_fmt": "jpeg"}
        result = connector.create_message_internal(None, image_data)

        message = result[0]
        assert message["content"][0]["source"]["media_type"] == "image/png"

        # Test JPG to PNG conversion
        image_data = {"data": "base64data", "img_fmt": "jpg"}
        result = connector.create_message_internal(None, image_data)

        message = result[0]
        assert message["content"][0]["source"]["media_type"] == "image/png"

    def test_create_message_no_content_error(self):
        """Test that Claude connector raises error with no content."""
        mock_client = MockClient("claude")
        connector = ClaudeConnector(mock_client)

        with pytest.raises(ValueError, match="Either text or image is required"):
            connector.create_message_internal(None, None)

    def test_adapt_functions_single_dict(self):
        """Test adapting a single function dictionary for Claude."""
        mock_client = MockClient("claude")
        connector = ClaudeConnector(mock_client)

        def test_func():
            return "test"

        function_schema = {
            "name": "test_function",
            "description": "A test function",
            "arguments": {
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
            "func_obj": test_func,
        }

        result = connector._adapt_functions(function_schema)

        assert len(result) == 1
        adapted = result[0]
        assert adapted["name"] == "test_function"
        assert "input_schema" in adapted
        assert "arguments" not in adapted
        assert "func_obj" not in adapted
        assert connector._func_obj_map["test_function"] == test_func

    def test_adapt_functions_list(self):
        """Test adapting a list of functions for Claude."""
        mock_client = MockClient("claude")
        connector = ClaudeConnector(mock_client)

        def func1():
            return "func1"

        def func2():
            return "func2"

        functions = [
            {
                "name": "function1",
                "description": "First function",
                "arguments": {"type": "object", "properties": {}},
                "func_obj": func1,
            },
            {
                "name": "function2",
                "description": "Second function",
                "arguments": {"type": "object", "properties": {}},
                "func_obj": func2,
            },
        ]

        result = connector._adapt_functions(functions)

        assert len(result) == 2
        assert all("input_schema" in func for func in result)
        assert connector._func_obj_map["function1"] == func1
        assert connector._func_obj_map["function2"] == func2


class TestAzureOpenSourceConnector:
    """Test cases for the Azure OpenSource connector."""

    def test_azure_connector_initialization(self):
        """Test Azure connector initialization."""
        mock_client = MockClient("azure")
        connector = AzureOpenSourceConnector(mock_client)

        assert connector.client == mock_client
        assert connector.config is not None


class TestConnectorIntegration:
    """Integration tests for connectors."""

    def test_all_connectors_inherit_from_base(self):
        """Test that all connectors inherit from base Connector class."""
        mock_client = MockClient()

        openai_connector = OpenAIConnector(mock_client)
        claude_connector = ClaudeConnector(mock_client)
        azure_connector = AzureOpenSourceConnector(mock_client)

        assert isinstance(openai_connector, Connector)
        assert isinstance(claude_connector, Connector)
        assert isinstance(azure_connector, Connector)

    def test_connectors_have_different_supported_roles(self):
        """Test that different connectors have appropriate supported roles."""
        mock_client = MockClient()

        openai_connector = OpenAIConnector(mock_client)
        claude_connector = ClaudeConnector(mock_client)

        # OpenAI supports more roles
        assert len(openai_connector.supported_roles) > len(
            claude_connector.supported_roles
        )
        assert "tool" in openai_connector.supported_roles
        assert "tool" not in claude_connector.supported_roles

    def test_connector_message_creation_consistency(self):
        """Test that all connectors create messages with consistent structure."""
        mock_client = MockClient()

        connectors = [
            OpenAIConnector(mock_client),
            ClaudeConnector(mock_client),
        ]

        for connector in connectors:
            result = connector.create_message_internal("Test message")

            # All should return a list
            assert isinstance(result, list)
            assert len(result) >= 1

            # First message should have role and content
            message = result[0]
            assert "role" in message
            assert "content" in message
            assert message["role"] == "user"
