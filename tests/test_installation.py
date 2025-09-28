#!/usr/bin/env python3
"""
Installation Test Script for Multimodal Agent Framework

This script tests that the package can be installed and used correctly
in a fresh environment, simulating real-world usage.
"""


def test_basic_imports():
    """Test that all main components can be imported."""
    print("🔍 Testing basic imports...")

    try:
        # Test main framework imports
        from multimodal_agent_framework import (
            MultiModalAgent,
            OpenAIConnector,
            ClaudeConnector,
            AzureOpenSourceConnector,
            generate_function_schema,
        )

        print("  ✅ Main framework imports successful")

        # Test helper function imports
        from multimodal_agent_framework import (
            get_openai_client,
            get_claude_client,
            get_azure_opensource_client,
        )

        print("  ✅ Helper function imports successful")

        # Test connector submodule imports
        from multimodal_agent_framework.connectors import Connector

        print("  ✅ Connector base class import successful")

        # Test conversation manager imports
        from multimodal_agent_framework.conversation_manager import (
            AgentConversation,
            AgentConversationManager,
        )

        print("  ✅ Conversation manager imports successful")

        # Test storage imports
        from multimodal_agent_framework.conversation_manager.storage import (
            BaseStorage,
            FileStorage,
            S3Storage,
        )

        print("  ✅ Storage backend imports successful")

        return True

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_function_schema_generator():
    """Test the function schema generator functionality."""
    print("\n🔍 Testing function schema generator...")

    try:
        from multimodal_agent_framework import generate_function_schema

        # Test function for schema generation
        def sample_function(name: str, age: int = 25, active: bool = True):
            """A sample function for testing schema generation."""
            return f"{name} is {age} years old and {'active' if active else 'inactive'}"

        # Generate schema
        schema = generate_function_schema(sample_function)

        # Verify schema structure
        required_keys = ["name", "description", "arguments", "func_obj"]
        if not all(key in schema for key in required_keys):
            print("  ❌ Schema missing required keys")
            return False

        # Verify function name
        if schema["name"] != "sample_function":
            print(f"  ❌ Wrong function name: {schema['name']}")
            return False

        # Verify arguments structure
        args = schema["arguments"]
        if not all(key in args for key in ["type", "properties", "required"]):
            print("  ❌ Arguments structure invalid")
            return False

        # Verify parameter types
        props = args["properties"]
        if props["name"]["type"] != "string":
            print("  ❌ Wrong type for 'name' parameter")
            return False

        if props["age"]["type"] != "number":
            print("  ❌ Wrong type for 'age' parameter")
            return False

        if props["active"]["type"] != "boolean":
            print("  ❌ Wrong type for 'active' parameter")
            return False

        # Verify required parameters (only 'name' should be required)
        if args["required"] != ["name"]:
            print(f"  ❌ Wrong required parameters: {args['required']}")
            return False

        print("  ✅ Function schema generator working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Function schema generator test failed: {e}")
        return False


def test_connector_instantiation():
    """Test that connectors can be instantiated (without API keys)."""
    print("\n🔍 Testing connector instantiation...")

    try:
        from multimodal_agent_framework.connectors import (
            OpenAIConnector,
            ClaudeConnector,
            AzureOpenSourceConnector,
        )

        # Test that we can import and access the classes
        # (We won't instantiate them without API keys)
        assert hasattr(OpenAIConnector, "__init__")
        assert hasattr(ClaudeConnector, "__init__")
        assert hasattr(AzureOpenSourceConnector, "__init__")

        print("  ✅ Connector classes accessible")
        return True

    except Exception as e:
        print(f"  ❌ Connector instantiation test failed: {e}")
        return False


def test_conversation_manager():
    """Test conversation manager functionality."""
    print("\n🔍 Testing conversation manager...")

    try:
        from multimodal_agent_framework.conversation_manager import (
            AgentConversation,
            AgentConversationManager,
        )
        from multimodal_agent_framework.conversation_manager.storage import FileStorage

        # Test AgentConversation creation
        conversation = AgentConversation(
            agent_name="test_agent",
            chat_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            metadata={"test": True},
        )

        assert conversation.agent_name == "test_agent"
        assert len(conversation.chat_history) == 2
        assert conversation.metadata["test"] == True

        # Test FileStorage instantiation
        storage = FileStorage(base_path="/tmp/test_conversations")
        assert hasattr(storage, "save_conversation")
        assert hasattr(storage, "load_conversation")

        print("  ✅ Conversation manager components working")
        return True

    except Exception as e:
        print(f"  ❌ Conversation manager test failed: {e}")
        return False


def test_package_metadata():
    """Test that package metadata is accessible."""
    print("\n🔍 Testing package metadata...")

    try:
        import multimodal_agent_framework

        # Test that __all__ is defined
        if hasattr(multimodal_agent_framework, "__all__"):
            print(
                f"  ✅ Package exports: {len(multimodal_agent_framework.__all__)} items"
            )
        else:
            print("  ⚠️  __all__ not defined")

        # Test version access (if defined)
        if hasattr(multimodal_agent_framework, "__version__"):
            print(f"  ✅ Package version: {multimodal_agent_framework.__version__}")
        else:
            print("  ⚠️  __version__ not defined")

        return True

    except Exception as e:
        print(f"  ❌ Package metadata test failed: {e}")
        return False


def main():
    """Run all installation tests."""
    print("🚀 Starting Multimodal Agent Framework Installation Tests\n")

    tests = [
        test_basic_imports,
        test_function_schema_generator,
        test_connector_instantiation,
        test_conversation_manager,
        test_package_metadata,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The package is ready for use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the package installation.")
        return 1


if __name__ == "__main__":
    exit(main())
