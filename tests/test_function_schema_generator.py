import pytest
import inspect
from typing import Dict, List, Any, Optional, Union
from multimodal_agent_framework.function_schema_generator import (
    generate_function_schema,
)


class TestFunctionSchemaGenerator:
    """Test cases for the function schema generator."""

    def test_simple_function_no_params(self):
        """Test schema generation for a function with no parameters."""

        def simple_func():
            """A simple function with no parameters."""
            return "hello"

        schema = generate_function_schema(simple_func)

        assert schema["name"] == "simple_func"
        assert schema["description"] == "A simple function with no parameters."
        assert schema["arguments"]["type"] == "object"
        assert schema["arguments"]["properties"] == {}
        assert schema["arguments"]["required"] == []
        assert schema["func_obj"] == simple_func

    def test_function_with_string_param(self):
        """Test schema generation for a function with string parameter."""

        def func_with_string(name: str):
            """Function with a string parameter."""
            return f"Hello {name}"

        schema = generate_function_schema(func_with_string)

        assert schema["name"] == "func_with_string"
        assert schema["description"] == "Function with a string parameter."
        assert "name" in schema["arguments"]["properties"]
        assert schema["arguments"]["properties"]["name"]["type"] == "string"
        assert (
            schema["arguments"]["properties"]["name"]["description"]
            == "Parameter: name"
        )
        assert "name" in schema["arguments"]["required"]

    def test_function_with_multiple_typed_params(self):
        """Test schema generation for function with multiple typed parameters."""

        def multi_param_func(
            name: str, age: int, active: bool, data: dict, items: list
        ):
            """Function with multiple typed parameters."""
            return {
                "name": name,
                "age": age,
                "active": active,
                "data": data,
                "items": items,
            }

        schema = generate_function_schema(multi_param_func)

        assert schema["name"] == "multi_param_func"
        assert len(schema["arguments"]["properties"]) == 5

        # Check types
        assert schema["arguments"]["properties"]["name"]["type"] == "string"
        assert schema["arguments"]["properties"]["age"]["type"] == "number"
        assert schema["arguments"]["properties"]["active"]["type"] == "boolean"
        assert schema["arguments"]["properties"]["data"]["type"] == "object"
        assert schema["arguments"]["properties"]["items"]["type"] == "array"

        # Check all are required
        assert len(schema["arguments"]["required"]) == 5
        assert all(
            param in schema["arguments"]["required"]
            for param in ["name", "age", "active", "data", "items"]
        )

    def test_function_with_optional_params(self):
        """Test schema generation for function with optional parameters."""

        def optional_param_func(required_param: str, optional_param: int = 42):
            """Function with optional parameter."""
            return f"{required_param}: {optional_param}"

        schema = generate_function_schema(optional_param_func)

        assert schema["name"] == "optional_param_func"
        assert len(schema["arguments"]["properties"]) == 2
        assert "required_param" in schema["arguments"]["required"]
        assert "optional_param" not in schema["arguments"]["required"]
        assert len(schema["arguments"]["required"]) == 1

    def test_function_with_no_annotations(self):
        """Test schema generation for function without type annotations."""

        def untyped_func(param1, param2):
            """Function without type annotations."""
            return param1 + param2

        schema = generate_function_schema(untyped_func)

        assert schema["name"] == "untyped_func"
        assert len(schema["arguments"]["properties"]) == 2

        # Should default to string type
        assert schema["arguments"]["properties"]["param1"]["type"] == "string"
        assert schema["arguments"]["properties"]["param2"]["type"] == "string"

        # Both should be required (no defaults)
        assert len(schema["arguments"]["required"]) == 2

    def test_function_with_no_docstring(self):
        """Test schema generation for function without docstring."""

        def no_doc_func(param: str):
            return param

        schema = generate_function_schema(no_doc_func)

        assert schema["name"] == "no_doc_func"
        assert schema["description"] == "No description available"

    def test_class_method_skips_self(self):
        """Test that class method schemas skip the 'self' parameter."""

        class TestClass:
            def method(self, param: str):
                """A class method."""
                return param

        instance = TestClass()
        schema = generate_function_schema(instance.method)

        assert schema["name"] == "method"
        assert "self" not in schema["arguments"]["properties"]
        assert len(schema["arguments"]["properties"]) == 1
        assert "param" in schema["arguments"]["properties"]

    def test_function_with_complex_types(self):
        """Test schema generation for function with complex type annotations."""

        def complex_func(
            union_param: Union[str, int], optional_list: Optional[List[str]] = None
        ):
            """Function with complex type annotations."""
            return union_param

        schema = generate_function_schema(complex_func)

        assert schema["name"] == "complex_func"
        assert len(schema["arguments"]["properties"]) == 2

        # Complex types should default to string
        assert schema["arguments"]["properties"]["union_param"]["type"] == "string"
        assert schema["arguments"]["properties"]["optional_list"]["type"] == "string"

        # Only union_param should be required
        assert "union_param" in schema["arguments"]["required"]
        assert "optional_list" not in schema["arguments"]["required"]

    def test_custom_docstring_override(self):
        """Test that custom docstring parameter overrides function docstring."""

        def func_with_doc():
            """Original docstring."""
            pass

        custom_doc = "Custom docstring override"
        schema = generate_function_schema(func_with_doc, doc=custom_doc)

        assert schema["description"] == custom_doc

    def test_function_with_mixed_params(self):
        """Test schema generation for function with mixed parameter types."""

        def mixed_func(
            required: str, optional_int: int = 10, optional_bool: bool = True
        ):
            """Function with mixed parameter types."""
            return {"required": required, "int": optional_int, "bool": optional_bool}

        schema = generate_function_schema(mixed_func)

        assert schema["name"] == "mixed_func"
        assert len(schema["arguments"]["properties"]) == 3
        assert len(schema["arguments"]["required"]) == 1
        assert "required" in schema["arguments"]["required"]

        # Check types are correct
        assert schema["arguments"]["properties"]["required"]["type"] == "string"
        assert schema["arguments"]["properties"]["optional_int"]["type"] == "number"
        assert schema["arguments"]["properties"]["optional_bool"]["type"] == "boolean"

    def test_schema_structure_completeness(self):
        """Test that generated schema has all required fields."""

        def test_func(param: str):
            """Test function."""
            return param

        schema = generate_function_schema(test_func)

        # Check top-level keys
        required_keys = ["name", "description", "arguments", "func_obj"]
        assert all(key in schema for key in required_keys)

        # Check arguments structure
        args = schema["arguments"]
        assert "type" in args
        assert "properties" in args
        assert "required" in args
        assert args["type"] == "object"
        assert isinstance(args["properties"], dict)
        assert isinstance(args["required"], list)

    def test_parameter_description_format(self):
        """Test that parameter descriptions follow the expected format."""

        def test_func(my_param: str, another_param: int):
            """Test function."""
            return f"{my_param}: {another_param}"

        schema = generate_function_schema(test_func)

        assert (
            schema["arguments"]["properties"]["my_param"]["description"]
            == "Parameter: my_param"
        )
        assert (
            schema["arguments"]["properties"]["another_param"]["description"]
            == "Parameter: another_param"
        )

    def test_empty_function_name_preservation(self):
        """Test that function names are preserved correctly."""

        def very_long_function_name_with_underscores():
            """Test function name preservation."""
            pass

        schema = generate_function_schema(very_long_function_name_with_underscores)
        assert schema["name"] == "very_long_function_name_with_underscores"

    def test_lambda_function(self):
        """Test schema generation for lambda functions."""
        lambda_func = lambda x: x * 2

        schema = generate_function_schema(lambda_func)

        assert schema["name"] == "<lambda>"
        assert "x" in schema["arguments"]["properties"]
        assert (
            schema["arguments"]["properties"]["x"]["type"] == "string"
        )  # No annotation
        assert "x" in schema["arguments"]["required"]
