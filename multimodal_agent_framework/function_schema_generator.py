import inspect
from typing import Callable, Dict, Any


def generate_function_schema(func: Callable, doc=None) -> Dict[str, Any]:
    """
    Generate a JSON schema for a given function object.

    Args:
        func: The function object to generate schema for

    Returns:
        A dictionary containing the function's schema
    """
    # Get function signature
    sig = inspect.signature(func)
    # Get function docstring
    doc = doc if doc is not None else inspect.getdoc(func) or "No description available"

    # Generate parameters schema
    parameters = {"type": "object", "properties": {}, "required": []}

    # Process each parameter
    for param_name, param in sig.parameters.items():
        # Skip self parameter for class methods
        if param_name == "self":
            continue

        param_type = "string"  # default type

        # Try to infer type from annotations
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == str:
                param_type = "string"
            elif param.annotation == int:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == dict:
                param_type = "object"
            elif param.annotation == list:
                param_type = "array"

        # Add parameter to properties
        parameters["properties"][param_name] = {
            "type": param_type,
            "description": f"Parameter: {param_name}",
        }

        # Add to required list if parameter has no default value
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    # Construct the complete schema
    schema = {
        "name": func.__name__,
        "description": doc,
        "arguments": parameters,
        "func_obj": func,
    }

    return schema
