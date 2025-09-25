import json
import copy
from typing import Union
from ..logging_config import get_logger
from ..token_tracker import (
    BaseTokenUsageTracker,
    DefaultTokenUsageTracker,
    token_tracker,
)

logger = get_logger()


class Connector:
    _default_token_tracker = DefaultTokenUsageTracker()

    def __init__(self, client, token_tracker: BaseTokenUsageTracker = None):
        if client is None:
            raise ValueError("Client is required")
        self.client = client
        self.token_tracker = token_tracker or self._default_token_tracker
        self._cost = 0
        self._tokens = {"input_tokens": 0, "output_tokens": 0}
        self._response_tokens = {"input_tokens": 0, "output_tokens": 0, "model": None}
        self._func_obj_map = {}
        self._context = {}

    @classmethod
    def set_default_token_tracker(cls, tracker: BaseTokenUsageTracker):
        """Set the default token tracker for all new Connector instances."""
        cls._default_token_tracker = tracker

    def get_cost(self):
        return self._cost

    def validate_arguments(self, text, base64_image):
        if text is None and base64_image is None:
            raise ValueError("Either text or image is required")
        if isinstance(text, bytes):
            raise ValueError("Text should be a string")
        if base64_image is not None and not isinstance(base64_image, dict):
            raise ValueError(
                "Image should be a dict object containing the fields {'data':data, 'img_fmt':image_format}"
            )

    def create_message(self, text=None, base64_image=None, img_fmt=None):
        self.validate_arguments(text, base64_image)
        return self.create_message_internal(text, base64_image)

    def create_message_internal(self, text=None, base64_image=None):
        raise NotImplementedError("Subclasses must implement create_message")

    def get_response(
        self,
        chat_history=None,
        system_message=None,
        model=None,
        max_tokens=None,
        temperature=None,
        json_response=False,
        reasoning=None,
        tools=None,
    ):
        raise NotImplementedError("Subclasses must implement get_response")

    def _adapt_chat_history(self, chat_history):
        raise NotImplementedError("Subclasses must implement _adapt_chat_history")

    def _adapt_functions(self, functions):
        raise NotImplementedError("Subclasses must implement _adapt_functions")

    def get_system_message(self, system_prompt, name):
        raise NotImplementedError("Subclasses must implement get_system_message")

    def get_agent_response(self, response, name):
        raise NotImplementedError("Subclasses must implement get_agent_response")

    def make_tool_calls(self, toolcalls, callback=None):
        raise NotImplementedError("Subclasses must implement make_tool_calls")

    def update_chat_history_with_toolcall_response(
        self, toolcall_response, chat_history
    ):
        raise NotImplementedError(
            "Subclasses must implement update_chat_history_with_toolcall_response"
        )

    def _execute_function(self, func_obj, function_args: Union[str, dict]):
        """
        Execute a function with given arguments and return its result

        Args:
            func_obj: The function object to execute
            function_args: Arguments to pass to the function

        Returns:
            The result of the function execution
        """
        try:
            if isinstance(function_args, str):
                function_args = json.loads(function_args)
            logger.info(
                f"Function name: {func_obj.__name__}, arguments: {str(function_args)}"
            )
            ## For cases where we have images, lets substitute the values that we had replaced after tool call
            for key, value in function_args.items():
                if isinstance(value, str) and value in self._context:
                    function_args[key] = self._context[value]
            response = func_obj(**function_args)
            if "image" in response:
                import uuid

                id = str(uuid.uuid4())
                self._context[id] = response["image"]["data"]
                response["image"]["data"] = id
            return response
        except Exception as e:
            return {"text": f"Error executing function {func_obj.__name__}: {str(e)}"}

    def get_chat_text_content(self, message):
        """
        Extracts text content from a single chat message.

        Handles different content formats:
        - Simple string content: {"content": "text"}
        - Complex content arrays: {"content": [{"type": "text", "text": "message"}]}
        - Mixed content with images and text

        Args:
            message (dict): A single chat message

        Returns:
            str: Extracted text content from the message
        """
        return message if isinstance(message, str) else message[0].get("text")

    def set_chat_text_content(self, message, new_content):
        if isinstance(message, str):
            return new_content
        message[0]["text"] = new_content
        return message
