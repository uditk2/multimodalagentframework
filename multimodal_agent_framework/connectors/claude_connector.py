import json
import copy
from .base import Connector
from ..configs.claude_config import ClaudeConfig
from typing import Optional
from ..logging_config import get_logger
from ..token_tracker import BaseTokenUsageTracker

logger = get_logger()


class ClaudeConnector(Connector):
    supported_roles = ["system", "assistant", "user"]
    convert_image_fmt = {"jpeg": "png", "jpg": "png"}

    def __init__(
        self,
        client,
        config: Optional[ClaudeConfig] = None,
        token_tracker: BaseTokenUsageTracker = None,
    ):
        super().__init__(client, token_tracker)
        self.config = config or ClaudeConfig()

    def create_message_internal(self, text=None, base64_image=None):
        message = {"role": "user"}
        if text is None and base64_image is None:
            raise ValueError("Either text or image is required")
        content = []
        if text is not None:
            content = content + [{"type": "text", "text": text}]
        if base64_image is not None:
            img_fmt = base64_image["img_fmt"]
            if img_fmt in self.convert_image_fmt:
                img_fmt = self.convert_image_fmt[img_fmt]
            content = content + [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{img_fmt}",
                        "data": base64_image["data"],
                    },
                }
            ]
        message["content"] = content
        return [message]

    def _adapt_functions(self, functions):
        if isinstance(functions, dict):
            functions = [functions]

        # Create a deep copy to preserve the original function objects
        functions_copy = copy.deepcopy(functions)

        # update the arguments to parameters to conform to claude standards.
        for func in functions_copy:
            if "arguments" in func:
                func["input_schema"] = func.pop("arguments")
                self._func_obj_map[func["name"]] = func.pop("func_obj")
        return functions_copy

    def get_response(
        self,
        chat_history=None,
        system_message=None,
        model=None,
        max_tokens=8192,
        temperature=0,
        json_response=False,
        reasoning=None,
        tools=None,
    ):
        if system_message is None:
            raise ValueError("System message is required")
        if chat_history is None or not isinstance(chat_history, list):
            raise ValueError("Chat history is required and should be a list")
        model = model or self.config.default_model
        # This should be enforced. Currently, it is not enforced in the code.
        chat_history = self._adapt_chat_history(chat_history)
        if tools is not None:
            tools = self._adapt_functions(tools)
        thinking = None
        thinking = None
        if reasoning is not None:
            thinking = {"type": "enabled", "budget_tokens": 10000}

        kwargs = {
            "system": system_message,
            "model": model,
            "messages": chat_history,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools is not None:
            kwargs["tools"] = tools

        if thinking is not None:
            kwargs["thinking"] = thinking
            ## If thinking is enabled, temperatue can only be 1.
            kwargs["temperature"] = 1
        logger.debug(f"Request to claude with kwargs: {kwargs}")
        response = self.client.messages.create(**kwargs)
        logger.debug(f"Response from claude: {response}")
        # Compute costs using config pricing. Unknown models fall back gracefully.
        prompt_per_token, completion_per_token = self.config.get_token_costs(model)
        # TODO: Check if we can remove the next 2 lines as we have introduced influx db for usage tracking.
        self._cost += (
            response.usage.input_tokens * prompt_per_token
            + response.usage.output_tokens * completion_per_token
        )
        self._tokens = {
            "input_tokens": self._tokens["input_tokens"] + response.usage.input_tokens,
            "output_tokens": self._tokens["output_tokens"]
            + response.usage.output_tokens,
        }
        self._response_tokens = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model,
        }
        self.token_tracker.track_token_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model_name=model,
        )
        logger.debug(f"Cost: {self._cost} and tokens: {self._tokens}")
        final_response = []
        tools = []
        thinking = []
        logger.info(response)
        for content in response.content:
            if content.type == "text":
                final_response.append({"type": "content", "value": content.text})
            elif content.type == "tool_use":
                tools.append(
                    {
                        "type": content.type,
                        "id": content.id,
                        "name": content.name,
                        "input": content.input,
                    }
                )
            elif content.type == "thinking":
                thinking.append(
                    {
                        "type": content.type,
                        "thinking": content.thinking,
                        "signature": content.signature,
                    }
                )
        if len(tools) > 0:
            final_response.append({"type": "toolcall", "value": tools})
        if len(thinking) > 0:
            final_response.append({"type": "thinking", "value": thinking})
        return final_response

    def get_system_message(self, system_prompt, name=None):
        return system_prompt

    def get_agent_response(self, response, name):
        agent_response = {"role": "assistant"}
        content = []
        ordering_map = {"content": 1, "toolcall": 2, "thinking": 0}
        for single_response in response:
            if single_response["type"] == "content":
                content.append({"type": "text", "text": single_response["value"]})
            elif single_response["type"] == "toolcall":
                content.extend(single_response["value"])
            elif single_response["type"] == "thinking":
                content.extend(single_response["value"])
        # Sort content based on the ordering map
        content.sort(key=lambda x: ordering_map.get(x["type"], 3))
        agent_response["content"] = content
        return agent_response

    def _adapt_chat_history(self, chat_history):
        adapted_messages = []
        chat_history_length = len(chat_history)
        current_index = 0
        for msg in chat_history:
            current_index += 1
            # Handle if msg is a list
            if isinstance(msg, list):
                continue  # Skip list messages or handle them differently if needed

            # Handle if msg is not a dictionary
            if not isinstance(msg, dict):
                continue
            # Handle if required keys are missing
            role = msg.get(
                "role", self.supported_roles[1]
            )  # default to 'assistant' if role is missing
            content = msg.get("content", "")
            ## Removing thinking messages from the chat history if they are not the last 3 messages to reduce noise.
            if current_index < (chat_history_length - 3) and current_index >= 0:
                content = [c for c in content if c.get("type") != "thinking"]
            adapted_message = {
                "role": (
                    role if role in self.supported_roles else self.supported_roles[1]
                ),
                "content": content,
            }
            ## this message was from openai, so we need to adapt it to claude format.
            if role == "assistant" and "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    adapted_messages.append(
                        {
                            "role": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": call["function"]["arguments"],
                        }
                    )
                del msg["tool_calls"]
            adapted_messages.append(adapted_message)

        return adapted_messages

    def make_tool_calls(self, toolcalls, callback=None):
        response_map = {}
        for toolcall in toolcalls:
            if self._func_obj_map.get(toolcall["name"]) is None:
                logger.error(
                    f"Function {toolcall['name']} not found in function object map key: {self._func_obj_map.keys()}"
                )
                response_map[toolcall["id"]] = json.dumps(
                    {"text": f"Function {toolcall['name']} not found"}
                )
            else:
                if callback is not None:
                    callback(
                        {
                            "function": toolcall["name"],
                            "arguments": json.dumps(toolcall["input"]),
                        }
                    )
                response_map[toolcall["id"]] = json.dumps(
                    self._execute_function(
                        self._func_obj_map[toolcall["name"]], toolcall["input"]
                    )
                )
                if callback is not None:
                    callback(
                        {
                            "function": toolcall["name"],
                            "response": response_map[toolcall["id"]],
                        }
                    )
        return response_map

    def update_chat_history_with_toolcall_response(
        self, toolcall_response, chat_history
    ):
        tool_response_message = {"role": "user"}
        extra_msgs = []
        content = []
        for key in toolcall_response.keys():
            tool_response = json.loads(toolcall_response[key])
            logger.info(toolcall_response[key])
            if "image" in tool_response:
                image_data_key = tool_response["image"]["data"]
                tool_response["image"]["data"] = self._context[
                    tool_response["image"]["data"]
                ]
                extra_msgs.extend(
                    self.create_message_internal(
                        text=f"If you need to pass the image associated with toolcall id {key} as an argument, use the data key {image_data_key}. I will take care of changing it with the correct image."
                    )
                )
            tool_content = self.create_message_internal(
                tool_response["text"] if "text" in tool_response else None,
                tool_response["image"] if "image" in tool_response else None,
            )
            tool_content = tool_content[0]["content"]
            content.append(
                {"type": "tool_result", "tool_use_id": key, "content": tool_content}
            )
        tool_response_message["content"] = content
        chat_history.append(tool_response_message)
        chat_history.extend(extra_msgs)
        return chat_history
