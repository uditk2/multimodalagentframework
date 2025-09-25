import json
import copy
from .base import Connector
from ..configs.openai_config import OpenAIConfig
from typing import Optional
from ..logging_config import get_logger
from ..token_tracker import BaseTokenUsageTracker

logger = get_logger()


class OpenAIConnector(Connector):
    supported_roles = ["system", "assistant", "user", "function", "tool", "developer"]
    reasoning = ["low", "medium", "high"]

    def __init__(
        self,
        client,
        config: Optional[OpenAIConfig] = None,
        token_tracker: BaseTokenUsageTracker = None,
    ):
        super().__init__(client, token_tracker)
        self.config = config or OpenAIConfig()

    def create_message_internal(self, text=None, base64_image=None):
        message = {"role": "user"}
        content = []
        if text is not None:
            content = content + [{"type": "text", "text": text}]
        if base64_image is not None:
            content = content + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{base64_image['img_fmt']};base64,{base64_image['data']}"
                    },
                }
            ]
        message["content"] = content
        message["name"] = "user"
        return [message]

    def _adapt_functions(self, functions):
        if isinstance(functions, dict):
            functions = [functions]

        # Create a deep copy to preserve the original function objects
        functions_copy = copy.deepcopy(functions)

        # update the arguments to parameters to conform to open ai standards.
        for func in functions_copy:
            if "arguments" in func:
                func["parameters"] = func.pop("arguments")
                self._func_obj_map[func["name"]] = func.pop("func_obj")
        functions_copy = [
            {"type": "function", "function": func} for func in functions_copy
        ]
        return functions_copy

    def _adapt_chat_history(self, chat_history):
        adapted_messages = []
        for msg in chat_history:
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
            name = msg.get("name", role)  # use role as name if name is missing
            ####
            adapted_message = {
                "role": (
                    role if role in self.supported_roles else self.supported_roles[1]
                ),
                "name": name,
            }
            logger.debug(f"Adapting message: {msg} to {adapted_message}")
            if "content" in msg:
                if isinstance(msg["content"], str):
                    adapted_message["content"] = [
                        {"type": "text", "text": msg["content"]}
                    ]
                elif isinstance(msg["content"], list):
                    ## iterate through content messages and dont take the thinking part.
                    adapted_message["content"] = [
                        c for c in msg["content"] if c.get("type") != "thinking"
                    ]
            if "tool_calls" in msg:
                adapted_message["tool_calls"] = msg.get("tool_calls", "")
            if "tool_call_id" in msg:
                adapted_message["tool_call_id"] = msg.get("tool_call_id")
            ## the message was from claude, so we need to adapt it to openai format.
            if role == "tool_use":
                # fold into assistant.tool_calls
                adapted_message["role"] = "assistant"
                adapted_message["tool_calls"] = [
                    {
                        "id": msg["id"],
                        "type": "function",
                        "function": {"name": msg["name"], "arguments": msg["input"]},
                    }
                ]
                adapted_message["content"] = None
            ## the message was from claude, so we need to adapt it to openai format.
            if role == "tool_result":
                # OpenAI devs usually post tool results as a 'tool' message
                adapted_message = {"role": "tool", **msg}
            adapted_messages.append(adapted_message)

        return adapted_messages

    def get_response(
        self,
        chat_history=None,
        system_message=None,
        model=None,
        max_tokens=None,
        temperature=0.7,
        json_response=False,
        reasoning=None,
        tools=None,
    ):
        if system_message is None or not isinstance(system_message, list):
            raise ValueError("System message is required and should be a list")
        if chat_history is None or not isinstance(chat_history, list):
            raise ValueError("Chat history is required and should be a list")
        model = model or self.config.default_model
        chat_history = self._adapt_chat_history(chat_history)
        kwargs = {}
        ## TODO: ideally, we should take in details of the web search options from the user and pass it to the API.
        if model in [
            "gpt-4o-search-preview",
            "gpt-4o-mini-search-preview",
            "gpt-5-search-preview",
        ]:
            kwargs["web_search_options"] = {}
        if model in ["gpt-4.1", "o4-mini", "o1", "o4", "o3"]:
            if tools is not None:
                kwargs["tool_choice"] = "auto"
        if model in [
            "o3-mini",
            "o1",
            "o4-mini",
            "o3",
            "o3-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ]:
            kwargs["max_completion_tokens"] = max_tokens
            if model in ["o3-mini", "o1"]:
                system_message[0]["role"] = self.supported_roles[5]
            if reasoning is None:
                kwargs["reasoning_effort"] = "low"
            elif isinstance(reasoning, str) and reasoning.lower() in self.reasoning:
                kwargs["reasoning_effort"] = reasoning.lower()
            elif isinstance(reasoning, bool) and reasoning:
                kwargs["reasoning_effort"] = "high" if reasoning else "low"
            else:
                raise ValueError(
                    f"Reasoning can only take the following values {str(self.reasoning)}"
                )
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}
        messages = system_message + chat_history
        if tools is not None:
            tools = self._adapt_functions(tools)
        logger.debug(f"Request to OpenAI with kwargs: {kwargs} and tools: {tools}")
        response = self.client.chat.completions.create(
            model=model, messages=messages, tools=tools, **kwargs
        )
        # Compute costs using config pricing. Unknown models fall back gracefully.
        prompt_per_token, completion_per_token = self.config.get_token_costs(model)
        # TODO: Check if we can remove the next 2 lines as we have introduced influx db for usage tracking.
        self._cost += (
            response.usage.prompt_tokens * prompt_per_token
            + response.usage.completion_tokens * completion_per_token
        )
        self._tokens = {
            "input_tokens": self._tokens["input_tokens"] + response.usage.prompt_tokens,
            "output_tokens": self._tokens["output_tokens"]
            + response.usage.completion_tokens,
        }
        self._response_tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model,
        }
        self.token_tracker.track_token_usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model_name=model,
        )
        logger.debug(f"Cost: {self._cost} and tokens: {self._tokens}")
        final_response = []
        if response.choices[0].message.tool_calls is not None:
            tool_calls = [
                {
                    "id": toolcall.id,
                    "type": toolcall.type,
                    "function": {
                        "name": toolcall.function.name,
                        "arguments": toolcall.function.arguments,
                    },
                }
                for toolcall in response.choices[0].message.tool_calls
            ]
            final_response.append({"type": "toolcall", "value": tool_calls})
        if response.choices[0].message.content is not None:
            final_response.append(
                {
                    "type": "content",
                    "value": response.choices[0].message.content,
                    "usage": self._response_tokens,
                }
            )
        return final_response

    def get_system_message(self, system_prompt, name):
        return [{"role": "system", "content": system_prompt, "name": name}]

    def get_agent_response(self, response, name):
        agent_response = {"role": "assistant", "name": name}
        for single_response in response:
            if single_response["type"] == "content":
                agent_response["content"] = [
                    {"type": "text", "text": single_response["value"]}
                ]
            elif single_response["type"] == "toolcall":
                agent_response["tool_calls"] = single_response["value"]
        return agent_response

    def make_tool_calls(self, toolcalls, callback=None):
        response_map = {}
        for toolcall in toolcalls:
            if self._func_obj_map.get(toolcall["function"]["name"]) is None:
                logger.error(
                    f"Function {toolcall['function']['name']} not found in function object map key: {self._func_obj_map.keys()}"
                )
                response_map[toolcall["id"]] = json.dumps(
                    {"text": f"Function {toolcall['function']['name']} not found"}
                )
            else:
                logger.info(f"Executing function {toolcall['function']['name']}")
                if callback is not None:
                    callback(
                        {
                            "function": toolcall["function"]["name"],
                            "arguments": json.dumps(toolcall["function"]["arguments"]),
                        }
                    )
                response_map[toolcall["id"]] = json.dumps(
                    self._execute_function(
                        self._func_obj_map[toolcall["function"]["name"]],
                        toolcall["function"]["arguments"],
                    )
                )
                if callback is not None:
                    callback(
                        {
                            "function": toolcall["function"]["name"],
                            "response": response_map[toolcall["id"]],
                        }
                    )
        return response_map

    def update_chat_history_with_toolcall_response(
        self, toolcall_response, chat_history
    ):
        image_messages = []
        for key in toolcall_response.keys():
            tool_response = json.loads(toolcall_response[key])
            if "text" not in tool_response and "image" not in tool_response:
                chat_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": key,
                        "content": "The tool did not return any answer",
                    }
                )
                continue
            if "text" in tool_response:
                chat_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": key,
                        "content": tool_response["text"],
                    }
                )
            else:
                chat_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": key,
                        "content": "The result of toolcall is an image attached later in the chat",
                    }
                )
            if "image" in tool_response:
                import uuid

                id = str(uuid.uuid4())
                self._context[id] = tool_response["image"]["data"]
                tool_response["image"]["data"] = id
                image_messages.extend(
                    self.create_message_internal(
                        text=f"If you want to pass the image associated with for toolcall id {key} use {id}. I will replace the key with actual image",
                        base64_image=tool_response["image"],
                    )
                )
        if len(image_messages) > 0:
            chat_history.extend(image_messages)
        return chat_history
