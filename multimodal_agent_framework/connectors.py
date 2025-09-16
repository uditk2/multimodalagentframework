import json
import copy
from typing import Union
from .logging_config import get_logger
logger = get_logger()
from .token_tracker import token_tracker
class Connector():
    def __init__(self, client):
        if client is None:
            raise ValueError("Client is required")
        self.client = client
        self._cost = 0
        self._tokens = {"input_tokens": 0, "output_tokens": 0}
        self._response_tokens = {"input_tokens": 0, "output_tokens": 0, "model": None}
        self._func_obj_map = {}
        self._context = {}

    def get_cost(self):
        return self._cost
    
    def validate_arguments(self, text, base64_image):
        if text is None and base64_image is None:
            raise ValueError("Either text or image is required")
        if isinstance(text, bytes):
            raise ValueError("Text should be a string")
        if base64_image is not None and not isinstance(base64_image, dict):
            raise ValueError("Image should be a dict object containing the fields {'data':data, 'img_fmt':image_format}")
    
    def create_message(self, text=None, base64_image=None, img_fmt=None):
        self.validate_arguments(text, base64_image)
        return self.create_message_internal(text, base64_image)
    
    def create_message_internal(self, text=None, base64_image=None):
        raise NotImplementedError("Subclasses must implement create_message")

    def get_response(self, chat_history=None, system_message=None, model=None, max_tokens=None, 
                     temperature=None, json_response=False, reasoning=None, tools=None):
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
    
    def update_chat_history_with_toolcall_response(self, toolcall_response, chat_history):
        raise NotImplementedError("Subclasses must implement update_chat_history_with_toolcall_response")
    
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
            logger.info(f"Function name: {func_obj.__name__}, arguments: {str(function_args)}")
            ## For cases where we have images, lets substitute the values that we had replaced after tool call
            for key, value in function_args.items():
                if isinstance(value, str) and value in self._context:
                    function_args[key] = self._context[value]
            response = func_obj(**function_args)
            if 'image' in response:
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
        return message if isinstance(message, str) \
            else message[0].get("text")
        
    def set_chat_text_content(self, message, new_content):
        if isinstance(message, str):
            return new_content
        message[0]["text"] = new_content
        return message

class OpenAIConnector(Connector):
    supported_roles = ['system', 'assistant', 'user', 'function', 'tool', 'developer']
    reasoning = ["low", "medium", "high"]
    
    def create_message_internal(self, text=None, base64_image=None):
        message = {"role": "user"}
        content = []
        if text is not None:
            content = content + [{"type": "text", "text": text}]
        if base64_image is not None:
            content = content + [{"type": "image_url", "image_url": 
                                  {"url": f"data:image/{base64_image['img_fmt']};base64,{base64_image['data']}"}}]
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
            if 'arguments' in func:
                func['parameters'] = func.pop('arguments')
                self._func_obj_map[func["name"]] = func.pop('func_obj')
        functions_copy = [{"type": "function", "function": func} for func in functions_copy]
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
            role = msg.get("role", self.supported_roles[1])  # default to 'assistant' if role is missing
            name = msg.get("name", role)  # use role as name if name is missing
            ####
            adapted_message = {
                "role": role if role in self.supported_roles else self.supported_roles[1],
                "name": name
            }
            logger.debug(f"Adapting message: {msg} to {adapted_message}")
            if "content" in msg:
                if isinstance(msg["content"], str):
                    adapted_message["content"] = [{"type": "text", "text": msg["content"]}]
                elif isinstance(msg["content"], list):
                ## iterate through content messages and dont take the thinking part.
                    adapted_message["content"] = [c for c in msg["content"] if c.get("type") != "thinking"]
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
                        "function": {
                            "name": msg["name"],
                            "arguments": msg["input"]
                        }
                    }
                ]
                adapted_message["content"] = None
            ## the message was from claude, so we need to adapt it to openai format.
            if role == "tool_result":
                # OpenAI devs usually post tool results as a 'tool' message
                adapted_message = {"role": "tool", **msg}
            adapted_messages.append(adapted_message)
        
        return adapted_messages
    
    def get_response(self, chat_history=None, system_message=None, model="gpt-4o", max_tokens=None, temperature=0.7, 
                     json_response=False, reasoning=None, tools=None):
        if system_message is None or not isinstance(system_message, list):
            raise ValueError("System message is required and should be a list")
        if chat_history is None or not isinstance(chat_history, list):
            raise ValueError("Chat history is required and should be a list")
        chat_history = self._adapt_chat_history(chat_history)
        kwargs = {}
        ## TODO: ideally, we should take in details of the web search options from the user and pass it to the API.
        if model in ["gpt-4o-search-preview", "gpt-4o-mini-search-preview", "gpt-5-search-preview"]:
            kwargs["web_search_options"] = {}
        if model in ["gpt-4.1", "o4-mini", "o1", "o4", "o3"]:
            if tools is not None:
                kwargs["tool_choice"] = "auto"
        if model in ["o3-mini", "o1", "o4-mini", "o3", "o3-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            kwargs["max_completion_tokens"] = max_tokens
            if model in ["o3-mini", "o1"]:
                system_message[0]["role"] = self.supported_roles[5]
            if reasoning is None:
                kwargs['reasoning_effort'] = "low"
            elif isinstance(reasoning,str) and reasoning.lower() in self.reasoning:
                kwargs['reasoning_effort'] = reasoning.lower()
            elif isinstance(reasoning, bool) and reasoning:
                kwargs['reasoning_effort'] = "high" if reasoning else "low"
            else:
                raise ValueError(f"Reasoning can only take the following values {str(self.reasoning)}")
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if json_response:
            kwargs["response_format"] = { "type": "json_object" }
        messages = system_message + chat_history
        if tools is not None:
            tools = self._adapt_functions(tools)
        logger.debug(f"Request to OpenAI with kwargs: {kwargs} and tools: {tools}")
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs
        )
        prompt_token_cost = {"gpt-4o": 0.0000025, "gpt-4o-mini": 0.00000015, "o4-mini" : 0.0000011, "o3-mini": 0.0000011, "o1": 0.000015, "gpt-4.1": 0.000002, "gpt-4.1-mini": 0.0000004, "o3": 0.000002, "gpt-5": 0.00000125, "gpt-5-mini": 0.00000025, "gpt-5-nano": 0.00000005}
        output_token_cost = {"gpt-4o": 0.000010, "gpt-4o-mini": 0.0000006, "o4-mini" : 0.0000044, "o3-mini": 0.0000044, "o1": 0.000060, "gpt-4.1": 0.000008, "gpt-4.1-mini": 0.0000016, "o3": 0.000008, "gpt-5": 0.000010, "gpt-5-mini": 0.000002, "gpt-5-nano": 0.0000004}
        #TODO: Check if we can remove the next 2 lines as we have introduced influx db for usage tracking.
        self._cost += response.usage.prompt_tokens * prompt_token_cost[model] + response.usage.completion_tokens * output_token_cost[model]
        self._tokens = {
            "input_tokens": self._tokens["input_tokens"] + response.usage.prompt_tokens,
            "output_tokens": self._tokens["output_tokens"] + response.usage.completion_tokens
        }
        self._response_tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model
        }
        token_tracker.track_token_usage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens, model_name=model)
        logger.debug(f"Cost: {self._cost} and tokens: {self._tokens}")
        final_response = []
        if response.choices[0].message.tool_calls is not None:
            tool_calls = [{"id": toolcall.id, "type": toolcall.type, 
                       "function": {"name": toolcall.function.name,"arguments": toolcall.function.arguments}} 
                       for toolcall in response.choices[0].message.tool_calls]
            final_response.append({ "type": "toolcall", "value": tool_calls})
        if response.choices[0].message.content is not None:
            final_response.append({"type": "content", "value": response.choices[0].message.content, "usage": self._response_tokens})
        return final_response
    

    def get_system_message(self, system_prompt, name):
        return [{"role": "system", "content": system_prompt, "name": name}]
    
    def get_agent_response(self, response, name):
        agent_response = {"role": "assistant", "name":name}
        for single_response in response:
            if single_response["type"] == "content":
                agent_response["content"] = [{"type": "text", "text": single_response["value"]}]
            elif single_response["type"] == "toolcall":
                agent_response["tool_calls"]=single_response["value"]
        return agent_response

    
    def make_tool_calls(self, toolcalls, callback=None):
       response_map = {}
       for toolcall in toolcalls:
           if self._func_obj_map.get(toolcall['function']['name']) is None:
               logger.error(f"Function {toolcall['function']['name']} not found in function object map key: {self._func_obj_map.keys()}")
               response_map[toolcall['id']] = json.dumps({"text": f"Function {toolcall['function']['name']} not found"})
           else:
               logger.info(f"Executing function {toolcall['function']['name']}")
               if callback is not None:
                     callback({"function": toolcall['function']['name'], "arguments": json.dumps(toolcall['function']['arguments'])})
               response_map[toolcall['id']] = json.dumps(self._execute_function(self._func_obj_map[toolcall['function']['name']], 
                                                                     toolcall['function']['arguments']))
               if callback is not None:
                   callback({"function": toolcall['function']['name'], "response": response_map[toolcall['id']]})
       return response_map

    def update_chat_history_with_toolcall_response(self, toolcall_response, chat_history):
        image_messages= []
        for key in toolcall_response.keys():
            tool_response = json.loads(toolcall_response[key])
            if 'text' not in tool_response and 'image' not in tool_response:
                chat_history.append({"role": "tool", "tool_call_id": key, "content": "The tool did not return any answer"})
                continue
            if 'text' in tool_response:
                chat_history.append({"role": "tool", "tool_call_id": key, "content": tool_response['text']})
            else:
                chat_history.append({"role": "tool", "tool_call_id": key, "content": "The result of toolcall is an image attached later in the chat"}) 
            if 'image' in tool_response:
                import uuid
                id = str(uuid.uuid4())
                self._context[id] = tool_response["image"]["data"]
                tool_response["image"]["data"] = id
                image_messages.extend(self.create_message_internal(text=f"If you want to pass the image associated with for toolcall id {key} use {id}. I will replace the key with actual image",base64_image=tool_response['image']))
        if len(image_messages) > 0 :
            chat_history.extend(image_messages)
        return chat_history

class ClaudeConnector(Connector):
    supported_roles = ['system', 'assistant', 'user']
    convert_image_fmt= {'jpeg': 'png', 'jpg': 'png'}

    def create_message_internal(self, text=None, base64_image=None):
        message = {"role": "user"}
        if text is None and base64_image is None:
            raise ValueError("Either text or image is required")
        content = []
        if text is not None:
            content = content + [{"type": "text", "text": text}]
        if base64_image is not None:
            img_fmt = base64_image['img_fmt']
            if img_fmt in self.convert_image_fmt:
                img_fmt = self.convert_image_fmt[img_fmt]
            content = content + [{"type": "image", "source": {"type": "base64", "media_type": f"image/{img_fmt}", "data": base64_image['data']}}]
        message["content"] = content
        return [message]


    def _adapt_functions(self, functions):
        if isinstance(functions, dict):
            functions = [functions]
        
        # Create a deep copy to preserve the original function objects
        functions_copy = copy.deepcopy(functions)
        
        # update the arguments to parameters to conform to claude standards.
        for func in functions_copy:
            if 'arguments' in func:
                func['input_schema'] = func.pop('arguments')
                self._func_obj_map[func["name"]] = func.pop('func_obj')
        return functions_copy
    
    def get_response(self, chat_history=None, system_message=None, model="claude-sonnet-4-20250514", max_tokens=20000, 
                     temperature=0, json_response=False, reasoning=None, tools=None):
        if system_message is None:
            raise ValueError("System message is required")
        if chat_history is None or not isinstance(chat_history, list):
            raise ValueError("Chat history is required and should be a list")
        # This should be enforced. Currently, it is not enforced in the code.
        chat_history = self._adapt_chat_history(chat_history)
        if tools is not None:
            tools = self._adapt_functions(tools)
        thinking = None
        thinking = None
        if reasoning is not None:
            thinking = {
            "type": "enabled",
            "budget_tokens": 10000
            }
        
        kwargs = {
            "system": system_message,
            "model": model,
            "messages": chat_history,
            "max_tokens": max_tokens,
            "temperature": temperature
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
        #TODO: Check if we can remove the next 2 lines as we have introduced influx db for usage tracking.
        self._cost += (response.usage.input_tokens * .000003 + response.usage.output_tokens * .000015)
        self._tokens = {
            "input_tokens": self._tokens["input_tokens"] + response.usage.input_tokens,
            "output_tokens": self._tokens["output_tokens"] + response.usage.output_tokens
        }
        self._response_tokens = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model
        }
        token_tracker.track_token_usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens, model_name=model)
        logger.debug(f"Cost: {self._cost} and tokens: {self._tokens}")
        final_response = []
        tools = []
        thinking = []
        logger.info(response)
        for content in response.content:
            if content.type == "text":
                final_response.append({"type": "content", "value": content.text})
            elif content.type == "tool_use":
                tools.append({"type": content.type, "id": content.id, "name": content.name, "input":content.input})
            elif content.type == "thinking":
                thinking.append({"type": content.type, "thinking": content.thinking, "signature": content.signature})
        if len(tools) > 0:
            final_response.append({"type": "toolcall", "value": tools})
        if len(thinking) > 0:
            final_response.append({"type": "thinking", "value": thinking})
        return final_response
    
    def get_system_message(self, system_prompt, name=None):
        return system_prompt
    
    def get_agent_response(self, response, name):
        agent_response = {"role": "assistant"}
        content= []
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
            role = msg.get("role", self.supported_roles[1])  # default to 'assistant' if role is missing
            content = msg.get("content", "")
            ## Removing thinking messages from the chat history if they are not the last 3 messages to reduce noise.
            if current_index < (chat_history_length - 3) and current_index >= 0:
                content = [c for c in content if c.get("type") != "thinking"]
            adapted_message = {
                "role": role if role in self.supported_roles else self.supported_roles[1],
                "content": content
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
              logger.error(f"Function {toolcall['name']} not found in function object map key: {self._func_obj_map.keys()}")
              response_map[toolcall['id']] = json.dumps({"text": f"Function {toolcall['name']} not found"})
           else:
              if callback is not None:
                    callback({"function": toolcall["name"], "arguments": json.dumps(toolcall["input"])})
              response_map[toolcall["id"]] = json.dumps(self._execute_function(self._func_obj_map[toolcall["name"]], 
                                                                  toolcall["input"]))
              if callback is not None:
                callback({"function": toolcall["name"], "response": response_map[toolcall['id']]})
       return response_map
    
    def update_chat_history_with_toolcall_response(self, toolcall_response, chat_history):
        tool_response_message = {"role": "user"}
        extra_msgs=[]
        content = []
        for key in toolcall_response.keys():
            tool_response = json.loads(toolcall_response[key])
            logger.info(toolcall_response[key])
            if "image" in tool_response:
                image_data_key = tool_response["image"]["data"]
                tool_response["image"]["data"] = self._context[tool_response["image"]["data"]]
                extra_msgs.extend(self.create_message_internal(text=f"If you need to pass the image associated with toolcall id {key} as an argument, use the data key {image_data_key}. I will take care of changing it with the correct image."))
            tool_content = self.create_message_internal(tool_response["text"] if "text" in tool_response else None, 
                                                        tool_response["image"] if "image" in tool_response else None)
            tool_content = tool_content[0]["content"]
            content.append({"type": "tool_result", "tool_use_id": key, "content": tool_content})
        tool_response_message["content"] = content
        chat_history.append(tool_response_message)
        chat_history.extend(extra_msgs)
        return chat_history
    
class AzureOpenSourceConnector(Connector):
    supported_roles = ['system', 'assistant', 'user', 'function', 'tool', 'developer']
    
    def create_message_internal(self, text=None, base64_image=None):
        message = {"role": "user"}
        if text is None:
            raise ValueError("Text is required")
        content = text
        if base64_image is not None:
            raise ValueError("Image is not supported")
        message["content"] = content
        return [message]

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
            role = msg.get("role", self.supported_roles[1])  # default to 'assistant' if role is missing
            content = msg.get("content", "")
            
            adapted_message = {
                "role": role if role in self.supported_roles else self.supported_roles[1],
                "content": content
            }
            adapted_messages.append(adapted_message)
        
        return adapted_messages
    
    
    def _adapt_functions(self, functions):
        if isinstance(functions, dict):
            functions = [functions]
        
        # Create a deep copy to preserve the original function objects
        functions_copy = copy.deepcopy(functions)
        
        # update the arguments to parameters to conform to open ai standards.
        for func in functions_copy:
            if 'arguments' in func:
                func['parameters'] = func.pop('arguments')
                self._func_obj_map[func["name"]] = func.pop('func_obj')
        functions_copy = [{"type": "function", "function": func} for func in functions_copy]
        return functions_copy

    
    def get_response(self, chat_history=None, system_message=None, model="Codestral-2501", 
                     max_tokens=None, temperature=0.7, json_response=False, reasoning=None, tools=None):
        if system_message is None or not isinstance(system_message, list):
            raise ValueError("System message is required and should be a list")
        if chat_history is None or not isinstance(chat_history, list):
            raise ValueError("Chat history is required and should be a list")
        chat_history = self._adapt_chat_history(chat_history)
        messages = system_message + chat_history
        kwargs = {}
        if json_response:
            kwargs = {"response_format" : "json_object"}
        if tools is not None:
            tools = self._adapt_functions(tools)
        response = self.client.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature, 
            **kwargs,
            tools=tools
        )
        prompt_token_cost = {"Codestral-2501": 0.0000003}
        output_token_cost = {"Codestral-2501": 0.0000009}
        #TODO: Check if we can remove the next 2 lines as we have introduced influx db for usage tracking.
        self._cost += response.usage.prompt_tokens * prompt_token_cost[model] + response.usage.completion_tokens * output_token_cost[model]
        self._tokens = {
            "input_tokens": self._tokens["input_tokens"] + response.usage.prompt_tokens,
            "output_tokens": self._tokens["output_tokens"] + response.usage.completion_tokens
        }
        self._response_tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model
        }
        token_tracker.track_token_usage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens, model_name=model)
        logger.debug(f"Cost: {self._cost} and tokens: {self._tokens}")
        final_response = []
        if response.choices[0].message.tool_calls is not None:
            final_response.append({ "type": "toolcall", "value": response.choices[0].message.tool_calls})
        if response.choices[0].message.content is not None:
            final_response.append({"type": "content", "value": response.choices[0].message.content})
        return final_response
    
    def get_system_message(self, system_prompt, name):
        return [{"role": "system", "content": system_prompt}]
    
    def get_agent_response(self, response, name):
        agent_response = {"role": "assistant"}
        for single_response in response:
            if single_response["type"] == "content":
                agent_response["content"] = [{"type": "text", "text": single_response["value"]}]
            elif single_response["type"] == "toolcall":
                agent_response["tool_calls"]=single_response["value"]
        return agent_response

    def make_tool_calls(self, toolcalls, callback=None):
       response_map = {}
       for toolcall in toolcalls:
           if self._func_obj_map.get(toolcall['function']['name']) is None:
               logger.error(f"Function {toolcall['function']['name']} not found in function object map key: {self._func_obj_map.keys()}")
               response_map[toolcall['id']] = json.dumps({"text": f"Function {toolcall['function']['name']} not found"})
           else:
               logger.info(f"Executing function {toolcall['function']['name']}")
               if callback is not None:
                    callback({"function": toolcall['function']['name'], "arguments": json.dumps(toolcall['function']['arguments'])})
               response_map[toolcall['id']] = json.dumps(self._execute_function(self._func_obj_map[toolcall['function']['name']], 
                                                                     toolcall['function']['arguments']))
               if callback is not None:
                    callback({"function": toolcall['function']['name'], "response": response_map[toolcall['id']]})
       return response_map

    def update_chat_history_with_toolcall_response(self, toolcall_response, chat_history):
        image_messages= []
        for key in toolcall_response.keys():
            tool_response = json.loads(toolcall_response[key])
            if 'text' not in tool_response and 'image' not in tool_response:
                chat_history.append({"role": "tool", "tool_call_id": key, "content": "The tool did not return any answer"})
                continue
            if 'text' in tool_response:
                chat_history.append({"role": "tool", "tool_call_id": key, "content": tool_response['text']})
            else:
                chat_history.append({"role": "tool", "tool_call_id": key, "content": "The result of toolcall is an image attached later in the chat"}) 
            if 'image' in tool_response:
                tool_response["image"]["data"] = self._context[tool_response["image"]["data"]]
                image_messages.extend(self.create_message_internal(text=f"image response for toolcall id {key}",base64_image=tool_response['image']))
        if len(image_messages) > 0 :
            chat_history.extend(image_messages)
        return chat_history