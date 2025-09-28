from .connectors import Connector, ClaudeConnector
from retrying import retry
from .logging_config import get_logger

logger = get_logger()


class NoTokensAvailableError(Exception):
    """Exception raised when a user does not have sufficient tokens available."""

    def __init__(
        self, message="User does not have sufficient tokens available. Please recharge."
    ):
        self.message = message
        super().__init__(self.message)


class Reviewer:
    """
    A reviewer that can process agent responses with custom review logic.

    The review_function, if provided, should follow this contract:
    - Input: (review_prompt: str, response: any)
    - Output: tuple[str, any] where:
      - First element: review message text to send back to the agent
      - Second element: optional image data (base64 dict or None)

    Example:
        def custom_reviewer(prompt, response):
            analysis = f"Reviewing: {response}"
            return analysis, None  # (message, image)
    """

    def __init__(self, review_prompt: str = None, review_function=None):
        """
        Initialize the reviewer.

        Args:
            review_prompt: Default prompt to use for reviews
            review_function: Custom function that takes (prompt, response)
                           and returns (review_message, image_data)
        """
        self.review_prompt = (
            review_prompt or "Please review the conversation and provide feedback."
        )
        self.review_function = review_function

    def get_message(self, response):
        """
        Get review message for the given response.

        Args:
            response: The agent's response to review

        Returns:
            tuple[str, any]: (review_message, image_data)
        """
        if self.review_function is None:
            return self.review_prompt, None
        # TODO This does not support reviewing the tools. To be tested.
        return self.review_function(self.review_prompt, response)


class MultiModalAgent:
    def __init__(
        self,
        name: str = None,
        system_prompt: str = None,
        reviewer: Reviewer = None,
        connector: Connector = None,
        reasoning=False,
        update_token_callback=None,
        check_token_callback=None,
    ):
        if name is None:
            raise ValueError("Name is required")
        self.name = name
        if system_prompt is None and not reasoning:
            raise ValueError("System prompt is required")
        if reasoning and isinstance(Connector, ClaudeConnector):
            raise ValueError("Reasoning is not supported with ClaudeConnector")

        self.system_prompt = system_prompt
        self.reviewer = reviewer
        self.connector = connector
        self.update_token_callback = update_token_callback
        self.check_token_callback = check_token_callback

    def filter_chat_history(self, chat_history, filters=None):
        if chat_history is None:
            return []
        if filters is None:
            return chat_history

        return [msg for msg in chat_history if msg["name"] in filters]

    def update_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def execute_user_ask(
        self,
        user_input=None,
        chat_history=None,
        base64image=None,
        temperature=0.7,
        filters=None,
        model=None,
        json_response=False,
        reasoning=None,
        tools=None,
        tool_call_info_callback=None,
    ) -> tuple:
        """Execute user's request and optionally process it through a reviewer.

        This method processes user input, optionally considering chat history and image data,
        and returns the AI's response. If a reviewer is configured, the initial response
        is passed through the reviewer for additional processing.

        Args:
            user_input (str): The user's input text/query
            chat_history (list, optional): Previous conversation history. Defaults to None.
            base64image (str, optional): Base64 encoded image data. Defaults to None.
            temperature (float, optional): Temperature parameter for response generation. Defaults to 0.7.
            filters (dict, optional): Filtering parameters for response generation. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - response (str): The AI's textual response
                - chat_history (list): Updated conversation history
        """
        response, chat_history = self._get_response(
            user_input=user_input,
            chat_history=chat_history,
            base64image=base64image,
            temperature=temperature,
            filters=filters,
            model=model,
            json_response=json_response,
            reasoning=reasoning,
            tools=tools,
        )
        ## As long as the agent is asking to make tool calls, lets do that.
        while len(response) > 0 and any(
            single_response.get("type") == "toolcall" for single_response in response
        ):
            for single_response in response:
                if single_response["type"] == "toolcall":
                    tool_response = self.connector.make_tool_calls(
                        single_response["value"], callback=tool_call_info_callback
                    )
                    logger.debug(f"Printing tool response : {tool_response}")
                    chat_history = (
                        self.connector.update_chat_history_with_toolcall_response(
                            tool_response, chat_history
                        )
                    )
            response, chat_history = self._get_response(
                chat_history=chat_history,
                temperature=temperature,
                json_response=json_response,
                reasoning=reasoning,
                filters=filters,
                model=model,
                tools=tools,
            )
        if self.reviewer is not None:
            reviewer_msg, reviewer_img = self.reviewer.get_message(response)
            response, chat_history = self._get_response(
                user_input=reviewer_msg,
                chat_history=chat_history,
                base64image=reviewer_img,
                temperature=temperature,
                filters=filters,
                model=model,
                json_response=json_response,
                reasoning=reasoning,
                tools=tools,
            )
        logger.debug(f"Response from agent {self.name} : {response}")
        final_response = None
        if response is not None and isinstance(response, list) and len(response) > 0:
            # Find the content response (text) among potentially multiple response types (thinking, toolcall, etc.)
            for item in response:
                if isinstance(item, dict) and item.get("type") == "content":
                    final_response = item.get("value", "")
                    break
            # Fallback to first item if no content type found (backward compatibility)
            if final_response is None and isinstance(response[0], dict):
                final_response = response[0].get("value", "")
        return final_response, chat_history

    def check_tokens(self, chat_history=None):
        if self.check_token_callback is not None:
            try:
                self.check_token_callback(chat_history)
            except Exception as e:
                logger.warning(f"Token check callback failed: {e}")

    def update_tokens(self, response_tokens=None):
        if self.update_token_callback is not None:
            try:
                self.update_token_callback(response_tokens)
            except Exception as e:
                logger.warning(f"Token update callback failed: {e}")

    def should_retry_exception(e):
        return "Rate limit" in str(e) or "429" in str(e) or "529" in str(e)

    @retry(
        stop_max_attempt_number=3,
        wait_fixed=1000,
        wait_exponential_multiplier=2000,
        retry_on_exception=should_retry_exception,
    )
    def _get_response(
        self,
        chat_history=None,
        user_input=None,
        base64image=None,
        temperature=0.7,
        filters=None,
        model=None,
        json_response=False,
        reasoning=None,
        tools=None,
    ):
        self.check_tokens()
        if user_input is None and chat_history is None:
            raise ValueError("Either user_input or chat_history is required")
        if base64image is not None and user_input is None:
            raise ValueError("User input is required when providing an image")

        system_message = self.connector.get_system_message(
            self.system_prompt, self.name
        )
        created_message = None
        filtered_chat_history = self.filter_chat_history(chat_history, filters)
        messages = filtered_chat_history
        if user_input is not None or base64image is not None:
            created_message = self.connector.create_message(
                base64_image=base64image, text=user_input
            )
            messages = messages + created_message

        if model is None:
            result = self.connector.get_response(
                chat_history=messages,
                system_message=system_message,
                temperature=temperature,
                json_response=json_response,
                reasoning=reasoning,
                tools=tools,
            )
        else:
            result = self.connector.get_response(
                chat_history=messages,
                system_message=system_message,
                temperature=temperature,
                model=model,
                json_response=json_response,
                reasoning=reasoning,
                tools=tools,
            )

        agent_response = self.connector.get_agent_response(result, self.name)
        interaction_response = (created_message or []) + [agent_response]
        chat_history = (chat_history or []) + interaction_response
        self.update_tokens(self.connector._response_tokens)
        ## TODO modify this statement to return the content as we have changed the code to return a json object from connector.
        return result, chat_history
