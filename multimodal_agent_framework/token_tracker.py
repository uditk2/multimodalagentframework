from abc import ABC, abstractmethod


class BaseTokenUsageTracker(ABC):
    """Abstract base class for token usage trackers."""

    @abstractmethod
    def track_token_usage(
        self, input_tokens=0, output_tokens=0, model_name=None, agent_id=None
    ):
        """Track token usage for a model and agent."""
        pass


class DefaultTokenUsageTracker(BaseTokenUsageTracker):
    """Default token usage tracker implementation."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def track_token_usage(
        self, input_tokens=0, output_tokens=0, model_name=None, agent_id=None
    ):
        """Track token usage for a model."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        print(
            f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Model: {model_name}, Agent: {agent_id}"
        )


token_tracker = DefaultTokenUsageTracker()
