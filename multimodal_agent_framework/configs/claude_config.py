from .base_config import BaseLLMConfig


class ClaudeConfig(BaseLLMConfig):
    def __init__(self, default_model: str = "claude-sonnet-4-latest"):
        # Claude pricing in the current code is flat per token
        super().__init__(
            provider="anthropic",
            default_model=default_model,
            prompt_token_costs={},
            completion_token_costs={},
            default_prompt_cost=0.000003,
            default_completion_cost=0.000015,
        )
