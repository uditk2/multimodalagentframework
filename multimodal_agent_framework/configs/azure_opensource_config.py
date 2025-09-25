from .base_config import BaseLLMConfig


class AzureOpenSourceConfig(BaseLLMConfig):
    def __init__(self, default_model: str = "Codestral-2501"):
        super().__init__(
            provider="azure-opensource",
            default_model=default_model,
            prompt_token_costs={
                "Codestral-2501": 0.0000003,
            },
            completion_token_costs={
                "Codestral-2501": 0.0000009,
            },
            default_prompt_cost=0.0,
            default_completion_cost=0.0,
        )
