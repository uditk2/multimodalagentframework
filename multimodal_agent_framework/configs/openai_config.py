from .base_config import BaseLLMConfig


class OpenAIConfig(BaseLLMConfig):
    def __init__(self, default_model: str = "gpt-5"):
        super().__init__(
            provider="openai",
            default_model=default_model,
            prompt_token_costs={
                "gpt-4o": 0.0000025,
                "gpt-4o-mini": 0.00000015,
                "o4-mini": 0.0000011,
                "o3-mini": 0.0000011,
                "o1": 0.000015,
                "gpt-4.1": 0.000002,
                "gpt-4.1-mini": 0.0000004,
                "o3": 0.000002,
                "gpt-5": 0.00000125,
                "gpt-5-mini": 0.00000025,
                "gpt-5-nano": 0.00000005,
                "gpt-4o-search-preview": 0.0,  # add entries as needed
                "gpt-4o-mini-search-preview": 0.0,
                "gpt-5-search-preview": 0.0,
            },
            completion_token_costs={
                "gpt-4o": 0.000010,
                "gpt-4o-mini": 0.0000006,
                "o4-mini": 0.0000044,
                "o3-mini": 0.0000044,
                "o1": 0.000060,
                "gpt-4.1": 0.000008,
                "gpt-4.1-mini": 0.0000016,
                "o3": 0.000008,
                "gpt-5": 0.000010,
                "gpt-5-mini": 0.000002,
                "gpt-5-nano": 0.0000004,
                "gpt-4o-search-preview": 0.0,  # add entries as needed
                "gpt-4o-mini-search-preview": 0.0,
                "gpt-5-search-preview": 0.0,
            },
            default_prompt_cost=0.0,
            default_completion_cost=0.0,
        )
