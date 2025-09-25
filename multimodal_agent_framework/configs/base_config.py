from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from ..logging_config import get_logger

logger = get_logger()


@dataclass
class BaseLLMConfig:
    """
    Base configuration for LLM providers.

    Holds default model and token pricing maps. Connectors use this
    to compute costs and avoid hard-coded pricing logic.
    """

    provider: str
    default_model: Optional[str] = None
    prompt_token_costs: Dict[str, float] = field(default_factory=dict)
    completion_token_costs: Dict[str, float] = field(default_factory=dict)
    default_prompt_cost: Optional[float] = None
    default_completion_cost: Optional[float] = None

    def get_token_costs(self, model: Optional[str]) -> Tuple[float, float]:
        """
        Return per-token costs (prompt_cost, completion_cost) for a given model.

        If the model is unknown, returns configured defaults or (0.0, 0.0),
        and logs a warning to aid diagnostics without breaking execution.
        """
        model_key = model or self.default_model
        in_cost = self.prompt_token_costs.get(model_key)
        out_cost = self.completion_token_costs.get(model_key)

        if in_cost is None:
            in_cost = (
                self.default_prompt_cost
                if self.default_prompt_cost is not None
                else 0.0
            )
        if out_cost is None:
            out_cost = (
                self.default_completion_cost
                if self.default_completion_cost is not None
                else 0.0
            )

        if (model_key not in self.prompt_token_costs) or (
            model_key not in self.completion_token_costs
        ):
            logger.debug(
                f"Pricing not found for provider={self.provider}, model={model_key}. "
                f"Using defaults prompt={in_cost}, completion={out_cost}."
            )
        return in_cost, out_cost
