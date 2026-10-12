"""What API calls cost: configured prices with the Batch API discount, and estimates made before a batch runs."""

import statistics
from dataclasses import dataclass
from typing import Any, Callable

from reflective_qa.paths import load_config


@dataclass(frozen=True)
class Costs:
    prices: dict[str, dict[str, float]]
    batch_discount: float
    assumed_output_tokens: int

    @classmethod
    def from_config(cls) -> "Costs":
        return cls(**load_config("answering")["costs"])

    def dollars(self, model: str, input_tokens: float, output_tokens: float) -> float:
        price = self.prices[model]
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6 * self.batch_discount

    def recorder(self, parse: Callable[[str, Any], Any], model: str) -> Callable[[str, Any], dict]:
        """A function that parses one Batch API result and files it as a row with what it cost."""
        def record(question_id: str, message: Any) -> dict:
            reply = parse(question_id, message)
            return reply.to_row(self.dollars(model, reply.input_tokens, reply.output_tokens))
        return record


class CostEstimator:
    """Batch cost of requests before they run. Input tokens are counted with the free count_tokens endpoint on an
    evenly spaced sample; output tokens are the mean of earlier results when there are any, else an assumption."""

    def __init__(self, client: Any, costs: Costs, sample: int = 50):
        self.client, self.costs, self.sample = client, costs, sample

    def input_tokens(self, requests: dict[str, dict]) -> float:
        """Mean input tokens per request."""
        ids = list(requests)
        sampled = ids[:: max(1, len(ids) // self.sample)][:self.sample]
        return statistics.mean(self.client.messages.count_tokens(**{k: v for k, v in requests[qid].items() if k != "max_tokens"})
                               .input_tokens for qid in sampled)

    def output_tokens(self, rows: list[dict], default: float | None = None) -> float:
        """Mean output tokens of earlier results; `default`, or the configured assumption, if there are none."""
        return self.mean_output(rows, self.costs.assumed_output_tokens if default is None else default)

    @staticmethod
    def mean_output(rows: list[dict], default: float) -> float:
        return statistics.mean(r["output_tokens"] for r in rows) if rows else default

    def estimate(self, model: str, requests: dict[str, dict], output_tokens: float) -> dict:
        """Mean input tokens per request (counted), and the dollars for all the requests."""
        if not requests:
            return {"input_tokens": 0, "dollars": 0.0}
        input_tokens = self.input_tokens(requests)
        return {"input_tokens": input_tokens,
                "dollars": self.costs.dollars(model, input_tokens * len(requests), output_tokens * len(requests))}

    def dollars(self, model: str, requests: dict[str, dict], output_tokens: float) -> float:
        return self.estimate(model, requests, output_tokens)["dollars"]
