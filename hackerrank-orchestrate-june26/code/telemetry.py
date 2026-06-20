import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TelemetryTracker:
    def __init__(self):
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.images_processed = set()
        self.latencies: List[float] = []
        self.costs = 0.0

    def track_call(self, model_name: str, input_tok: int, output_tok: int, images: List[str], duration: float):
        self.model_calls += 1
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        for img in images:
            self.images_processed.add(img)
        self.latencies.append(duration)
        
        # Calculate cost based on current standard pricing guidelines
        cost = 0.0
        name_lower = model_name.lower()
        if "gemini-2.5-flash" in name_lower or "gemini-2.0-flash" in name_lower or "gemini-1.5-flash" in name_lower:
            # Gemini 2.5 Flash pricing assumptions: $0.075 / 1M input, $0.30 / 1M output
            cost = (input_tok * 0.075 / 1_000_000) + (output_tok * 0.30 / 1_000_000)
        elif "gpt-4o-mini" in name_lower:
            # GPT-4o-mini pricing assumptions: $0.15 / 1M input, $0.60 / 1M output
            cost = (input_tok * 0.15 / 1_000_000) + (output_tok * 0.60 / 1_000_000)
        elif "claude-3-5-sonnet" in name_lower:
            # Claude 3.5 Sonnet pricing assumptions: $3.00 / 1M input, $15.00 / 1M output
            cost = (input_tok * 3.00 / 1_000_000) + (output_tok * 15.00 / 1_000_000)
        else:
            # Default fallback to Gemini 2.5 Flash rates
            cost = (input_tok * 0.075 / 1_000_000) + (output_tok * 0.30 / 1_000_000)
            
        self.costs += cost

    def get_summary(self) -> Dict[str, Any]:
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        total_images = len(self.images_processed)
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "images_processed_count": total_images,
            "total_latency_seconds": sum(self.latencies),
            "average_latency_seconds": avg_latency,
            "estimated_cost_usd": self.costs
        }

def retry_api_call(max_retries: int = 5, initial_backoff: float = 1.0):
    """Decorator or helper to retry API calls with exponential backoff on failure (e.g. rate limits)."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            backoff = initial_backoff
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    # Retry on 429, rate limit, or resource exhausted errors
                    is_rate_limit = "429" in err_msg or "rate" in err_msg or "exhausted" in err_msg or "overloaded" in err_msg
                    if is_rate_limit and attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit: {e}. Retrying in {backoff:.2f}s...")
                        time.sleep(backoff)
                        backoff *= 2.0
                    else:
                        raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator
