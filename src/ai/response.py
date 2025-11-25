"""
Unified response format for web search across different AI providers.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class WebSearchResponse:
    """Unified response format for web search across different AI providers."""
    final_output: str  # The final generated text response
    citations: list[dict[str, str]]  # List of citations with 'url' and 'title'
    raw_response: Any  # The complete raw response from the API

    @property
    def text(self) -> str:
        """Alias for final_output for backward compatibility."""
        return self.final_output


# Backward compatibility alias
ClaudeResponse = WebSearchResponse
