"""AI module for deep research platform."""

from .response import WebSearchResponse, ClaudeResponse
from .claude import ClaudeWebSearch
from .openai_ws import OpenAIWebSearch

__all__ = [
    'ClaudeWebSearch',
    'OpenAIWebSearch',
    'WebSearchResponse',
    'ClaudeResponse',  # Backward compatibility
]
