"""
Claude AI Helper with Web Search Support

This module provides a helper class to interact with Claude using Anthropic's SDK
with built-in web search capabilities and citation extraction.
"""

import sys
from typing import Optional

from anthropic import Anthropic

from src.ai.response import WebSearchResponse


class ClaudeWebSearch:
    """Helper class for Claude with web search capabilities."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        """
        Initialize Claude with web search support.

        Args:
            model: Claude model name (default: claude-sonnet-4-5)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = Anthropic()

    def search(self, prompt: str, system_prompt: Optional[str] = None) -> WebSearchResponse:
        """
        Execute a search query with Claude using web search tool.

        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to guide Claude's behavior

        Returns:
            WebSearchResponse with final_output, citations, and raw response
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Define web search tool
        # Note: Using the latest web_search tool version
        # The name field is required for web_search tools
        tools = [
            {
                "type": "web_search_20250305",
                "name": "web_search"
            }
        ]

        # Make API call with web search enabled
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "tools": tools,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        # Extract text and citations from response
        # Claude's response pattern: [thinking text blocks] -> [tool use] -> [results] -> [final answer]
        # We only want the last text block(s) which contain the final answer with citations
        text_blocks = []
        citations = []
        citations_seen = set()  # Track unique citations by URL

        for block in response.content:
            # Collect all text blocks
            if block.type == "text":
                text_blocks.append(block)

        # Use only the last text block(s) after the last search result
        # Find the index of the last web_search_tool_result
        last_search_idx = -1
        for i, block in enumerate(response.content):
            if block.type == "web_search_tool_result":
                last_search_idx = i

        # Get text blocks that come after the last search
        final_text_blocks = []
        if last_search_idx >= 0:
            for block in response.content[last_search_idx + 1:]:
                if block.type == "text":
                    final_text_blocks.append(block)
        else:
            # No search results found, use all text blocks
            final_text_blocks = text_blocks

        # Extract text and citations from final text blocks
        final_text_parts = []
        for block in final_text_blocks:
            final_text_parts.append(block.text)

            # Extract citations from text blocks (Claude includes citations here)
            if hasattr(block, 'citations') and block.citations:
                for citation in block.citations:
                    if hasattr(citation, 'type') and citation.type == "web_search_result_location":
                        url = getattr(citation, 'url', None)
                        title = getattr(citation, 'title', None)

                        # Only add unique citations with both url and title
                        if url and title and url not in citations_seen:
                            citations.append({"url": url, "title": title})
                            citations_seen.add(url)

        final_output = "\n".join(final_text_parts)

        return WebSearchResponse(
            final_output=final_output,
            citations=citations,
            raw_response=response.model_dump()
        )


def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    if len(sys.argv) == 2:
        prompt = sys.argv[1]
    else:
        prompt = "What is Nvidia stock price today? Also, what is the top news on Nvidia?"

    claude = ClaudeWebSearch(
        model="claude-sonnet-4-5-20250929",
        temperature=0.1,
    )

    print(f"\nPrompt: {prompt}\n")
    print("Executing search...\n")
    
    response = claude.search(prompt)

    print("=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response.text)
    print("\n" + "=" * 80)
    print("CITATIONS:")
    print("=" * 80)

    if response.citations:
        for i, citation in enumerate(response.citations, 1):
            print(f"{i}. {citation}")
    else:
        print("No citations found")


if __name__ == "__main__":
    main()