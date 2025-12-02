"""
Claude AI Helper with Web Search Support

This module provides a helper class to interact with Claude using Anthropic's SDK
with built-in web search capabilities and citation extraction.
"""

import json
from typing import Optional

from anthropic import Anthropic

from src.ai.response import WebSearchResponse


class ClaudeWebSearch:
    """Helper class for Claude with web search capabilities."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        max_uses: int = 3,
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
        self.max_uses = max_uses
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
                "name": "web_search",
                "max_uses": self.max_uses
            }
        ]

        # Make API call with web search enabled
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
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
        
        # Print number of web searches performed
        web_search_requests = 0
        if hasattr(response, 'usage') and response.usage:
            server_tool_use = getattr(response.usage, 'server_tool_use', None)
            if server_tool_use:
                web_search_requests = getattr(server_tool_use, 'web_search_requests', 0)
        
        print(f"🔍 Web searches performed: {web_search_requests}")

        return WebSearchResponse(
            final_output=final_output,
            citations=citations,
            raw_response=response.model_dump()
        )


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Claude web search with configurable max uses")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What is Nvidia stock price today? Also, what is the top news on Nvidia?",
        help="Search prompt/question"
    )
    parser.add_argument(
        "--max-uses",
        type=int,
        default=2,
        help="Maximum number of web searches (default: 2)"
    )
    
    args = parser.parse_args()

    claude = ClaudeWebSearch(
        max_uses=args.max_uses,
    )

    print(f"\nPrompt: {args.prompt}\n")
    print("Executing search...\n")
    
    response = claude.search(args.prompt)

    print("Raw response saved to tmp/claude_raw.json")
    with open("tmp/claude_raw.json", "w", encoding="utf-8") as f:
        json.dump(response.raw_response, f, indent=2)

    print("Final output:")
    print(response.final_output)

    print("Citations:")
    if response.citations:
        for i, citation in enumerate(response.citations, 1):
            print(f"{i}. {citation}")
    else:
        print("No citations found")


if __name__ == "__main__":
    main()