"""
OpenAI Helper with Web Search Support

This module provides a helper class to interact with OpenAI using the OpenAI SDK
with built-in web search capabilities and citation extraction using the Responses API.
"""

import json
import sys
from pathlib import Path
import warnings
from typing import Optional

from openai import OpenAI

from src.ai.response import WebSearchResponse



class OpenAIWebSearch:
    """Helper class for OpenAI with web search capabilities using Responses API."""

    def __init__(
        self,
        model: str = "gpt-5.1"
    ):
        """
        Initialize OpenAI with web search support.

        Args:
            model: OpenAI model name (default: gpt-5.1)
        """
        self.model = model
        self.client = OpenAI()

    def search(self, prompt: str, system_prompt: Optional[str] = None) -> WebSearchResponse:
        """
        Execute a search query with OpenAI using web search tool via Responses API.

        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to guide OpenAI's behavior

        Returns:
            WebSearchResponse with final_output, citations, and raw response
        """
        # Combine system prompt and user prompt if system prompt is provided
        full_input = prompt
        if system_prompt:
            full_input = f"{system_prompt}\n\n{prompt}"

        # Make API call with web search enabled using Responses API
        response = self.client.responses.create(
            model=self.model,
            input=full_input,
            tools=[{"type": "web_search"}],
        )

        # Extract text and citations from response
        final_output = ""
        citations = []


        # Extract citations from annotations in message content blocks
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                # Only look for message items with content
                if hasattr(item, 'type') and item.type == "message":
                    if hasattr(item, 'content') and item.content:
                        for content_block in item.content:
                            # Extract text if not already set
                            if not final_output and hasattr(content_block, 'text'):
                                final_output = content_block.text

                            # Extract citations from annotations
                            if hasattr(content_block, 'annotations') and content_block.annotations:
                                for annotation in content_block.annotations:
                                    if hasattr(annotation, 'type') and annotation.type == "url_citation":
                                        citation = {}
                                        if hasattr(annotation, 'url'):
                                            citation['url'] = annotation.url
                                        if hasattr(annotation, 'title'):
                                            citation['title'] = annotation.title
                                        # Only add if we have both url and title
                                        if citation.get('url') and citation.get('title'):
                                            if citation not in citations:
                                                citations.append(citation)

        # Serialize response without triggering pydantic warnings
        raw_response = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_response = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        except Exception:
            raw_response = response

        return WebSearchResponse(
            final_output=final_output,
            citations=citations,
            raw_response=raw_response
        )


def main():
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) == 2:
        prompt = sys.argv[1]
    else:
        prompt = "What is Nvidia stock price today? Also, what is the top news on Nvidia?"

    openai_search = OpenAIWebSearch()

    print(f"\nPrompt: {prompt}\n")
    print("Executing search...\n")

    response = openai_search.search(prompt)

    print("Raw response saved to tmp/openai_raw.json")
    with open("tmp/openai_raw.json", "w", encoding="utf-8") as f:
        json.dump(response.raw_response, f, indent=2)

    print("Final output:")
    print(response.final_output)

    print("Citations:")
    if response.citations:
        for i, citation in enumerate(response.citations, 1):
            title = citation.get('title', 'No title')
            url = citation.get('url', 'No URL')
            print(f"{i}. {title}")
            print(f"   {url}")
    else:
        print("No citations found")


if __name__ == "__main__":
    main()
