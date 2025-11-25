"""
VC Firm Investment Research Script

This script researches whether a VC firm has led or co-led any $100M+ Series B/C/D
investments in the past 5 years using GPT-4 and Claude Sonnet 4.5 with web search.
"""

import argparse
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich import print_json

from src.ai import ClaudeWebSearch, OpenAIWebSearch


# Load environment variables from .env file
load_dotenv()

class StructuredInvestmentOutput(BaseModel):
    """Structured output with Yes/No, summary, and credible links."""
    has_qualifying_investment: bool = Field(
        description="Whether the VC lead or co-lead $100M+ round in the last 5 years"
    )
    summary: str = Field(
        description="Max 1-2 line summary of the investment, including round type, date, invested company name, any co-investor"
    )
    links: list[str] = Field(
        max_length=2,
        description="Max 1-2 links from the most credible sources (press release, TechCrunch, Forbes or any top rated business news website)"
    )


def create_structured_output(
    research_summary: str,
    supporting_links: list[str]
) -> StructuredInvestmentOutput:
    """
    Create structured output from research summary and links.

    Args:
        research_summary: The research summary text
        supporting_links: List of supporting URLs

    Returns:
        StructuredInvestmentOutput with yes/no, summary, and top credible links
    """
    client = OpenAI()

    prompt = f"""Given the following research summary and supporting links, extract and structure the information:

RESEARCH SUMMARY:
{research_summary}

SUPPORTING LINKS:
{chr(10).join(f"- {link}" for link in supporting_links)}

Please extract:
1. Whether the VC led or co-led a $100M+ round in the last 5 years (Yes/No)
2. A concise 1-2 line summary including: round type, date, invested company name, and any co-investors
3. The 1-2 most credible links (prioritize press releases, TechCrunch, Forbes, or other top business news sources)

If the answer is NO, provide a brief explanation in the summary field and return empty links list."""

    completion = client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a financial research analyst specializing in venture capital investments."},
            {"role": "user", "content": prompt}
        ],
        response_format=StructuredInvestmentOutput
    )

    return completion.choices[0].message.parsed


def create_research_prompt(vc_firm_name: str) -> str:
    """Create the research prompt for the models."""

    current_date = datetime.now()
    five_years_ago = current_date - timedelta(days=5*365)

    prompt = f"""You are a professional investment research analyst. Your task is to research whether the venture capital firm "{vc_firm_name}" has led or co-led any $100M+ Series B, Series C, or Series D investment within the past 5 years (from {five_years_ago.strftime('%B %Y')} to {current_date.strftime('%B %Y')}).

CRITICAL RESEARCH REQUIREMENTS:

1. ONLY use authoritative external sources found through web search:
   - The firm's official press releases or news pages
   - Major business publications (TechCrunch, Bloomberg, Reuters, VentureBeat, The Information, WSJ, Forbes, etc.)
   - Reputable financial news sources

2. DO NOT use your training data, prior knowledge, assumptions, or guesses
   - If you cannot find credible web sources, state that no qualifying investment was found
   - DO NOT fabricate investments, dates, companies, or round details

3. VERIFICATION REQUIREMENTS:
   - The investment must be $100M or more
   - Must be Series B, Series C, or Series D round (not Series A, seed, or later stages)
   - {vc_firm_name} must have LED or CO-LED the round (not just participated)
   - Must have occurred within the past 5 years
   - Must be verified by at least one credible source URL


5. OUTPUT REQUIREMENTS:
   - Provide a clear 2-3 sentence summary if a qualifying investment is found
   - Include: invested company name, round type, amount, date, and whether they led or co-led
   - List only the most relevant source URLs (3-5 maximum from official sources or major publications)
   - If no qualifying investment found, clearly state this and explain what was searched

IMPORTANT: Be thorough but precise. Only report investments you can verify through credible web sources found in your search. If sources conflict, prioritize the firm's official press releases and major business publications.

OUTPUT FORMAT:
- Start with "YES" or "NO" to indicate if a qualifying investment was found
- Then provide the summary and source URLs
- Do NOT show your search process or thinking steps

Begin your research now using web search to find qualifying investments by {vc_firm_name}."""

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Research VC firm investments using web search")
    parser.add_argument("vc_firm", type=str, help="Name of the VC firm to research")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gpt-5.1",
        help="Model to use for research (default: gpt-5.1)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (optional, defaults to stdout)"
    )

    args = parser.parse_args()

    if "gpt" in args.model.lower():
        researcher = OpenAIWebSearch(model=args.model)
    elif "claude" in args.model.lower():
        researcher = ClaudeWebSearch(model=args.model)
    else:
        print(f"❌ Unknown model provider for {args.model}")
        return 1


    print(f"\n🔍 Researching VC firm: {args.vc_firm}")
    print(f"📊 Using model: {args.model}\n")

    prompt = create_research_prompt(args.vc_firm)
    response = researcher.search(prompt)

    print(f"Model summary: {response.final_output}")
    print_json(data=response.citations)

    print("Structured output:")
    print("=" * 60)
    structured_output = create_structured_output(response.final_output, response.citations)
    print_json(data=structured_output.model_dump())

if __name__ == "__main__":
    exit(main())
