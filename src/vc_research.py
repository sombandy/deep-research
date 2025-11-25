"""
VC Firm Investment Research Script

This script researches whether a VC firm has led or co-led any $100M+ Series B/C/D
investments in the past 5 years using GPT-4 and Claude Sonnet 4.5 with web search.
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Literal, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.ai import ClaudeWebSearch, OpenAIWebSearch

# Load environment variables from .env file
load_dotenv()


@dataclass
class ResearchResult:
    """Result from a single model's research."""
    model_name: str
    summary: str
    supporting_links: list[str]
    raw_response: str
    has_qualifying_investment: bool


class InvestmentFinding(BaseModel):
    """Structured output for investment findings."""
    has_qualifying_investment: bool = Field(
        description="Whether the firm has made a qualifying $100M+ Series B/C/D investment as lead or co-lead in the past 5 years"
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Name of the company that received the investment"
    )
    round_type: Optional[Literal["Series B", "Series C", "Series D"]] = Field(
        default=None,
        description="Type of funding round"
    )
    amount_millions: Optional[float] = Field(
        default=None,
        description="Investment amount in millions of USD"
    )
    investment_date: Optional[str] = Field(
        default=None,
        description="Date of investment in YYYY-MM-DD format or 'YYYY-MM' if day unknown"
    )
    firm_role: Optional[Literal["led", "co-led"]] = Field(
        default=None,
        description="Whether the firm led or co-led the round"
    )
    summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence summary of the qualifying investment including company, round, amount, date, and role"
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="List of credible source URLs (press releases, major business publications)"
    )


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
   - DO NOT include your thinking process, search steps, or "Now I'll..." statements
   - Start directly with your findings
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


def research_with_model(
    model_name: str,
    vc_firm_name: str,
    model_provider: str = "openai"
) -> ResearchResult:
    """
    Research VC firm investments using a specific model with web search.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
        vc_firm_name: Name of the VC firm to research
        model_provider: Provider name ("openai" or "anthropic")

    Returns:
        ResearchResult with findings
    """
    print(f"\n{'='*60}")
    print(f"Starting research with {model_name}")
    print(f"{'='*60}\n")

    # Create the research prompt
    prompt = create_research_prompt(vc_firm_name)

    # Use different approaches based on provider
    if model_provider == "anthropic":
        # Use Claude helper with native web search support
        print(f"Using Claude native web search API...")
        claude = ClaudeWebSearch(model=model_name)

        response = claude.search(prompt)
        text_response = response.final_output
        # Convert citations to list of URLs
        citations = [cite.get('url', '') for cite in response.citations if cite.get('url')]

    elif model_provider == "openai":
        # Use OpenAI helper with native web search support
        print(f"Using OpenAI native web search API...")
        openai = OpenAIWebSearch(model=model_name)

        response = openai.search(prompt)
        text_response = response.final_output
        # Convert citations to list of URLs
        citations = [cite.get('url', '') for cite in response.citations if cite.get('url')]

    else:
        raise ValueError(f"Unknown model provider: {model_provider}")

    # Parse the response to determine if there's a qualifying investment
    has_qualifying = False
    summary = text_response

    # Simple heuristic: check if the response mentions specific investment details
    if any(keyword in text_response.lower() for keyword in ["series b", "series c", "series d"]):
        if any(keyword in text_response.lower() for keyword in ["led", "co-led", "$100m", "$100 m"]):
            has_qualifying = True

    return ResearchResult(
        model_name=model_name,
        summary=summary,
        supporting_links=citations if citations else ["NA"],
        raw_response=text_response,
        has_qualifying_investment=has_qualifying
    )


def format_output(results: list[ResearchResult], vc_firm_name: str) -> str:
    """Format the research results for display."""

    output = []
    output.append("=" * 80)
    output.append(f"VC FIRM INVESTMENT RESEARCH REPORT")
    output.append(f"Firm: {vc_firm_name}")
    output.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    output.append("")

    for i, result in enumerate(results, 1):
        output.append(f"\n{'─'*80}")
        output.append(f"RESULT #{i}: {result.model_name}")
        output.append(f"{'─'*80}\n")

        output.append("QUALIFYING INVESTMENT FOUND:")
        output.append(f"  {result.has_qualifying_investment}")
        output.append("")

        output.append("SUMMARY:")
        # Format summary with proper wrapping
        summary_lines = result.summary.split('\n')
        for line in summary_lines:
            if line.strip():
                output.append(f"  {line.strip()}")
        output.append("")

        output.append("SUPPORTING ARTICLE LINKS:")
        if result.supporting_links and result.supporting_links != ["NA"]:
            for link in result.supporting_links:
                output.append(f"  • {link}")
        else:
            output.append("  • NA")
        output.append("")

    output.append("=" * 80)
    output.append("END OF REPORT")
    output.append("=" * 80)

    return "\n".join(output)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Research VC firm investments using GPT-4 and Claude Sonnet 4.5 with web search"
    )
    parser.add_argument(
        "vc_firm",
        type=str,
        help="Name of the VC firm to research (e.g., 'Sequoia Capital', 'Andreessen Horowitz')"
    )
    parser.add_argument(
        "-m", "--models",
        type=str,
        nargs="+",
        default=["gpt-5.1", "claude-sonnet-4-5"],
        help="Models to use for research (default: gpt-4o claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (optional, defaults to stdout)"
    )

    args = parser.parse_args()

    print(f"\n🔍 Researching VC firm: {args.vc_firm}")
    print(f"📊 Using models: {', '.join(args.models)}\n")

    # Conduct research with each model
    results = []

    for model in args.models:
        try:
            # Determine provider
            if "gpt" in model.lower():
                provider = "openai"
            elif "claude" in model.lower():
                provider = "anthropic"
            else:
                print(f"⚠️  Unknown model provider for {model}, skipping...")
                continue

            result = research_with_model(model, args.vc_firm, provider)
            results.append(result)

            print(f"✅ Completed research with {model}\n")

        except Exception as e:
            import traceback
            print(f"❌ Error with {model}: {str(e)}")
            print(f"Full traceback:\n{traceback.format_exc()}\n")
            continue

    # Format and display results
    if results:
        output_text = format_output(results, args.vc_firm)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"\n📄 Results saved to: {args.output}")
        else:
            print(output_text)

        return 0
    else:
        print("❌ No results obtained from any model")
        return 1


if __name__ == "__main__":
    exit(main())
