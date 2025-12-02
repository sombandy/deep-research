"""
VC Firm Investment Research Script

This script researches whether a VC firm has led or co-led any $100M+ Series B/C/D
investments in the past 5 years using GPT-4 and Claude Sonnet 4.5 with web search.
"""

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich import print_json
from tqdm import tqdm

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

    # Format links with title and URL for better selection
    formatted_links = []
    for link in supporting_links:
        if isinstance(link, dict):
            title = link.get('title', 'No title')
            url = link.get('url', '')
            formatted_links.append(f"- {title}\n  URL: {url}")
        else:
            formatted_links.append(f"- {link}")

    prompt = f"""Given the following research summary and supporting links, extract and structure the information:

RESEARCH SUMMARY:
{research_summary}

SUPPORTING LINKS:
{chr(10).join(formatted_links)}

Please extract:
1. Whether the VC led or co-led a $100M+ round in the last 5 years (Yes/No)
2. A concise 1-2 line summary including: round type, date, invested company name, and any co-investors
3. The 1-2 most credible links supporting (prioritize press releases, TechCrunch, Forbes, or other top business news sources)

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


def get_processed_firms(csv_path: str) -> set[str]:
    """Get set of already processed VC firms from CSV."""
    if not Path(csv_path).exists():
        return set()

    processed = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row['VC Firm Name'])
    return processed


def append_to_csv(csv_path: str, vc_firm: str, structured_output: StructuredInvestmentOutput):
    """Append research result to CSV file."""
    file_exists = Path(csv_path).exists()

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['VC Firm Name', 'Has Qualifying Investment', 'Summary', 'Supporting Links']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'VC Firm Name': vc_firm,
            'Has Qualifying Investment': 'Yes' if structured_output.has_qualifying_investment else 'No',
            'Summary': structured_output.summary,
            'Supporting Links': ' | '.join(structured_output.links) if structured_output.links else ''
        })


def main():
    parser = argparse.ArgumentParser(description="Research VC firm investments using web search")

    # Create mutually exclusive group for input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input-file",
        type=str,
        help="Path to file with VC firm names (one per line)"
    )
    input_group.add_argument(
        "-n", "--name",
        type=str,
        help="Single VC firm name to research"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model to use for research (default: claude-sonnet-4-5)"
    )
    parser.add_argument("-o", "--output-csv", type=str, help="Output CSV file path")
    parser.add_argument(
        "--max-searches",
        type=int,
        default=3,
        help="Maximum number of web searches per query (default: 2)"
    )

    args = parser.parse_args()

    # Get list of VC firms from either file or single name
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            vc_firms = [line.strip() for line in f if line.strip()]
    else:
        vc_firms = [args.name]

    # Get already processed firms only if output CSV is provided
    if args.output_csv:
        processed_firms = get_processed_firms(args.output_csv)
        remaining_firms = [firm for firm in vc_firms if firm not in processed_firms]

        if not remaining_firms:
            print("✅ All firms already processed!")
            return 0

        print(f"\n📋 Total firms: {len(vc_firms)}")
        print(f"✅ Already processed: {len(processed_firms)}")
        print(f"🔄 Remaining: {len(remaining_firms)}")
    else:
        remaining_firms = vc_firms
        print(f"\n📋 Total firms: {len(vc_firms)}")

    print(f"📊 Using model: {args.model}\n")

    # Initialize researcher
    if "gpt" in args.model.lower():
        researcher = OpenAIWebSearch(model=args.model)
    elif "claude" in args.model.lower():
        researcher = ClaudeWebSearch(model=args.model, max_uses=args.max_searches)
    else:
        print(f"❌ Unknown model provider for {args.model}")
        return 1

    # Process each VC firm with progress bar
    for vc_firm in tqdm(remaining_firms, desc="Researching VC firms"):
        try:
            print(f"Researching {vc_firm}")
            prompt = create_research_prompt(vc_firm)
            response = researcher.search(prompt)

            structured_output = create_structured_output(response.final_output, response.citations)
            print_json(data=structured_output.model_dump())
            
            if args.output_csv:
                append_to_csv(args.output_csv, vc_firm, structured_output)
            
            tqdm.write(f"✅ {vc_firm}: {'Yes' if structured_output.has_qualifying_investment else 'No'}")

        except Exception as e:
            tqdm.write(f"❌ Error processing {vc_firm}: {str(e)}")
            continue

    print(f"\n✅ Research complete! Results saved to: {args.output_csv}")
    return 0

if __name__ == "__main__":
    exit(main())
