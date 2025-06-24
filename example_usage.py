# Example usage of the AEO Analytics API

import asyncio
import aiohttp
import json
from datetime import datetime

# API base URL (adjust if running on different host/port)
API_BASE_URL = "http://localhost:8000"

async def test_analysis():
    """Example of how to use the AEO Analytics API"""
    
    # Define the websites to analyze
    analysis_request = {
        "target_website": "https://invideo.io",
        "competitor_websites": [
            "https://www.veed.io",
            "https://www.synthesia.io",
            "https://www.canva.com",
            "https://fliki.ai"
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        # 1. Start the analysis
        print("Starting brand visibility analysis...")
        async with session.post(
            f"{API_BASE_URL}/analyze",
            json=analysis_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            job_id = result["job_id"]
            print(f"Analysis started with job ID: {job_id}")
        
        # 2. Wait a bit for processing (in real app, you'd poll or use webhooks)
        print("Waiting for analysis to complete...")
        await asyncio.sleep(30)  # Wait 30 seconds
        
        # 3. Fetch results
        print("Fetching results...")
        async with session.get(f"{API_BASE_URL}/results/{job_id}") as response:
            results = await response.json()
            print(f"Results: {json.dumps(results, indent=2)}")

async def test_health_check():
    """Test the health check endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/") as response:
            result = await response.json()
            print(f"Health check: {result}")

# Example of how to process results
def analyze_results(results):
    """Process and summarize the analysis results"""
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "models_used": set(),
        "target_mentions": 0,
        "competitor_mentions": {},
        "queries_where_target_mentioned": [],
        "queries_where_competitors_beat_target": []
    }
    
    for result in results:
        summary["models_used"].add(result["model"])
        
        if result["target_mentioned"]:
            summary["target_mentions"] += 1
            summary["queries_where_target_mentioned"].append({
                "query": result["prompt"],
                "model": result["model"],
                "snippet": result["target_snippet"]
            })
        
        # Check if any competitor was mentioned but not the target
        competitor_mentioned = any(result["competitors_mentioned"].values())
        if competitor_mentioned and not result["target_mentioned"]:
            summary["queries_where_competitors_beat_target"].append({
                "query": result["prompt"],
                "model": result["model"],
                "competitors": [k for k, v in result["competitors_mentioned"].items() if v]
            })
        
        # Count competitor mentions
        for comp, mentioned in result["competitors_mentioned"].items():
            if comp not in summary["competitor_mentions"]:
                summary["competitor_mentions"][comp] = 0
            if mentioned:
                summary["competitor_mentions"][comp] += 1
    
    # Calculate visibility rates
    summary["target_visibility_rate"] = (summary["target_mentions"] / summary["total_queries"]) * 100 if summary["total_queries"] > 0 else 0
    
    for comp in summary["competitor_mentions"]:
        mentions = summary["competitor_mentions"][comp]
        summary[f"{comp}_visibility_rate"] = (mentions / summary["total_queries"]) * 100 if summary["total_queries"] > 0 else 0
    
    return summary

# Example of generating a report
def generate_report(summary):
    """Generate a human-readable report from the analysis summary"""
    
    report = f"""
=== AEO Analytics Report ===
Generated: {summary['timestamp']}

Overview:
- Total queries analyzed: {summary['total_queries']}
- AI models used: {', '.join(summary['models_used'])}
- Target brand visibility rate: {summary['target_visibility_rate']:.1f}%

Target Brand Performance:
- Mentioned in {summary['target_mentions']} out of {summary['total_queries']} queries
- Visibility rate: {summary['target_visibility_rate']:.1f}%

Competitor Performance:
"""
    
    for comp, mentions in summary['competitor_mentions'].items():
        rate = summary.get(f"{comp}_visibility_rate", 0)
        report += f"- {comp}: {mentions} mentions ({rate:.1f}% visibility)\n"
    
    report += f"\nQueries where target was mentioned ({len(summary['queries_where_target_mentioned'])}):\n"
    for item in summary['queries_where_target_mentioned'][:5]:  # Show first 5
        report += f"- \"{item['query'][:50]}...\" on {item['model']}\n"
    
    if summary['queries_where_competitors_beat_target']:
        report += f"\nQueries where competitors outperformed target ({len(summary['queries_where_competitors_beat_target'])}):\n"
        for item in summary['queries_where_competitors_beat_target'][:5]:  # Show first 5
            report += f"- \"{item['query'][:50]}...\" - Mentioned: {', '.join(item['competitors'])}\n"
    
    return report

# Run the example
if __name__ == "__main__":
    print("AEO Analytics API Example")
    print("=" * 50)
    
    # Run health check
    asyncio.run(test_health_check())
    
    # Run analysis
    asyncio.run(test_analysis())
    
    # Example of processing results (with mock data for demonstration)
    mock_results = [
        {
            "model": "openai/chatgpt-4o-latest",
            "prompt": "What's the best AI video editor for creating marketing content?",
            "target_mentioned": True,
            "target_snippet": "InVideo is a great choice for AI-powered video editing...",
            "competitors_mentioned": {"competitor_1": True, "competitor_2": False, "competitor_3": True, "competitor_4": False},
            "competitors_snippets": {"competitor_1": "Veed.io offers...", "competitor_2": None, "competitor_3": "Canva's video editor...", "competitor_4": None},
            "full_response": "Full AI response here...",
            "timestamp": "2025-06-02T10:00:00"
        },
        {
            "model": "google/gemini-2.5-pro-preview:online",
            "prompt": "I need an AI tool to create professional videos quickly",
            "target_mentioned": False,
            "target_snippet": None,
            "competitors_mentioned": {"competitor_1": True, "competitor_2": True, "competitor_3": False, "competitor_4": False},
            "competitors_snippets": {"competitor_1": "Veed.io is perfect for...", "competitor_2": "Synthesia specializes in...", "competitor_3": None, "competitor_4": None},
            "full_response": "Full AI response here...",
            "timestamp": "2025-06-02T10:01:00"
        }
    ]
    
    print("\nExample Report Generation:")
    print("=" * 50)
    summary = analyze_results(mock_results)
    report = generate_report(summary)
    print(report)
