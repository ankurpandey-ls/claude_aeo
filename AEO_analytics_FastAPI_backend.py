# AEO Analytics Backend
# AI Engine Optimization tool for tracking brand visibility across AI platforms

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="AEO Analytics API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG = {
    "concurrent_processing": os.getenv("CONCURRENT_PROCESSING", "true").lower() == "true",
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    "api_keys": {
        "exa": os.getenv("EXA_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
    }
}

# Request/Response Models
class AnalysisRequest(BaseModel):
    target_website: HttpUrl
    competitor_websites: List[HttpUrl]  # up to 4 competitors
    
class AnalysisResponse(BaseModel):
    status: str
    job_id: str
    message: str

class QueryResult(BaseModel):
    model: str
    prompt: str
    target_mentioned: bool
    target_snippet: Optional[str]
    competitors_mentioned: Dict[str, bool]
    competitors_snippets: Dict[str, Optional[str]]
    full_response: str
    timestamp: datetime

# Retry decorator for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def make_api_request(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> Dict:
    """Make API request with retry logic"""
    try:
        async with session.request(method, url, **kwargs) as response:
            if response.status == 429:
                raise Exception(f"Rate limit hit for {url}")
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {e}")
        raise

# Website Analysis Functions
async def analyze_website_with_exa(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Analyze website content using Exa API"""
    headers = {
        "x-api-key": CONFIG["api_keys"]["exa"],
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": f"site:{url}",
        "num_results": 1,
        "type": "auto",
        "contents": {
            "text": {"max_characters": 5000},
            "summary": {
                "query": "What are the main products, services, and value propositions of this company?"
            }
        }
    }
    
    try:
        result = await make_api_request(
            session, 
            "POST", 
            "https://api.exa.ai/search",
            headers=headers,
            json=payload
        )
        
        if result.get("results") and len(result["results"]) > 0:
            return {
                "url": url,
                "text": result["results"][0].get("text", "No content found"),
                "summary": result["results"][0].get("summary", "No summary available"),
                "error": False
            }
        else:
            return {
                "url": url,
                "text": "No content found",
                "summary": "No summary available",
                "error": True
            }
    except Exception as e:
        logger.error(f"Exa API error for {url}: {e}")
        return {
            "url": url,
            "text": "Error fetching content",
            "summary": "Error in analysis",
            "error": True
        }

async def extract_business_topics(session: aiohttp.ClientSession, website_data: Dict) -> Dict[str, List[str]]:
    """Extract business topics from website content using GPT-4"""
    headers = {
        "Authorization": f"Bearer {CONFIG['api_keys']['openai']}",
        "Content-Type": "application/json"
    }
    
    system_message = "You are an expert at analyzing company websites and extracting key business topics. Always respond with valid JSON. Today is June 2, 2025."
    
    user_message = f"""Analyze the following website content and identify 2-3 core business topics/themes:
    Website: {website_data['url']}
    Summary: {website_data['summary']}
    Main Content: {website_data['text']}
    
    Requirements:
    1. Identify 2-3 main product/service categories or value propositions
    2. Focus on specific, actionable topics (not generic like "customer service")
    3. Consider the unique selling points and key offerings
    
    Return a JSON object with this exact structure:
    {{
        "url": "the website url",
        "name": "company name",
        "topics": [
            "First specific topic related to their core offering",
            "Second key area of their business",
            "Third focus area if applicable"
        ]
    }}"""
    
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }
    
    try:
        result = await make_api_request(
            session,
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        content = json.loads(result["choices"][0]["message"]["content"])
        return content
    except Exception as e:
        logger.error(f"GPT topic extraction error: {e}")
        return {
            "url": website_data["url"],
            "name": "Unknown",
            "topics": ["General services", "Products"]
        }

# Query Generation Functions
async def generate_competitive_queries(session: aiohttp.ClientSession, all_topics: List[Dict]) -> List[str]:
    """Generate competitive analysis queries based on extracted topics"""
    headers = {
        "Authorization": f"Bearer {CONFIG['api_keys']['openai']}",
        "Content-Type": "application/json"
    }
    
    topics_json = json.dumps(all_topics, indent=2)
    
    system_message = """Today is June 3rd, 2025. You are an Agent specialized for Optimizing a products discoverability on LLMs such as ChatGPT, Claude and Gemini. The prompts should be middle of the funnel MOFU style prompts. Prompts themselves should not have specific product names in them."""
    
    user_message = f"""From these questions:
    {topics_json}
    
    Transform each question into a natural ChatGPT prompt that a user would type to get tool recommendations.
    
    Requirements:
    - Each prompt should be 15-25 words
    - Write like a real user asking ChatGPT for help
    - Include context and specific needs
    - Make it conversational and natural
    - Focus on getting AI to recommend specific tools
    
    Output exactly 20 prompts in JSON format:
    {{
        "prompts": [
            "natural user prompt 1",
            "natural user prompt 2",
            ...
        ]
    }}"""
    
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"}
    }
    
    try:
        result = await make_api_request(
            session,
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        content = json.loads(result["choices"][0]["message"]["content"])
        return content.get("prompts", [])[:20]  # Ensure max 20 prompts
    except Exception as e:
        logger.error(f"Query generation error: {e}")
        return ["What's the best video editing tool for beginners?"]  # Fallback

async def get_perplexity_related_questions(session: aiohttp.ClientSession, topic: str) -> List[str]:
    """Get related questions from Perplexity API"""
    headers = {
        "Authorization": f"Bearer {CONFIG['api_keys']['perplexity']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Today is June 2, 2025"},
            {"role": "user", "content": topic}
        ],
        "temperature": 0.2,
        "return_related_questions": True,
        "search_recency_filter": "month"
    }
    
    try:
        result = await make_api_request(
            session,
            "POST",
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        
        return result.get("related_questions", [])
    except Exception as e:
        logger.error(f"Perplexity API error: {e}")
        return []

# AI Model Query Functions
async def query_ai_model(session: aiohttp.ClientSession, model: str, prompt: str) -> Dict[str, Any]:
    """Query different AI models with the given prompt"""
    
    if model.startswith("openai/"):
        return await query_openai(session, prompt)
    else:
        return await query_openrouter(session, model, prompt)

async def query_openai(session: aiohttp.ClientSession, prompt: str) -> Dict[str, Any]:
    """Query OpenAI ChatGPT"""
    headers = {
        "Authorization": f"Bearer {CONFIG['api_keys']['openai']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        result = await make_api_request(
            session,
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        return {
            "text": result["choices"][0]["message"]["content"],
            "model": "openai/chatgpt-4o-latest",
            "error": False
        }
    except Exception as e:
        logger.error(f"OpenAI query error: {e}")
        return {"text": "", "model": "openai/chatgpt-4o-latest", "error": True}

async def query_openrouter(session: aiohttp.ClientSession, model: str, prompt: str) -> Dict[str, Any]:
    """Query models via OpenRouter"""
    headers = {
        "Authorization": f"Bearer {CONFIG['api_keys']['openrouter']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        result = await make_api_request(
            session,
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        return {
            "text": result["choices"][0]["message"]["content"],
            "model": model,
            "error": False
        }
    except Exception as e:
        logger.error(f"OpenRouter query error for {model}: {e}")
        return {"text": "", "model": model, "error": True}

# Tool Extraction and Comparison Functions
async def extract_tools_from_response(session: aiohttp.ClientSession, ai_response: Dict, prompt: str) -> Dict[str, Any]:
    """Extract mentioned tools from AI response"""
    headers = {
        "Authorization": f"Bearer {CONFIG['api_keys']['openai']}",
        "Content-Type": "application/json"
    }
    
    system_message = """You are an expert in AI content creation, SEO optimization, and marketing automation tools. You have deep knowledge of tools like Jasper AI, Copy.ai, Writesonic, Surfer SEO, Clearscope, MarketMuse, Semrush, Ahrefs, and many others. Always respond with valid JSON containing specific tool recommendations."""
    
    user_message = f"""Below is the response:
    ```
    {ai_response['text']}
    ```
    
    Model: {ai_response['model']}
    Question: 
    ```
    {prompt}
    ```
    
    Provide all tools/products mentioned in this response that directly answer this question.
    
    Requirements:
    - Recommend actual tools by name (e.g., "Jasper AI", "Surfer SEO", "Copy.ai")
    - NOT generic categories like "AI writing tool"
    - Each tool should directly address the user's question
    - Include market leaders and specialized solutions
    
    Return this exact JSON format:
    {{
        "model": "specific model mentioned above",
        "prompt": "specific question mentioned above",
        "tools": [
            {{
                "tool_name": "Specific Product Name",
                "tool_link": "Exact URL of the specific product"
            }}
        ]
    }}
    
    **CRITICAL REMINDER**: 
    1. You MUST ONLY extract tools/products available in the response, do not manufacture or assume any sort of data by yourself.
    2. Make sure that the tool_link is the exact URL of the corresponding tool_name in the output"""
    
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }
    
    try:
        result = await make_api_request(
            session,
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        return json.loads(result["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"Tool extraction error: {e}")
        return {"model": ai_response['model'], "prompt": prompt, "tools": []}

def get_main_domain(url: str) -> str:
    """Extract main domain from URL"""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or parsed.path.split('/')[0]
        
        # Handle two-level TLDs
        two_level_tlds = ['co.uk', 'com.au', 'co.in', 'co.za', 'com.br', 'co.jp']
        parts = hostname.split('.')
        
        if len(parts) >= 3:
            last_two = '.'.join(parts[-2:])
            if last_two in two_level_tlds:
                return '.'.join(parts[-3:])
        
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
            
        return hostname
    except:
        return url

def check_brand_mention(tool_url: str, brand_url: str) -> bool:
    """Check if a tool URL matches a brand URL"""
    return get_main_domain(tool_url).lower() == get_main_domain(brand_url).lower()

def extract_snippet(text: str, brand_domain: str, context_length: int = 100) -> Optional[str]:
    """Extract snippet mentioning the brand from AI response"""
    domain_variations = [
        brand_domain,
        brand_domain.replace('.com', ''),
        brand_domain.replace('.io', ''),
        brand_domain.replace('.ai', ''),
        brand_domain.split('.')[0]
    ]
    
    text_lower = text.lower()
    for variation in domain_variations:
        if variation.lower() in text_lower:
            # Find position and extract context
            pos = text_lower.find(variation.lower())
            start = max(0, pos - context_length)
            end = min(len(text), pos + len(variation) + context_length)
            return text[start:end].strip()
    
    return None

# Main Analysis Function
async def analyze_brand_visibility(websites: List[str]) -> List[QueryResult]:
    """Main function to analyze brand visibility across AI platforms"""
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Analyze all websites
        logger.info("Step 1: Analyzing websites with Exa API")
        if CONFIG["concurrent_processing"]:
            website_data = await asyncio.gather(
                *[analyze_website_with_exa(session, url) for url in websites]
            )
        else:
            website_data = []
            for url in websites:
                data = await analyze_website_with_exa(session, url)
                website_data.append(data)
        
        # Step 2: Extract business topics
        logger.info("Step 2: Extracting business topics")
        if CONFIG["concurrent_processing"]:
            topics_data = await asyncio.gather(
                *[extract_business_topics(session, data) for data in website_data]
            )
        else:
            topics_data = []
            for data in website_data:
                topics = await extract_business_topics(session, data)
                topics_data.append(topics)
        
        # Step 3: Generate queries
        logger.info("Step 3: Generating competitive queries")
        queries = await generate_competitive_queries(session, topics_data)
        
        # Also get Perplexity related questions for each topic
        all_perplexity_questions = []
        for topic_data in topics_data:
            for topic in topic_data.get("topics", []):
                questions = await get_perplexity_related_questions(session, topic)
                all_perplexity_questions.extend(questions)
        
        # Combine and limit queries
        all_queries = list(set(queries + all_perplexity_questions))[:20]
        
        # Step 4: Query AI models
        logger.info("Step 4: Querying AI models")
        ai_models = [
            "openai/chatgpt-4o-latest",
            "google/gemini-2.5-pro-preview:online",
            "x-ai/grok-3-beta:online",
            "anthropic/claude-sonnet-4:online",
            "perplexity/sonar"
        ]
        
        for query in all_queries[:2]:  # Limit to 2 queries for demo
            for model in ai_models:
                logger.info(f"Querying {model} with: {query[:50]}...")
                
                # Query the AI model
                ai_response = await query_ai_model(session, model, query)
                
                if ai_response["error"]:
                    logger.error(f"Failed to query {model}")
                    continue
                
                # Extract tools mentioned
                tools_data = await extract_tools_from_response(session, ai_response, query)
                
                # Analyze mentions
                target_mentioned = False
                target_snippet = None
                competitors_mentioned = {}
                competitors_snippets = {}
                
                target_url = websites[0]
                target_domain = get_main_domain(target_url)
                
                # Check if target is mentioned in tools
                for tool in tools_data.get("tools", []):
                    if check_brand_mention(tool.get("tool_link", ""), target_url):
                        target_mentioned = True
                        break
                
                # Extract snippet if mentioned
                if target_mentioned or target_domain.split('.')[0].lower() in ai_response["text"].lower():
                    target_snippet = extract_snippet(ai_response["text"], target_domain)
                    if target_snippet:
                        target_mentioned = True
                
                # Check competitors
                for i, comp_url in enumerate(websites[1:], 1):
                    comp_domain = get_main_domain(comp_url)
                    comp_mentioned = False
                    
                    for tool in tools_data.get("tools", []):
                        if check_brand_mention(tool.get("tool_link", ""), comp_url):
                            comp_mentioned = True
                            break
                    
                    if not comp_mentioned and comp_domain.split('.')[0].lower() in ai_response["text"].lower():
                        comp_snippet = extract_snippet(ai_response["text"], comp_domain)
                        if comp_snippet:
                            comp_mentioned = True
                            competitors_snippets[f"competitor_{i}"] = comp_snippet
                    else:
                        competitors_snippets[f"competitor_{i}"] = None
                    
                    competitors_mentioned[f"competitor_{i}"] = comp_mentioned
                
                # Create result
                result = QueryResult(
                    model=model,
                    prompt=query,
                    target_mentioned=target_mentioned,
                    target_snippet=target_snippet,
                    competitors_mentioned=competitors_mentioned,
                    competitors_snippets=competitors_snippets,
                    full_response=ai_response["text"],
                    timestamp=datetime.now()
                )
                
                results.append(result)
                
        return results

# FastAPI Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AEO Analytics API"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_brand(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start brand visibility analysis"""
    
    # Validate inputs
    if len(request.competitor_websites) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 competitor websites allowed")
    
    # Combine target and competitor websites
    all_websites = [str(request.target_website)] + [str(url) for url in request.competitor_websites]
    
    # Generate job ID
    job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Start analysis in background
    background_tasks.add_task(run_analysis, job_id, all_websites)
    
    return AnalysisResponse(
        status="started",
        job_id=job_id,
        message=f"Analysis started for {len(all_websites)} websites"
    )

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results by job ID"""
    # In a real implementation, this would fetch from a database
    # For now, return a placeholder
    return {
        "job_id": job_id,
        "status": "completed",
        "message": "Results would be fetched from database"
    }

async def run_analysis(job_id: str, websites: List[str]):
    """Run the analysis in background"""
    try:
        logger.info(f"Starting analysis for job {job_id}")
        results = await analyze_brand_visibility(websites)
        
        # In a real implementation, save to database
        # For now, just log
        logger.info(f"Analysis complete for job {job_id}: {len(results)} results")
        
        # Here you would save to Replit database or your chosen storage
        
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
