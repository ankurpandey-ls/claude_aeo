# AEO Analytics Backend

AI Engine Optimization (AEO) analytics tool for tracking brand visibility across multiple AI platforms (ChatGPT, Claude, Gemini, Perplexity, Grok).

## Features

- **Multi-AI Platform Analysis**: Queries ChatGPT, Claude, Gemini, Perplexity, and Grok
- **Competitive Benchmarking**: Compare your brand visibility against up to 4 competitors
- **Smart Query Generation**: Automatically generates relevant MOFU (Middle of Funnel) queries
- **Detailed Insights**: Captures both presence markers and actual response snippets
- **Async Processing**: Configurable concurrent/sequential processing for optimal performance
- **Retry Logic**: Automatic retry with exponential backoff for failed API calls
- **RESTful API**: Built with FastAPI for easy integration

## Architecture

The system follows this workflow:
1. **Website Analysis**: Uses Exa API to analyze website content and extract key information
2. **Topic Extraction**: Uses GPT-4 to identify 2-3 core business topics from each website
3. **Query Generation**: 
   - Generates competitive analysis queries using GPT-4
   - Fetches related questions from Perplexity
   - Combines and deduplicates to create up to 20 queries
4. **AI Model Querying**: Sends each query to 5 different AI models
5. **Response Analysis**: 
   - Extracts mentioned tools/brands from responses
   - Checks for brand mentions and captures snippets
   - Compares target brand vs competitors
6. **Results Storage**: Stores results with presence markers and snippets

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd aeo-analytics-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Configuration

### Required API Keys
- `EXA_API_KEY`: For website content analysis
- `OPENAI_API_KEY`: For GPT-4 topic extraction and query generation
- `OPENROUTER_API_KEY`: For accessing Claude, Gemini, Grok via OpenRouter
- `PERPLEXITY_API_KEY`: For Perplexity API access

### Optional Configuration
- `CONCURRENT_PROCESSING`: Set to `true` for async processing, `false` for sequential
- `MAX_RETRIES`: Number of retry attempts for failed API calls (default: 3)

## Usage

### Starting the Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Health Check
```bash
GET /
```

Response:
```json
{
  "status": "healthy",
  "service": "AEO Analytics API"
}
```

#### 2. Start Analysis
```bash
POST /analyze
```

Request body:
```json
{
  "target_website": "https://yourbrand.com",
  "competitor_websites": [
    "https://competitor1.com",
    "https://competitor2.com",
    "https://competitor3.com",
    "https://competitor4.com"
  ]
}
```

Response:
```json
{
  "status": "started",
  "job_id": "job_20250602123456",
  "message": "Analysis started for 5 websites"
}
```

#### 3. Get Results
```bash
GET /results/{job_id}
```

Response includes:
- Model used
- Query/prompt
- Target brand mention (true/false)
- Target snippet (if mentioned)
- Competitor mentions
- Competitor snippets
- Full AI response
- Timestamp

### Example Usage

See `example_usage.py` for a complete example of how to:
- Start an analysis
- Fetch results
- Process and analyze the data
- Generate reports

```python
# Quick example
import aiohttp
import asyncio

async def analyze_brand():
    async with aiohttp.ClientSession() as session:
        # Start analysis
        async with session.post(
            "http://localhost:8000/analyze",
            json={
                "target_website": "https://yourbrand.com",
                "competitor_websites": ["https://competitor.com"]
            }
        ) as response:
            result = await response.json()
            print(f"Job ID: {result['job_id']}")

asyncio.run(analyze_brand())
```

## Data Storage

The current implementation includes placeholders for data storage. To integrate with Replit Database:

1. Install the Replit package:
```bash
pip install replit
```

2. Update the code to save results:
```python
from replit import db

# In run_analysis function
db[f"job_{job_id}"] = {
    "status": "completed",
    "results": [result.dict() for result in results],
    "timestamp": datetime.now().isoformat()
}
```

## Rate Limiting and Error Handling

- **Automatic Retries**: Failed API calls are retried up to 3 times with exponential backoff
- **Rate Limit Detection**: Returns clear error messages when rate limits are hit
- **Graceful Degradation**: If one AI model fails, others continue processing

## Performance Optimization

- **Concurrent Processing**: Enable async processing for faster results
- **Query Limiting**: Defaults to 2 queries and 5 models for demo (adjust in code)
- **Caching**: Consider implementing Redis/memory caching for repeated queries

## Monitoring and Logging

The application includes comprehensive logging:
- API call status
- Processing steps
- Error details
- Performance metrics

View logs in console or redirect to a file:
```bash
python main.py > app.log 2>&1
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env`
2. **Rate Limits**: Reduce concurrent requests or add delays
3. **Memory Issues**: Process fewer queries at once or enable sequential processing
4. **Network Timeouts**: Increase timeout values in `aiohttp` sessions

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- WebSocket support for real-time updates
- Batch processing for multiple brands
- Historical tracking and trends
- Export functionality (CSV, JSON, PDF reports)
- Webhook notifications
- Advanced filtering and search
- Dashboard UI integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license here]

## Support

For issues or questions:
- Open a GitHub issue
- Check existing documentation
- Review example code
