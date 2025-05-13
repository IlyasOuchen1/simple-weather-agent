# Simple Weather Agent

A sophisticated weather information assistant that uses LangChain and OpenAI's GPT models to provide detailed weather information and location context. The agent employs multiple reasoning strategies (ReAct, Chain of Thought, and Tree of Thoughts) to handle weather queries intelligently.

## Features

- **Multiple Reasoning Strategies**:
  - ReAct (Reasoning and Acting)
  - Chain of Thought (CoT)
  - Tree of Thoughts (ToT)
- **Comprehensive Weather Information**:
  - Current temperature
  - Feels like temperature
  - Weather conditions
  - Humidity
- **Location Context**:
  - Wikipedia integration for location information
  - Background details about places
- **Intelligent Query Processing**:
  - Location extraction
  - Ambiguity resolution
  - Confidence scoring
  - Multiple location consideration

## Prerequisites

- Python 3.8+
- OpenAI API key
- OpenWeatherMap API key
- Wikipedia API access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IlyasOuchen1/simple-weather-agent.git
cd simple-weather-agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
```

## Usage

```python
from simple_langchain_agent import SimpleLangChainWeatherAgent

# Initialize the agent
agent = SimpleLangChainWeatherAgent()

# Process a query using different reasoning strategies
# Available strategies: "react", "cot", "tot"
response = agent.process_query("What's the weather like in Paris?", reasoning_type="tot")
print(response)
```

## Reasoning Strategies

### ReAct (Reasoning and Acting)
- Combines reasoning and action in a step-by-step process
- Plans information collection
- Evaluates data quality
- Provides structured responses

### Chain of Thought (CoT)
- Step-by-step reasoning process
- Detailed explanation of location identification
- Clear reasoning trail
- Structured response generation

### Tree of Thoughts (ToT)
- Explores multiple possible interpretations
- Assigns confidence scores to locations
- Handles ambiguous queries
- Provides alternative locations when relevant

## Project Structure

```
simple-weather-agent/
├── simple_langchain_agent.py    # Main agent implementation
├── api_services.py             # Weather and Wikipedia API services
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## API Services

### Weather Service
- Uses OpenWeatherMap API
- Provides current weather conditions
- Includes temperature, humidity, and weather descriptions

### Wikipedia Service
- Fetches location information
- Provides background context
- Includes source attribution

## Error Handling

The agent includes robust error handling:
- API failure fallbacks
- Location extraction fallbacks
- Graceful degradation of services
- Informative error messages

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- OpenWeatherMap for weather data
- Wikipedia for location information
- LangChain for the framework 
