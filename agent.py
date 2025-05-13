from api_services import WeatherService, WikipediaService

class WeatherAgent:
    def __init__(self):
        self.weather_service = WeatherService()
        self.wiki_service = WikipediaService()
    
    def process_query(self, query):
        """
        Process user query using ReAct pattern:
        1. Reason - Parse query and decide what information to get
        2. Act - Call appropriate services to get information
        3. Respond - Format and return the response
        """
        # Step 1: Reason - Parse query to extract location
        location = self._extract_location(query)
        if not location:
            return "I couldn't identify a location in your query. Please specify a city or place."
        
        # Step 2: Act - Get weather and location information
        weather_data = self.weather_service.get_weather(location)
        wiki_data = self.wiki_service.get_location_info(location)
        
        # Check if there were errors in the API calls
        if "error" in weather_data:
            return f"Sorry, I couldn't get weather information: {weather_data['error']}"
        
        # Step 3: Respond - Format the response
        return self._format_response(location, weather_data, wiki_data)
    
    def _extract_location(self, query):
        """
        Extract location from user query.
        This is a simplified version. For a more robust solution,
        consider using NLP libraries like spaCy for entity recognition.
        """
        # Clean the query and convert to lowercase
        query = query.lower().strip()
        
        # Remove any punctuation at the end (like question marks)
        query = query.rstrip('?!.,;:')
        
        # Look for patterns like "weather in [location]" or "what's the weather in [location]"
        if "weather in " in query:
            location = query.split("weather in ")[1].strip()
        elif "weather for " in query:
            location = query.split("weather for ")[1].strip()
        elif "weather at " in query:
            location = query.split("weather at ")[1].strip()
        else:
            # If no pattern matched, use a default fallback
            # This is very simple and could be improved
            location = query
        
        # Clean the location of any punctuation and extra spaces
        location = location.rstrip('?!.,;:')
        return location
    
    def _format_response(self, location, weather_data, wiki_data):
        """Format the response with weather and Wikipedia information"""
        # Extract relevant weather information
        temp = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        weather_desc = weather_data["weather"][0]["description"]
        
        # Create the response
        response = f"Weather in {location.title()}:\n"
        response += f"• Current condition: {weather_desc}\n"
        response += f"• Temperature: {temp}°C (feels like {feels_like}°C)\n"
        response += f"• Humidity: {humidity}%\n\n"
        
        # Add Wikipedia information if available
        if "error" not in wiki_data:
            response += f"About {location.title()}:\n"
            response += f"{wiki_data['summary']}\n"
            response += f"Learn more: {wiki_data['url']}"
        
        return response