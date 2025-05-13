import requests
import wikipedia
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeatherService:
    def __init__(self):
        # Use the API key directly
        self.api_key = "58a1809bd5d800285b6ff1070121ae2b"
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, location):
        """Get weather information for a given location"""
        try:
            # Clean the location string to ensure no unexpected characters
            location = location.strip()
            
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Print more detailed error information for debugging
            print(f"Error URL: {response.url if 'response' in locals() else 'No response'}")
            print(f"Error details: {e}")
            return {"error": f"Weather API error: {str(e)}"}

class WikipediaService:
    def get_location_info(self, location):
        """Get brief description of a location from Wikipedia"""
        try:
            # Search for the location
            search_results = wikipedia.search(location)
            if not search_results:
                return {"error": f"No Wikipedia information found for {location}"}
            
            # Get summary of the first search result
            page = wikipedia.page(search_results[0])
            summary = wikipedia.summary(search_results[0], sentences=2)
            return {
                "summary": summary,
                "url": page.url
            }
        except wikipedia.exceptions.WikipediaException as e:
            return {"error": f"Wikipedia error: {str(e)}"}