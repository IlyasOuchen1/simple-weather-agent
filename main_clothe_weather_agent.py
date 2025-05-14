from simple_langchain_agent import SimpleLangChainWeatherAgent
from clothe_agent import ClotheAgent
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Welcome to the Weather & Clothing Recommendation Agent!")
    print("Ask about the weather in any location, and I'll suggest what to wear.")
    print("Available reasoning types for weather analysis: react, cot, tot")
    print("Type 'exit' to quit.\n")
    
    try:
        # Create the agents
        print("Initializing Weather and Clothing Agents...")
        weather_agent = SimpleLangChainWeatherAgent()
        clothe_agent = ClotheAgent()
        print("Agents initialized successfully!\n")
        
        while True:
            # Get reasoning type
            reasoning_type = input("Select reasoning type (react, cot, tot) or press Enter for default: ").lower()
            if reasoning_type not in ["react", "cot", "tot", ""]:
                print("Invalid reasoning type. Using default (react).")
                reasoning_type = "react"
            elif reasoning_type == "":
                reasoning_type = "react"
                print("Using default reasoning type: react")
            
            # Get the query
            query = input("Your query (about weather and what to wear): ")
            if query.lower() == "exit":
                print("Goodbye!")
                break
            
            if not query.strip():
                print("Empty query. Please try again.")
                continue
            
            # Add clothing-specific prompt if not explicitly mentioned
            if not any(keyword in query.lower() for keyword in ["wear", "clothing", "dress", "outfit", "attire"]):
                weather_query = query
                asking_for_clothing = False
            else:
                weather_query = query
                asking_for_clothing = True
            
            print(f"Using reasoning type: {reasoning_type}")
            
            try:
                # Get weather information
                weather_response = weather_agent.process_query(weather_query, reasoning_type)
                
                print("\nWeather information:")
                print(weather_response)
                
                # Always provide clothing recommendations
                weather_data = None
                location = None
                
                # Extract location and weather data from the weather agent
                if hasattr(weather_agent, "last_weather_data"):
                    weather_data = weather_agent.last_weather_data
                    location = weather_agent.last_location
                else:
                    # Need to parse from the weather agent's response
                    lines = weather_response.split('\n')
                    for line in lines:
                        if "Temperature:" in line or "temperature:" in line:
                            temp_text = line.split(":", 1)[1].strip().split("°C")[0].strip()
                            if temp_text:
                                try:
                                    temp = float(temp_text)
                                    if not weather_data:
                                        weather_data = {"main": {"temp": temp}}
                                    else:
                                        weather_data["main"]["temp"] = temp
                                except:
                                    pass
                        
                        # Try to find location
                        location_indicators = ["weather in ", "weather for ", "weather at ", "in ", "for "]
                        for indicator in location_indicators:
                            if indicator in query.lower():
                                location = query.lower().split(indicator, 1)[1].strip().split(" ")[0].strip()
                                break
                
                if not weather_data:
                    weather_data = {"main": {"temp": 20, "feels_like": 20, "humidity": 50}, 
                                   "weather": [{"description": "unknown"}]}
                
                if not location:
                    location = "the provided location"
                
                # Generate clothing recommendation if requested or implied
                if asking_for_clothing or input("\nWould you like clothing recommendations? (y/n): ").lower().startswith("y"):
                    clothing_recommendation = clothe_agent.generate_clothing_recommendation(
                        weather_data, location
                    )
                    
                    print("\nClothing recommendation:")
                    print(clothing_recommendation)
                
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Please try a different query or reasoning type.")
                traceback.print_exc()
    except Exception as e:
        print(f"Error initializing agents: {e}")
        traceback.print_exc()
        print("\nPlease check your API keys and internet connection.")
        print("Make sure you have installed all required packages.")

if __name__ == "__main__":
    main()