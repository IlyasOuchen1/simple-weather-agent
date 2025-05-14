from simple_langchain_agent import SimpleLangChainWeatherAgent
from clothe_agent import ClotheAgent
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Welcome to the LangChain Weather Agent!")
    print("This agent uses LangChain for different reasoning patterns.")
    print("You can ask about the weather in any location.")
    print("Available reasoning types: react, cot, tot")
    print("After getting weather information, you can ask for clothing recommendations.")
    print("Type 'exit' to quit.\n")
    
    try:
        # Initialize both agents
        print("Initializing Weather and Clothing Agents...")
        weather_agent = SimpleLangChainWeatherAgent()
        clothe_agent = ClotheAgent()
        print("Agents initialized successfully!\n")
        
        while True:
            # Separate prompts for reasoning type and query
            reasoning_type = input("Select reasoning type (react, cot, tot) or press Enter for default: ").lower()
            if reasoning_type not in ["react", "cot", "tot", ""]:
                print("Invalid reasoning type. Using default (react).")
                reasoning_type = "react"
            elif reasoning_type == "":
                reasoning_type = "react"
                print("Using default reasoning type: react")
            
            query = input("Your query: ")
            if query.lower() == "exit":
                print("Goodbye!")
                break
            
            if not query.strip():
                print("Empty query. Please try again.")
                continue
            
            print(f"Using reasoning type: {reasoning_type}")
            try:
                # Get weather information
                weather_response = weather_agent.process_query(query, reasoning_type)
                
                print("\nAgent response:")
                print(weather_response)
                print()
                
                # Check if user wants clothing recommendations
                if input("Would you like clothing recommendations based on this weather? (y/n): ").lower().startswith('y'):
                    # Extract weather data and location
                    # We need to access the last processed weather data
                    if hasattr(weather_agent, 'last_weather_data') and hasattr(weather_agent, 'last_location'):
                        weather_data = weather_agent.last_weather_data
                        location = weather_agent.last_location
                        
                        # Generate clothing recommendation
                        clothing_recommendation = clothe_agent.generate_clothing_recommendation(
                            weather_data, location
                        )
                        
                        print("\nClothing recommendation:")
                        print(clothing_recommendation)
                        print()
                    else:
                        print("\nSorry, I don't have enough weather information to make clothing recommendations.")
                        print("Please try another weather query first.\n")
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Please try a different query or reasoning type.")
                traceback.print_exc()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        traceback.print_exc()
        print("\nPlease check your API keys and internet connection.")
        print("Make sure you have installed all required packages:")
        print("pip install langchain langchain-openai python-dotenv")

if __name__ == "__main__":
    main()