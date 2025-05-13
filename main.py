from openai_agent import OpenAIWeatherAgent
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Welcome to the Advanced Weather Agent!")
    print("You can ask about the weather in any location.")
    print("You can also specify reasoning types: react, cot, or tot")
    print("Examples:")
    print("  - 'What's the weather in Paris?' (uses default reasoning)")
    print("  - 'Use cot: Weather in New York'")
    print("  - 'tot: How's the weather in Springfield?'")
    print("Type 'exit' to quit.\n")
    
    try:
        # Create the agent
        agent = OpenAIWeatherAgent()
        
        while True:
            # Separate prompts for reasoning type and query
            reasoning_type = input("Select reasoning type (react, cot, tot) or press Enter for default: ").lower()
            if reasoning_type not in ["react", "cot", "tot", ""]:
                print("Invalid reasoning type. Using default (react).")
                reasoning_type = "react"
            elif reasoning_type == "":
                reasoning_type = "react"
                
            query = input("Your query: ")
            if query.lower() == "exit":
                print("Goodbye!")
                break
            elif query.lower() == "debug":
                # Special command to show the thought process
                debug_info = agent.get_thought_process(input("Enter a query to debug: "))
                print(debug_info)
                continue
            
            print(f"Using reasoning type: {reasoning_type}")
            response = agent.process_query(query, reasoning_type)
            
            print("\nAgent response:")
            print(response)
            print()
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your API keys and internet connection.")

if __name__ == "__main__":
    main()