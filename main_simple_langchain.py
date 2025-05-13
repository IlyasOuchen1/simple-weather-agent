from simple_langchain_agent import SimpleLangChainWeatherAgent
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Welcome to the Simple LangChain Weather Agent!")
    print("This agent uses LangChain for different reasoning patterns.")
    print("You can ask about the weather in any location.")
    print("Available reasoning types: react, cot, tot")
    print("Type 'exit' to quit.\n")
    
    try:
        # Create the agent
        print("Initializing Simple LangChain Weather Agent...")
        agent = SimpleLangChainWeatherAgent()
        print("Agent initialized successfully!\n")
        
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
                response = agent.process_query(query, reasoning_type)
                
                print("\nAgent response:")
                
                # Check for possibly truncated response
                if response and len(response) > 0 and response[0].islower():
                    print("Note: Response appears to be truncated at the beginning.")
                
                print(response)
                print()
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