import os
import json
import re
from dotenv import load_dotenv

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Import our weather and Wikipedia services
from api_services import WeatherService, WikipediaService

# Load environment variables
load_dotenv()

class SimpleLangChainWeatherAgent:
    def __init__(self):
        # Load API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=openai_api_key
        )
        
        # Initialize services
        self.weather_service = WeatherService()
        self.wiki_service = WikipediaService()
        
        # Available reasoning types
        self.reasoning_types = ["react", "cot", "tot"]
        
        # Debug flag
        self.debug = True
    
    def process_query(self, query, reasoning_type="react"):
        """
        Process a query using the specified reasoning type
        
        Args:
            query (str): The user's query
            reasoning_type (str): The reasoning type to use ("react", "cot", or "tot")
            
        Returns:
            str: The response to the query
        """
        # Validate reasoning type
        reasoning_type = reasoning_type.lower()
        if reasoning_type not in self.reasoning_types:
            return f"Invalid reasoning type. Please choose from: {', '.join(self.reasoning_types)}"
        
        # Process based on reasoning type
        if reasoning_type == "react":
            return self._process_query_react(query)
        elif reasoning_type == "cot":
            return self._process_query_cot(query)
        elif reasoning_type == "tot":
            return self._process_query_tot(query)

    def _process_query_react(self, query):
        """Process query using the ReAct pattern with LangChain"""
        print("\n" + "="*80)
        print("REACT REASONING PROCESS (LangChain)".center(80))
        print("="*80)
        print("1. Query received:", query)
        
        # Step 1 & 2: Reason and Plan
        print("\n" + "-"*80)
        print("2. REASONING AND PLANNING".center(80))
        print("-"*80)
        
        # Create reasoning prompt
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that analyzes weather queries.
             First, identify the location in the query.
             Then, plan what information should be collected.
             
             IMPORTANT: For most weather queries, users appreciate both weather data AND
             background information about the location, so include location info unless
             explicitly not needed.
             
             Respond in this exact format:
             
             LOCATION: [extracted location]
             NEEDS_WEATHER: [yes/no]
             NEEDS_LOCATION_INFO: [yes/no]
             TIME_PERIOD: [current/today/tomorrow/week]
             WEATHER_ASPECTS: [comma-separated list, e.g., temperature, humidity, conditions]
             """),
            ("human", "{query}")
        ])
        
        try:
            # Run the planning chain
            planning_response = self.llm.invoke(planning_prompt.format(query=query))
            planning_text = planning_response.content
            
            if self.debug:
                print("\nPlanning Response:")
                print(planning_text)
                print()
            
            # Parse text response
            location = ""
            needs_weather = True
            needs_location_info = True
            time_period = "current"
            weather_aspects = []
            
            lines = planning_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("LOCATION:"):
                    location = line.replace("LOCATION:", "").strip()
                elif line.startswith("NEEDS_WEATHER:"):
                    needs_weather = line.replace("NEEDS_WEATHER:", "").strip().lower() == "yes"
                elif line.startswith("NEEDS_LOCATION_INFO:"):
                    needs_location_info = line.replace("NEEDS_LOCATION_INFO:", "").strip().lower() == "yes"
                elif line.startswith("TIME_PERIOD:"):
                    time_period = line.replace("TIME_PERIOD:", "").strip().lower()
                elif line.startswith("WEATHER_ASPECTS:"):
                    aspects = line.replace("WEATHER_ASPECTS:", "").strip()
                    weather_aspects = [aspect.strip() for aspect in aspects.split(",")]
            
            print(f"• Extracted location: '{location}'")
            print(f"• Need weather data: {needs_weather}")
            print(f"• Need location info: {needs_location_info}")
            print(f"• Time period: {time_period}")
            
            if weather_aspects:
                print(f"• Weather aspects of interest: {', '.join(weather_aspects)}")
            
            if not location:
                return "I couldn't identify a location in your query. Please specify a city or place."
        except Exception as e:
            print(f"Error in planning: {e}")
            # Fallback to a simpler approach
            print("Falling back to simple location extraction...")
            location = query.lower()
            for prefix in ["weather in ", "weather at ", "weather for ", "how's the weather in ", "how is the weather in "]:
                if prefix in location:
                    location = location.split(prefix)[1].strip()
                    break
            location = location.strip("?!.,;:")
            needs_weather = True
            needs_location_info = True
            print(f"• Extracted location: '{location}'")
        
        # Step 3: Act
        print("\n" + "-"*80)
        print(f"3. TAKING ACTIONS FOR '{location}'".center(80))
        print("-"*80)
        
        # Get weather data if needed
        weather_data = None
        if needs_weather:
            print(f"• Fetching weather data for '{location}'...")
            weather_data = self.weather_service.get_weather(location)
            
            if "error" in weather_data:
                print(f"  ✗ Error: {weather_data['error']}")
                return f"Sorry, I couldn't get weather information: {weather_data['error']}"
            else:
                print(f"  ✓ Successfully retrieved weather data:")
                print(f"    - Temperature: {weather_data['main']['temp']}°C")
                print(f"    - Feels like: {weather_data['main']['feels_like']}°C")
                print(f"    - Conditions: {weather_data['weather'][0]['description']}")
                print(f"    - Humidity: {weather_data['main']['humidity']}%")
        
        # Get location info if needed
        wiki_data = None
        if needs_location_info:
            print(f"• Fetching Wikipedia data for '{location}'...")
            wiki_data = self.wiki_service.get_location_info(location)
            
            if "error" in wiki_data:
                print(f"  ✗ Error: {wiki_data['error']}")
                print("  → Will proceed with weather data only")
            else:
                print(f"  ✓ Successfully retrieved location information")
                if "summary" in wiki_data:
                    summary_preview = wiki_data["summary"][:100] + "..." if len(wiki_data["summary"]) > 100 else wiki_data["summary"]
                    print(f"    - Summary: {summary_preview}")
                if "url" in wiki_data:
                    print(f"    - Source: {wiki_data['url']}")
        
        # Step 4: Reflect
        print("\n" + "-"*80)
        print("4. REFLECTING ON COLLECTED INFORMATION".center(80))
        print("-"*80)
        
        # Prepare data summary for reflection
        weather_available = weather_data is not None and "error" not in weather_data
        wiki_available = wiki_data is not None and "error" not in wiki_data
        
        data_summary = {
            "query": query,
            "location": location,
            "weather_available": weather_available,
            "location_info_available": wiki_available
        }
        
        # Add weather details if available
        if weather_available:
            try:
                data_summary["weather_details"] = {
                    "has_temperature": "main" in weather_data and "temp" in weather_data["main"],
                    "has_conditions": "weather" in weather_data and len(weather_data["weather"]) > 0,
                    "has_humidity": "main" in weather_data and "humidity" in weather_data["main"]
                }
            except (KeyError, TypeError):
                data_summary["weather_details"] = {"error": "Malformed weather data"}
        
        # Create reflection prompt using text format instead of JSON
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that evaluates information quality.
             Analyze the available data and determine:
             1. Is the information sufficient to answer the query?
             2. Is any critical information missing?
             3. Is there anything unusual about the data that should be noted?
             4. Should any additional actions be taken?
             
             Respond in this exact format:
             
             SUFFICIENT: [yes/no]
             MISSING_INFORMATION: [comma-separated list, or "none"]
             NOTES: [your observations about the data]
             SUGGESTED_ACTION: [action to take if needed, or "none"]
             ALTERNATIVE_LOCATION: [alternative location if ambiguous, or "none"]
             """),
            ("human", "{data_summary}")
        ])
        
        try:
            # Run the reflection chain
            reflection_response = self.llm.invoke(reflection_prompt.format(data_summary=json.dumps(data_summary)))
            reflection_text = reflection_response.content
            
            if self.debug:
                print("\nReflection Response:")
                print(reflection_text)
                print()
            
            # Parse the text response
            is_sufficient = True
            notes = ""
            missing_info = []
            suggested_action = ""
            alternative_location = ""
            
            lines = reflection_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("SUFFICIENT:"):
                    is_sufficient = line.replace("SUFFICIENT:", "").strip().lower() == "yes"
                elif line.startswith("MISSING_INFORMATION:"):
                    missing_str = line.replace("MISSING_INFORMATION:", "").strip()
                    if missing_str.lower() != "none":
                        missing_info = [item.strip() for item in missing_str.split(",")]
                elif line.startswith("NOTES:"):
                    notes = line.replace("NOTES:", "").strip()
                elif line.startswith("SUGGESTED_ACTION:"):
                    suggested_action = line.replace("SUGGESTED_ACTION:", "").strip()
                    if suggested_action.lower() == "none":
                        suggested_action = ""
                elif line.startswith("ALTERNATIVE_LOCATION:"):
                    alternative_location = line.replace("ALTERNATIVE_LOCATION:", "").strip()
                    if alternative_location.lower() == "none":
                        alternative_location = ""
            
            print(f"• Is the data sufficient? {'Yes' if is_sufficient else 'No'}")
            if notes:
                print(f"• Notes: {notes}")
            if missing_info:
                print("• Missing information:")
                for item in missing_info:
                    print(f"  - {item}")
            
            # Handle suggested actions from reflection
            if not is_sufficient and suggested_action:
                print(f"• Taking suggested action: {suggested_action}")
                
                # Try alternative location if suggested
                if suggested_action == "try_alternative_location" and alternative_location:
                    alt_location = alternative_location
                    print(f"  → Trying alternative location: '{alt_location}'")
                    
                    if needs_weather:
                        weather_data = self.weather_service.get_weather(alt_location)
                        if "error" not in weather_data:
                            print(f"  ✓ Successfully retrieved weather data for '{alt_location}'")
                        else:
                            print(f"  ✗ Error getting weather for '{alt_location}': {weather_data['error']}")
                    
                    if needs_location_info:
                        wiki_data = self.wiki_service.get_location_info(alt_location)
                        if "error" not in wiki_data:
                            print(f"  ✓ Successfully retrieved location information for '{alt_location}'")
                        else:
                            print(f"  ✗ Error getting location info for '{alt_location}': {wiki_data['error']}")
                    
                    # Update location to the alternative
                    location = alt_location
        except Exception as e:
            print(f"Error in reflection: {e}")
            # Use default values for reflection
            is_sufficient = True
            notes = "Reflection process encountered an error"
            print(f"• Is the data sufficient? Yes (default)")
            print(f"• Notes: {notes}")
        
        # Step 5: Generate response
        print("\n" + "-"*80)
        print("5. GENERATING FINAL RESPONSE".center(80))
        print("-"*80)
        
        # Extract data for response
        if not weather_data or "error" in weather_data:
            return f"I couldn't find weather information for {location}."
        
        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        conditions = weather_data["weather"][0]["description"]
        
        # Extract source information
        weather_source_url = "https://openweathermap.org/"
        wiki_source_url = wiki_data.get("url", "") if wiki_data and "error" not in wiki_data else ""
        wiki_summary = wiki_data.get("summary", "No information available") if wiki_data and "error" not in wiki_data else "No information available"
        
        # Create response prompt
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful weather assistant using the ReAct pattern.
             Create a friendly and informative response about the weather and location based on the provided data.
             
             Include:
             1. Current weather conditions
             2. Temperature and how it feels
             3. Humidity
             4. A brief description of the location (if available)
             
             Consider the reflection information in your response. If there were issues with the data,
             you may want to acknowledge them subtly.
             
             IMPORTANT: Always include source attribution at the end of your response.
             Format it as: "Sources: [Weather data from OpenWeatherMap](url), [Location info from Wikipedia](url)"
             """),
            ("human", """
             Query: {query}
             
             Weather data for {location}:
             - Temperature: {temperature}°C
             - Feels like: {feels_like}°C
             - Humidity: {humidity}%
             - Conditions: {conditions}
             
             Location information:
             {wiki_summary}
             
             Sources:
             - Weather: {weather_source_url}
             - Location: {wiki_source_url}
             
             Reflection notes: {notes}
             """)
        ])
        
        try:
            # Run the response chain
            response = self.llm.invoke(response_prompt.format(
                query=query,
                location=location,
                temperature=temperature,
                feels_like=feels_like,
                humidity=humidity,
                conditions=conditions,
                wiki_summary=wiki_summary,
                weather_source_url=weather_source_url,
                wiki_source_url=wiki_source_url,
                notes=notes
            ))
            
            response_text = response.content
            
            if self.debug:
                print("\nRaw Response Text:")
                print(repr(response_text))
                print()
            
            print("• Response generation complete")
            print("="*80 + "\n")
            
            return response_text
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            return f"In {location}, the current temperature is {temperature}°C, feels like {feels_like}°C, with {conditions} and {humidity}% humidity.\n\nSources: [Weather data from OpenWeatherMap]({weather_source_url})"

    
    def _process_query_cot(self, query):
        """Process query using Chain of Thought (CoT) reasoning with LangChain"""
        print("\n" + "="*80)
        print("CHAIN OF THOUGHT (CoT) REASONING PROCESS (LangChain)".center(80))
        print("="*80)
        print("1. Query received:", query)
        
        # Step 1: Chain of Thought reasoning to extract location
        print("\n" + "-"*80)
        print("2. CHAIN OF THOUGHT REASONING".center(80))
        print("-"*80)
        
        # Simplified approach - just use a direct query for location with reasoning
        location_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a weather query. 
            Extract the location mentioned and explain your thinking step by step.
            Think about what clues in the query indicate a location.
            Respond in this exact format:
            
            Step 1: [your reasoning]
            Step 2: [your reasoning]
            Step 3: [your reasoning]
            Step 4: [your conclusion]
            
            LOCATION: [just the location name]
            """),
            ("human", "{query}")
        ])
        
        try:
            # Run a simpler reasoning approach
            cot_response = self.llm.invoke(location_prompt.format(query=query))
            cot_text = cot_response.content
            
            if self.debug:
                print("\nRaw CoT Reasoning Response:")
                print(repr(cot_text))
                print()
            
            # Extract location from the response
            location = ""
            reasoning_steps = []
            
            # Parse the text output
            lines = cot_text.strip().split('\n')
            for line in lines:
                if line.startswith("Step "):
                    reasoning_steps.append(line)
                elif line.startswith("LOCATION:"):
                    location = line.replace("LOCATION:", "").strip()
            
            # If we can't extract a location, try direct extraction
            if not location:
                print("No location found in CoT output, extracting directly from query...")
                location = query.lower()
                for prefix in ["weather in ", "weather at ", "weather for ", "how's the weather in ", "how is the weather in "]:
                    if prefix in location:
                        location = location.split(prefix)[1].strip()
                        break
                location = location.strip("?!.,;:")
            
            print(f"• Final extracted location: '{location}'")
            print("• Reasoning steps:")
            for i, step in enumerate(reasoning_steps):
                print(f"  {step}")
            
            if not location:
                return "After careful consideration, I couldn't identify a location in your query. Please specify a city or place."
        except Exception as e:
            print(f"Error in CoT reasoning: {e}")
            # Fallback to a simpler approach
            print("Falling back to simple location extraction...")
            location = query.lower()
            for prefix in ["weather in ", "weather at ", "weather for ", "how's the weather in ", "how is the weather in "]:
                if prefix in location:
                    location = location.split(prefix)[1].strip()
                    break
            location = location.strip("?!.,;:")
            reasoning_steps = ["Simple extraction of location from query"]
            print(f"• Extracted location: '{location}'")
        
        # Step 2: Get data
        print("\n" + "-"*80)
        print(f"3. GATHERING DATA FOR '{location}'".center(80))
        print("-"*80)
        
        print(f"• Fetching weather data for '{location}'...")
        weather_data = self.weather_service.get_weather(location)
        
        if "error" in weather_data:
            print(f"  ✗ Error: {weather_data['error']}")
            return f"Sorry, I couldn't get weather information: {weather_data['error']}"
        else:
            print(f"  ✓ Successfully retrieved weather data:")
            print(f"    - Temperature: {weather_data['main']['temp']}°C")
            print(f"    - Feels like: {weather_data['main']['feels_like']}°C")
            print(f"    - Conditions: {weather_data['weather'][0]['description']}")
            print(f"    - Humidity: {weather_data['main']['humidity']}%")
        
        print(f"• Fetching Wikipedia data for '{location}'...")
        wiki_data = self.wiki_service.get_location_info(location)
        
        if "error" in wiki_data:
            print(f"  ✗ Error: {wiki_data['error']}")
            print("  → Will proceed with weather data only")
            wiki_summary = "No information available"
            wiki_source_url = ""
        else:
            print(f"  ✓ Successfully retrieved location information")
            wiki_summary = wiki_data.get("summary", "No information available")
            wiki_source_url = wiki_data.get("url", "")
            if "summary" in wiki_data:
                summary_preview = wiki_data["summary"][:100] + "..." if len(wiki_data["summary"]) > 100 else wiki_data["summary"]
                print(f"    - Summary: {summary_preview}")
            if "url" in wiki_data:
                print(f"    - Source: {wiki_data['url']}")
        
        # Step 3: Generate response
        print("\n" + "-"*80)
        print("4. GENERATING RESPONSE WITH COT REASONING".center(80))
        print("-"*80)
        
        # Extract data from weather response
        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        conditions = weather_data["weather"][0]["description"]
        
        # Weather source information
        weather_source_url = "https://openweathermap.org/"
        
        # Prepare reasoning steps as context
        steps_text = "\n".join([f"- {step}" for step in reasoning_steps])
        
        # Create response prompt
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful weather assistant using Chain of Thought (CoT) reasoning.
            Create a friendly and informative response about the weather and location based on the provided data.
            
            Start your response with a clear, complete sentence about the current weather conditions.
            
            Include:
            1. Current weather conditions
            2. Temperature and how it feels
            3. Humidity
            4. A brief description of the location (if available)
            
            Include a brief mention of the reasoning process that led to identifying the location,
            but focus primarily on providing weather information.
            
            IMPORTANT: Always include source attribution at the end of your response.
            Format it as: "Sources: [Weather data from OpenWeatherMap](url), [Location info from Wikipedia](url)"
            """),
            ("human", """
            Query: {query}
            
            Weather data for {location}:
            - Temperature: {temperature}°C
            - Feels like: {feels_like}°C
            - Humidity: {humidity}%
            - Conditions: {conditions}
            
            Location information:
            {wiki_summary}
            
            Sources:
            - Weather: {weather_source_url}
            - Location: {wiki_source_url}
            
            Chain of Thought reasoning steps:
            {steps_text}
            """)
        ])
        
        try:
            # Run the response chain
            response = self.llm.invoke(response_prompt.format(
                query=query,
                location=location,
                temperature=temperature,
                feels_like=feels_like,
                humidity=humidity,
                conditions=conditions,
                wiki_summary=wiki_summary,
                weather_source_url=weather_source_url,
                wiki_source_url=wiki_source_url,
                steps_text=steps_text
            ))
            
            response_text = response.content
            
            if self.debug:
                print("\nRaw Response Text:")
                print(repr(response_text))
                print()
            
            print("• Response generation complete")
            print("="*80 + "\n")
            
            return response_text
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            return f"Through chain-of-thought reasoning, I identified your location as {location}. Currently, the temperature is {temperature}°C, feels like {feels_like}°C, with {conditions} and {humidity}% humidity.\n\nSources: [Weather data from OpenWeatherMap]({weather_source_url})"


    def _process_query_tot(self, query):
        """Process query using Tree of Thoughts (ToT) reasoning with LangChain"""
        print("\n" + "="*80)
        print("TREE OF THOUGHTS (ToT) REASONING PROCESS (LangChain)".center(80))
        print("="*80)
        print("1. Query received:", query)
        
        # Step 1: Generate multiple possible interpretations
        print("\n" + "-"*80)
        print("2. GENERATING AND EVALUATING MULTIPLE POSSIBLE LOCATIONS".center(80))
        print("-"*80)
        
        # Create ToT reasoning prompt
        tot_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a weather query using Tree of Thoughts (ToT) reasoning.
             For potentially ambiguous queries, explore multiple possible interpretations:
             1. Generate 2-4 possible locations that could be referenced in the query.
             2. For each possible location, assign a confidence score (0-100) and provide reasoning.
             3. Select the most likely location based on your evaluation.
             
             Format your response exactly like this:
             
             POSSIBLE LOCATION: [location1]
             SCORE: [score1]
             REASON: [reason1]
             
             POSSIBLE LOCATION: [location2]
             SCORE: [score2]
             REASON: [reason2]
             
             ... (repeat for each location)
             
             SELECTED LOCATION: [final location]
             SELECTION REASONING: [explanation for your choice]
             """),
            ("human", "{query}")
        ])
        
        try:
            # Run the ToT reasoning chain
            tot_response = self.llm.invoke(tot_prompt.format(query=query))
            tot_text = tot_response.content
            
            if self.debug:
                print("\nRaw ToT Reasoning Response:")
                print(repr(tot_text))
                print()
            
            # Parse the text output
            possible_locations = []
            evaluations = {}
            selected_location = ""
            reasoning = ""
            
            current_location = None
            current_score = None
            current_reason = None
            
            lines = tot_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("POSSIBLE LOCATION:"):
                    if current_location and current_score is not None:
                        # Save the previous location data
                        evaluations[current_location] = {
                            "score": current_score,
                            "reason": current_reason or "No reason provided"
                        }
                    
                    # Start a new location
                    current_location = line.replace("POSSIBLE LOCATION:", "").strip()
                    current_score = None
                    current_reason = None
                    possible_locations.append(current_location)
                elif line.startswith("SCORE:"):
                    current_score = line.replace("SCORE:", "").strip()
                    try:
                        current_score = int(current_score)
                    except ValueError:
                        current_score = 0
                elif line.startswith("REASON:"):
                    current_reason = line.replace("REASON:", "").strip()
                elif line.startswith("SELECTED LOCATION:"):
                    selected_location = line.replace("SELECTED LOCATION:", "").strip()
                elif line.startswith("SELECTION REASONING:"):
                    reasoning = line.replace("SELECTION REASONING:", "").strip()
            
            # Save the last location if needed
            if current_location and current_score is not None and current_location not in evaluations:
                evaluations[current_location] = {
                    "score": current_score,
                    "reason": current_reason or "No reason provided"
                }
            
            # If only one location was found, use it as the selected location
            if len(possible_locations) == 1 and not selected_location:
                selected_location = possible_locations[0]
                reasoning = "Only one location was identified."
            
            print(f"• Possible locations identified: {len(possible_locations)}")
            
            if possible_locations:
                print("• Evaluations of each location:")
                for loc in possible_locations:
                    if loc in evaluations:
                        eval_data = evaluations[loc]
                        score = eval_data.get("score", "N/A")
                        reason = eval_data.get("reason", "No reason provided")
                        print(f"  - '{loc}' (Score: {score}/100): {reason}")
            
            print(f"• Selected location: '{selected_location}'")
            print(f"• Selection reasoning: {reasoning}")
            
            if not selected_location:
                return "After exploring multiple possibilities, I couldn't determine a location from your query. Please specify a city or place."
        except Exception as e:
            print(f"Error in ToT reasoning: {e}")
            # Fallback to a simpler approach
            print("Falling back to simple location extraction...")
            selected_location = query.lower()
            for prefix in ["weather in ", "weather at ", "weather for ", "how's the weather in ", "how is the weather in "]:
                if prefix in selected_location:
                    selected_location = selected_location.split(prefix)[1].strip()
                    break
            selected_location = selected_location.strip("?!.,;:")
            possible_locations = [selected_location]
            evaluations = {}
            reasoning = "Simple extraction of location from query"
            print(f"• Extracted location: '{selected_location}'")
        
        # Step 2: Get data for the selected location
        print("\n" + "-"*80)
        print(f"3. GATHERING DATA FOR '{selected_location}'".center(80))
        print("-"*80)
        
        print(f"• Fetching weather data for '{selected_location}'...")
        weather_data = self.weather_service.get_weather(selected_location)
        
        if "error" in weather_data:
            print(f"  ✗ Error: {weather_data['error']}")
            return f"Sorry, I couldn't get weather information: {weather_data['error']}"
        else:
            print(f"  ✓ Successfully retrieved weather data:")
            print(f"    - Temperature: {weather_data['main']['temp']}°C")
            print(f"    - Feels like: {weather_data['main']['feels_like']}°C")
            print(f"    - Conditions: {weather_data['weather'][0]['description']}")
            print(f"    - Humidity: {weather_data['main']['humidity']}%")
        
        print(f"• Fetching Wikipedia data for '{selected_location}'...")
        wiki_data = self.wiki_service.get_location_info(selected_location)
        
        if "error" in wiki_data:
            print(f"  ✗ Error: {wiki_data['error']}")
            print("  → Will proceed with weather data only")
            wiki_summary = "No information available"
            wiki_source_url = ""
        else:
            print(f"  ✓ Successfully retrieved location information")
            wiki_summary = wiki_data.get("summary", "No information available")
            wiki_source_url = wiki_data.get("url", "")
            if "summary" in wiki_data:
                summary_preview = wiki_data["summary"][:100] + "..." if len(wiki_data["summary"]) > 100 else wiki_data["summary"]
                print(f"    - Summary: {summary_preview}")
            if "url" in wiki_data:
                print(f"    - Source: {wiki_data['url']}")
        
        # Step 3: Generate response
        print("\n" + "-"*80)
        print("4. GENERATING RESPONSE WITH ToT REASONING".center(80))
        print("-"*80)
        
        # Extract data from weather response
        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        conditions = weather_data["weather"][0]["description"]
        
        # Weather source information
        weather_source_url = "https://openweathermap.org/"
        
        # Prepare alternatives as context
        if len(possible_locations) > 1:
            alternatives = []
            for loc in possible_locations:
                if loc in evaluations:
                    score = evaluations[loc].get("score", "N/A")
                    alternatives.append(f"- {loc}: Score {score}/100")
            alternatives_text = "\n".join(alternatives)
        else:
            alternatives_text = "No alternative locations considered"
        
        # Create response prompt
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful weather assistant using Tree of Thoughts (ToT) reasoning.
             Create a friendly and informative response about the weather and location based on the provided data.
             
             Start your response with a clear, complete sentence about the current weather.
             
             Include:
             1. Current weather conditions
             2. Temperature and how it feels
             3. Humidity
             4. A brief description of the location (if available)
             
             If there were multiple possible interpretations of the location,
             briefly mention this and explain why you selected this particular location.
             
             IMPORTANT: Always include source attribution at the end of your response.
             Format it as: "Sources: [Weather data from OpenWeatherMap](url), [Location info from Wikipedia](url)"
             """),
            ("human", """
             Query: {query}
             
             Selected location: {location}
             
             Weather data:
             - Temperature: {temperature}°C
             - Feels like: {feels_like}°C
             - Humidity: {humidity}%
             - Conditions: {conditions}
             
             Location information:
             {wiki_summary}
             
             Sources:
             - Weather: {weather_source_url}
             - Location: {wiki_source_url}
             
             Alternative locations considered:
             {alternatives_text}
             
             Selection reasoning: {reasoning}
             """)
        ])
        
        try:
            # Run the response chain
            response = self.llm.invoke(response_prompt.format(
                query=query,
                location=selected_location,
                temperature=temperature,
                feels_like=feels_like,
                humidity=humidity,
                conditions=conditions,
                wiki_summary=wiki_summary,
                weather_source_url=weather_source_url,
                wiki_source_url=wiki_source_url,
                alternatives_text=alternatives_text,
                reasoning=reasoning
            ))
            
            response_text = response.content
            
            if self.debug:
                print("\nRaw Response Text:")
                print(repr(response_text))
                print()
            
            print("• Response generation complete")
            print("="*80 + "\n")
            
            return response_text
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            return f"After considering multiple possible locations, I identified {selected_location} as the most likely. The current temperature is {temperature}°C, feels like {feels_like}°C, with {conditions} and {humidity}% humidity.\n\nSources: [Weather data from OpenWeatherMap]({weather_source_url})"
    
    def _extract_json(self, text):
        """Extract JSON from text that may contain other content"""
        if self.debug:
            print("Extracting JSON from text:")
            print(repr(text))
        
        # First, try to parse the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to find JSON content within the text
            try:
                # Look for the outermost braces
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx]
                    
                    if self.debug:
                        print(f"Extracted JSON string (method 1): {repr(json_str)}")
                    
                    return json.loads(json_str)
                else:
                    # Try a more lenient approach with regex
                    json_pattern = r'({.*})'
                    match = re.search(json_pattern, text, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        
                        if self.debug:
                            print(f"Extracted JSON string (method 2): {repr(json_str)}")
                        
                        return json.loads(json_str)
                    
                    # One more attempt - look for objects with double quotes
                    json_pattern = r'(\{".*"\s*:.*})'
                    match = re.search(json_pattern, text, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        
                        if self.debug:
                            print(f"Extracted JSON string (method 3): {repr(json_str)}")
                        
                        return json.loads(json_str)
                    
                    raise ValueError("No JSON object found in text")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error extracting JSON: {e}")
                print(f"Problematic text: {repr(text)}")
                return {}