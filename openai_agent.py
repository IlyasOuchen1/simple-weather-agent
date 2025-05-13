import os
import json
from dotenv import load_dotenv

# Install OpenAI package if not already installed
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "openai"])
    from openai import OpenAI

from api_services import WeatherService, WikipediaService

# Load environment variables
load_dotenv()

class OpenAIWeatherAgent:
    def __init__(self):
        # Load API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize services
        self.weather_service = WeatherService()
        self.wiki_service = WikipediaService()
        
        # Available reasoning types
        self.reasoning_types = ["react", "cot", "tot"]
        
        # Debug flag - set to True to see detailed API responses
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
        """
        Process query using the ReAct pattern:
        1. Reason - Parse and understand query
        2. Plan - Decide what information is needed
        3. Act - Execute actions to get information
        4. Reflect - Evaluate if information is sufficient
        5. Generate - Create the final response
        """
        print("\n" + "="*80)
        print("REACT REASONING PROCESS".center(80))
        print("="*80)
        print("1. Query received:", query)
        
        # Step 1 & 2: Reason and Plan
        print("\n" + "-"*80)
        print("2. REASONING AND PLANNING".center(80))
        print("-"*80)
        planning = self._reason_and_plan(query)
        location = planning.get("location")
        needs_weather = planning.get("needs_weather", True)
        needs_location_info = True
        
        # Display the full JSON result for debugging
        if self.debug:
            print("\nRaw Planning Result:")
            print(json.dumps(planning, indent=2))
            print()
        
        print(f"• Extracted location: '{location}'")
        print(f"• Need weather data: {needs_weather}")
        print(f"• Need location info: {needs_location_info}")
        
        if "time_period" in planning:
            print(f"• Time period: {planning['time_period']}")
            
        if "weather_aspects" in planning and planning["weather_aspects"]:
            print(f"• Weather aspects of interest: {', '.join(planning['weather_aspects'])}")
        
        if not location:
            return "I couldn't identify a location in your query. Please specify a city or place."
        
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
        reflection = self._reflect_on_data(query, location, weather_data, wiki_data)
        
        # Display the full JSON result for debugging
        if self.debug:
            print("\nRaw Reflection Result:")
            print(json.dumps(reflection, indent=2))
            print()
        
        is_sufficient = reflection.get("sufficient", True)
        print(f"• Is the data sufficient? {'Yes' if is_sufficient else 'No'}")
        
        if "notes" in reflection and reflection["notes"]:
            print(f"• Notes: {reflection['notes']}")
            
        if "missing_information" in reflection and reflection["missing_information"]:
            print("• Missing information:")
            for item in reflection["missing_information"]:
                print(f"  - {item}")
        
        # If reflection indicates insufficient data and suggests an action
        if not is_sufficient and "suggested_action" in reflection:
            suggested_action = reflection.get("suggested_action")
            print(f"• Taking suggested action: {suggested_action}")
            
            # Implement any suggested actions here
            if suggested_action == "try_alternative_location" and "alternative_location" in reflection:
                alt_location = reflection.get("alternative_location")
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
        
        # Step 5: Generate response
        print("\n" + "-"*80)
        print("5. GENERATING FINAL RESPONSE".center(80))
        print("-"*80)
        response = self._generate_response(query, location, weather_data, wiki_data, reflection, "react")
        print("• Response generation complete")
        print("="*80 + "\n")
        
        return response
    
    def _process_query_cot(self, query):
        """
        Process query using Chain of Thought (CoT) reasoning:
        1. Step-by-step reasoning to determine location
        2. Get weather and location data
        3. Generate response with CoT reasoning
        """
        print("\n" + "="*80)
        print("CHAIN OF THOUGHT (CoT) REASONING PROCESS".center(80))
        print("="*80)
        print("1. Query received:", query)
        
        # Step 1: Chain of Thought reasoning to extract location
        print("\n" + "-"*80)
        print("2. CHAIN OF THOUGHT REASONING".center(80))
        print("-"*80)
        cot_result = self._reason_with_cot(query)
        location = cot_result.get("location", "")
        reasoning_steps = cot_result.get("reasoning_steps", [])
        
        # Display the full JSON result for debugging
        if self.debug:
            print("\nRaw Chain of Thought Result:")
            print(json.dumps(cot_result, indent=2))
            print()
        
        print(f"• Final extracted location: '{location}'")
        print("• Reasoning steps:")
        for i, step in enumerate(reasoning_steps):
            print(f"  Step {i+1}: {step}")
        
        if not location:
            return "After careful consideration, I couldn't identify a location in your query. Please specify a city or place."
        
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
        else:
            print(f"  ✓ Successfully retrieved location information")
            if "summary" in wiki_data:
                summary_preview = wiki_data["summary"][:100] + "..." if len(wiki_data["summary"]) > 100 else wiki_data["summary"]
                print(f"    - Summary: {summary_preview}")
            if "url" in wiki_data:
                print(f"    - Source: {wiki_data['url']}")
        
        # Step 3: Generate response
        print("\n" + "-"*80)
        print("4. GENERATING RESPONSE WITH COT REASONING".center(80))
        print("-"*80)
        response = self._generate_response(query, location, weather_data, wiki_data, 
                                         {"reasoning_steps": reasoning_steps}, "cot")
        print("• Response generation complete")
        print("="*80 + "\n")
        
        return response
    
    def _process_query_tot(self, query):
        """
        Process query using Tree of Thoughts (ToT) reasoning:
        1. Generate multiple possible locations from the query
        2. Evaluate each option and select the most likely
        3. Get data for the selected location
        4. Generate response with ToT reasoning
        """
        print("\n" + "="*80)
        print("TREE OF THOUGHTS (ToT) REASONING PROCESS".center(80))
        print("="*80)
        print("1. Query received:", query)
        
        # Step 1: Generate multiple possible interpretations
        print("\n" + "-"*80)
        print("2. GENERATING AND EVALUATING MULTIPLE POSSIBLE LOCATIONS".center(80))
        print("-"*80)
        tot_result = self._reason_with_tot(query)
        possible_locations = tot_result.get("possible_locations", [])
        evaluations = tot_result.get("evaluations", {})
        selected_location = tot_result.get("selected_location", "")
        reasoning = tot_result.get("reasoning", "")
        
        # Display the full JSON result for debugging
        if self.debug:
            print("\nRaw Tree of Thoughts Result:")
            print(json.dumps(tot_result, indent=2))
            print()
        
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
        else:
            print(f"  ✓ Successfully retrieved location information")
            if "summary" in wiki_data:
                summary_preview = wiki_data["summary"][:100] + "..." if len(wiki_data["summary"]) > 100 else wiki_data["summary"]
                print(f"    - Summary: {summary_preview}")
            if "url" in wiki_data:
                print(f"    - Source: {wiki_data['url']}")
        
        # Step 3: Generate response
        print("\n" + "-"*80)
        print("4. GENERATING RESPONSE WITH ToT REASONING".center(80))
        print("-"*80)
        tot_metadata = {
            "possible_locations": possible_locations,
            "evaluations": evaluations,
            "selected_location": selected_location,
            "reasoning": reasoning
        }
        
        response = self._generate_response(query, selected_location, weather_data, wiki_data, tot_metadata, "tot")
        print("• Response generation complete")
        print("="*80 + "\n")
        
        return response
    
    def _reason_and_plan(self, query):
        """Enhanced reasoning with planning (for ReAct)"""
        print("• Analyzing query to extract location and plan information needs...")
        
        messages = [
            {"role": "system", "content": """
             You are an assistant that analyzes weather queries.
             First, identify the location in the query.
             Then, plan what information should be collected.
             
             IMPORTANT: For most weather queries, users appreciate both weather data AND
             background information about the location, so set needs_location_info to true
             unless the query explicitly indicates only weather data is needed.
             
             Respond with a JSON object containing:
             {"location": "extracted location", 
              "needs_weather": true/false,
              "needs_location_info": true/false,
              "time_period": "current/today/tomorrow/week",
              "weather_aspects": ["temperature", "precipitation", "etc"]}
              
             If no location is found, return {"location": ""}
             Do not include any punctuation marks or special characters in the location.
             """},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Print the raw response content for debugging
            if self.debug:
                print("\nOpenAI API Response (raw):")
                print(response.choices[0].message.content)
                print()
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Clean up the location string
            if "location" in result and result["location"]:
                result["location"] = result["location"].strip().rstrip('?!.,;:')
            
            print("• Reasoning complete.")
            return result
        except Exception as e:
            print(f"• ✗ Error in reasoning and planning: {e}")
            return {"location": "", "needs_weather": True, "needs_location_info": True}
    
    def _reason_with_cot(self, query):
        """Chain of Thought reasoning to extract location"""
        print("• Performing step-by-step Chain of Thought reasoning...")
        
        messages = [
            {"role": "system", "content": """
             You are an assistant that uses Chain of Thought (CoT) reasoning to analyze weather queries.
             Follow these steps to extract a location from the query:
             1. Identify all potential locations mentioned directly or indirectly in the query.
             2. For each potential location, evaluate whether it's likely the user is asking about weather there.
             3. Consider context, language, and implicit references.
             4. Draw a conclusion about which location the user is most likely asking about.
             
             Respond with a JSON object containing:
             {
                "reasoning_steps": [
                    "Step 1: [your reasoning]",
                    "Step 2: [your reasoning]",
                    "Step 3: [your reasoning]",
                    "Step 4: [your conclusion]"
                ],
                "location": "extracted location"
             }
             
             If no location is found, return {"reasoning_steps": [...], "location": ""}
             """},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Print the raw response content for debugging
            if self.debug:
                print("\nOpenAI API Response (raw):")
                print(response.choices[0].message.content)
                print()
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Clean up the location string
            if "location" in result and result["location"]:
                result["location"] = result["location"].strip().rstrip('?!.,;:')
            
            print("• Chain of Thought reasoning complete.")
            return result
        except Exception as e:
            print(f"• ✗ Error in CoT reasoning: {e}")
            return {"reasoning_steps": ["Error occurred during reasoning"], "location": ""}
    
    def _reason_with_tot(self, query):
        """Tree of Thoughts reasoning to explore multiple possible locations"""
        print("• Exploring multiple possibilities with Tree of Thoughts reasoning...")
        
        messages = [
            {"role": "system", "content": """
             You are an assistant that uses Tree of Thoughts (ToT) reasoning to analyze weather queries.
             For ambiguous queries, explore multiple possible interpretations:
             
             1. Generate 2-4 possible locations that could be referenced in the query.
             2. For each possible location, assign a confidence score (0-100) and provide reasoning.
             3. Select the most likely location based on your evaluation.
             4. Explain your final selection.
             
             Respond with a JSON object containing:
             {
                "possible_locations": ["location1", "location2", ...],
                "evaluations": {
                    "location1": {"score": 85, "reason": "..."},
                    "location2": {"score": 60, "reason": "..."}
                },
                "selected_location": "most likely location",
                "reasoning": "explanation of why this location was selected"
             }
             
             If no locations can be identified, return an empty string for selected_location.
             """},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Print the raw response content for debugging
            if self.debug:
                print("\nOpenAI API Response (raw):")
                print(response.choices[0].message.content)
                print()
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Clean up the selected location string
            if "selected_location" in result and result["selected_location"]:
                result["selected_location"] = result["selected_location"].strip().rstrip('?!.,;:')
            
            print("• Tree of Thoughts reasoning complete.")
            return result
        except Exception as e:
            print(f"• ✗ Error in ToT reasoning: {e}")
            return {
                "possible_locations": [],
                "evaluations": {},
                "selected_location": "",
                "reasoning": "Error occurred during reasoning"
            }
    
    def _reflect_on_data(self, query, location, weather_data, wiki_data):
        """Reflect on the collected data to evaluate its quality and completeness"""
        print("• Reflecting on the quality and completeness of collected data...")
        
        # Prepare a summary of the collected data
        weather_available = weather_data is not None and "error" not in weather_data
        wiki_available = wiki_data is not None and "error" not in wiki_data
        
        data_summary = {
            "query": query,
            "location": location,
            "weather_available": weather_available,
            "location_info_available": wiki_available
        }
        
        # If weather data is available, add some details
        if weather_available:
            try:
                data_summary["weather_details"] = {
                    "has_temperature": "main" in weather_data and "temp" in weather_data["main"],
                    "has_conditions": "weather" in weather_data and len(weather_data["weather"]) > 0,
                    "has_humidity": "main" in weather_data and "humidity" in weather_data["main"]
                }
            except (KeyError, TypeError):
                data_summary["weather_details"] = {"error": "Malformed weather data"}
        
        messages = [
            {"role": "system", "content": """
             You are an assistant that evaluates information quality.
             Analyze the available data and determine:
             1. Is the information sufficient to answer the query?
             2. Is any critical information missing?
             3. Is there anything unusual about the data that should be noted?
             4. Should any additional actions be taken?
             
             IMPORTANT: For most weather queries, users appreciate having background 
             information about the location. If location information is missing,
             consider suggesting getting that information.
             
             Respond with a JSON object containing:
             {"sufficient": true/false,
              "missing_information": ["list", "of", "missing", "info"],
              "notes": "any observations about the data",
              "suggested_action": "action to take if needed",
              "alternative_location": "if location might be ambiguous"}
             """},
            {"role": "user", "content": json.dumps(data_summary)}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Print the raw response content for debugging
            if self.debug:
                print("\nOpenAI API Response (raw):")
                print(response.choices[0].message.content)
                print()
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            print("• Reflection complete.")
            return result
        except Exception as e:
            print(f"• ✗ Error in reflection: {e}")
            # Return a default reflection
            return {"sufficient": True, "notes": "Reflection process encountered an error"}
    
    def _generate_response(self, query, location, weather_data, wiki_data, metadata=None, reasoning_type="react"):
        """Generate a response based on the collected data and metadata from reasoning"""
        print("• Generating natural language response...")
        
        try:
            # Skip if no weather data
            if not weather_data or "error" in weather_data:
                return f"I couldn't find weather information for {location}."
            
            # Extract relevant weather information
            temperature = weather_data["main"]["temp"]
            feels_like = weather_data["main"]["feels_like"]
            humidity = weather_data["main"]["humidity"]
            weather_desc = weather_data["weather"][0]["description"]
            
            # Extract source information
            weather_source = "OpenWeatherMap"
            weather_source_url = "https://openweathermap.org/"
            
            wiki_source = ""
            wiki_source_url = ""
            if wiki_data and "url" in wiki_data:
                wiki_source = "Wikipedia"
                wiki_source_url = wiki_data.get("url", "")
            
            # Format the data for the LLM
            data = {
                "query": query,
                "location": location,
                "weather": {
                    "temperature": temperature,
                    "feels_like": feels_like,
                    "humidity": humidity,
                    "condition": weather_desc,
                    "source": weather_source,
                    "source_url": weather_source_url
                },
                "wiki_info": {
                    "summary": wiki_data.get("summary", "No information available") if wiki_data else "Not requested",
                    "source": wiki_source,
                    "source_url": wiki_source_url
                },
                "metadata": metadata or {},
                "reasoning_type": reasoning_type
            }
            
            # Build system prompt based on reasoning type
            system_prompt = """
             You are a helpful weather assistant.
             Create a friendly and informative response about the weather and location based on the provided data.
             
             Include:
             1. Current weather conditions
             2. Temperature and how it feels
             3. Humidity
             4. A brief description of the location (if available)
             
             IMPORTANT: Always include source attribution at the end of your response.
             Format it as: "Sources: [Weather data from OpenWeatherMap](url), [Location info from Wikipedia](url)"
             """
             
            # Add reasoning-specific instructions
            if reasoning_type == "react":
                system_prompt += """
                You are using ReAct (Reason+Act) reasoning.
                Consider the reflection information in your response. If there were issues with the data,
                you may want to acknowledge them subtly.
                """
            elif reasoning_type == "cot":
                system_prompt += """
                You are using Chain of Thought (CoT) reasoning.
                Include a brief mention of the reasoning process that led to identifying the location,
                but focus primarily on providing weather information.
                """
            elif reasoning_type == "tot":
                system_prompt += """
                You are using Tree of Thoughts (ToT) reasoning.
                If there were multiple possible interpretations of the location,
                briefly mention this and explain why you selected this particular location.
                """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(data)}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"• ✗ Error generating response: {e}")
            return f"I found information about the weather in {location}, but encountered an error creating a response."

    def get_thought_process(self, query):
        """
        For debugging: Show the full ReAct thought process
        """
        print("\n--- ReAct Debug Mode ---")
        
        # Step 1 & 2: Reason and Plan
        planning = self._reason_and_plan(query)
        location = planning.get("location", "")
        
        # Step 3: Act
        weather_data = None
        wiki_data = None
        
        if location:
            weather_data = self.weather_service.get_weather(location)
            wiki_data = self.wiki_service.get_location_info(location)
        
        # Step 4: Reflect
        reflection = None
        if location and weather_data:
            reflection = self._reflect_on_data(query, location, weather_data, wiki_data)
        
        # Output the full thought process
        thought_process = {
            "query": query,
            "planning": planning,
            "actions": {
                "location_extracted": location,
                "weather_data": weather_data,
                "wiki_data": wiki_data
            },
            "reflection": reflection
        }
        
        return json.dumps(thought_process, indent=2)