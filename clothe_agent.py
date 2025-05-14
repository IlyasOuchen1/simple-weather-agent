import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class ClotheAgent:
    def __init__(self):
        # Load API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.2,  # Slightly higher temperature for more varied clothing suggestions
            api_key=openai_api_key
        )
        
        # Debug flag
        self.debug = True
    
    def generate_clothing_recommendation(self, weather_data, location, time_period="current"):
        """
        Generate clothing recommendations based on weather data
        
        Args:
            weather_data (dict): Weather information including temperature, feels_like, humidity, conditions
            location (str): The location name
            time_period (str): Time period for the recommendation (current, today, tomorrow, week)
            
        Returns:
            str: Clothing recommendation
        """
        print("\n" + "="*80)
        print("CLOTHING RECOMMENDATION PROCESS".center(80))
        print("="*80)
        print(f"Generating clothing recommendations for {location}")
        
        # Extract weather information
        temperature = weather_data.get("main", {}).get("temp", 0)
        feels_like = weather_data.get("main", {}).get("feels_like", 0)
        humidity = weather_data.get("main", {}).get("humidity", 0)
        conditions = weather_data.get("weather", [{}])[0].get("description", "unknown")
        
        # Create the recommendation prompt
        recommendation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful fashion advisor that recommends appropriate clothing based on weather conditions.
             Provide specific, practical clothing suggestions that are appropriate for the weather.
             Consider temperature, humidity, weather conditions, and location when making your recommendation.
             
             For different temperature ranges:
             - Very Hot (>30°C): Very light clothing, sun protection
             - Hot (25-30°C): Light summer clothing
             - Warm (20-25°C): Light layers, maybe a light jacket in the evening
             - Mild (15-20°C): Light layers, light jacket or sweater
             - Cool (10-15°C): Jacket, light layers, possibly a scarf
             - Cold (5-10°C): Warm jacket, sweater, scarf, maybe gloves
             - Very Cold (0-5°C): Winter coat, hat, gloves, scarf, warm layers
             - Freezing (<0°C): Heavy winter clothing, thermal layers, winter accessories
             
             Also consider weather conditions:
             - Rain: Waterproof jacket, umbrella, waterproof footwear
             - Snow: Waterproof and insulated clothing, boots with good traction
             - Wind: Windproof outer layer, secure hat
             - Sun: Sunglasses, hat, sunscreen
             - Humidity: Breathable fabrics in high humidity, extra layers in low humidity
             
             Provide recommendations for:
             1. Top/Upper body clothing
             2. Bottom/Lower body clothing
             3. Footwear
             4. Accessories (if relevant)
             5. Extra items to carry based on conditions
             
             Be conversational and friendly in your response. Mention the weather conditions you're basing your recommendations on.
             """),
            ("human", """
             Location: {location}
             Time period: {time_period}
             Weather conditions:
             - Temperature: {temperature}°C
             - Feels like: {feels_like}°C
             - Humidity: {humidity}%
             - Conditions: {conditions}
             
             What clothing would you recommend for this weather?
             """)
        ])
        
        try:
            # Run the recommendation generation
            response = self.llm.invoke(recommendation_prompt.format(
                location=location,
                time_period=time_period,
                temperature=temperature,
                feels_like=feels_like,
                humidity=humidity,
                conditions=conditions
            ))
            
            recommendation_text = response.content
            
            if self.debug:
                print("\nRaw Recommendation Response:")
                print(recommendation_text)
                print()
            
            print("• Recommendation generation complete")
            print("="*80 + "\n")
            
            return recommendation_text
        except Exception as e:
            print(f"Error generating clothing recommendation: {e}")
            # Fallback response
            return self._generate_fallback_recommendation(temperature, conditions)
    
    def _generate_fallback_recommendation(self, temperature, conditions):
        """Generate a simple fallback recommendation based on temperature and conditions"""
        if temperature > 25:
            clothing = "light clothing such as t-shirts, shorts or light dresses"
        elif temperature > 15:
            clothing = "medium-weight clothing such as long sleeves, light jackets, or jeans"
        elif temperature > 5:
            clothing = "warm clothing such as sweaters, jackets, and long pants"
        else:
            clothing = "heavy winter clothing including a warm coat, scarf, gloves, and hat"
        
        # Add condition-specific advice
        if "rain" in conditions.lower() or "drizzle" in conditions.lower():
            clothing += ". Don't forget a waterproof jacket or umbrella"
        elif "snow" in conditions.lower():
            clothing += ". Make sure to wear waterproof boots and a warm hat"
        elif "clear" in conditions.lower() and temperature > 20:
            clothing += ". Consider wearing sunglasses and applying sunscreen"
        
        return f"Based on the current temperature of {temperature}°C with {conditions}, I recommend wearing {clothing}."