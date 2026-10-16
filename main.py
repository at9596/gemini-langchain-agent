import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from google.genai.errors import ServerError

# Load environment variables
load_dotenv()

# 1. Define a custom tool using the @tool decorator
@tool
def calculate_shipping_cost(weight_kg: float, distance_km: float) -> float:
    """Calculate the shipping cost based on package weight in kg and distance in km."""
    base_rate = 5.0
    cost = base_rate + (weight_kg * 1.5) + (distance_km * 0.1)
    return round(cost, 2)

tools = [calculate_shipping_cost]

# 2. Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# 3. Bind the tools to the Gemini model
llm_with_tools = llm.bind_tools(tools)

def main():
    query = "How much will it cost to ship a 4.5 kg package over a distance of 120 km?"
    print(f"User Query: {query}\n")
    
    # Retry loop to handle temporary 503 server capacity spikes
    max_retries = 3
    response = None
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to invoke Gemini model (Attempt {attempt + 1}/{max_retries})...")
            response = llm_with_tools.invoke(query)
            break
        except ServerError as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Server is busy (503). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

    # 4. Check if Gemini decided to invoke a tool
    if response and response.tool_calls:
        print("\nGemini requested a tool call:")
        for tool_call in response.tool_calls:
            print(f"Tool Name: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            
            # Execute the tool locally using the arguments provided by Gemini
            if tool_call['name'] == "calculate_shipping_cost":
                result = calculate_shipping_cost.invoke(tool_call['args'])
                print(f"\nTool Execution Result: ${result}")
    elif response:
        print("Direct Response:", response.content)

if __name__ == "__main__":
    main()