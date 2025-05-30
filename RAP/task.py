import asyncio
from RAP.tool_deeptox import deeptox_agent
from toxicity_schema import TOXICITY_SCHEMA
import os
from datetime import datetime
from time import time

async def analyze_chemical_toxicity():
    chemical_name = "Gentamicin"
    toxicity_type = "nephrotoxicity"
    
    """Analyze toxicity of a chemical by combining chemical properties with deep search."""
    
    # First get the chemical properties
    print(f"\nFetching chemical properties for {chemical_name}...")
    chemprop_query = f"Is {chemical_name} {toxicity_type}?"
    
    # Get the response from the agent and extract the content
    response = await deeptox_agent.arun(chemprop_query)
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/{chemical_name}_{toxicity_type}_{timestamp}.md'

    # Save the response based on its type
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response.content)

if __name__ == "__main__":
    start = time()

    asyncio.run(analyze_chemical_toxicity())

    elapsed = round(time() - start, 2)
    print(f"{elapsed} secs elapsed")
