from datetime import datetime
import asyncio
from agno.playground import Playground, serve_playground_app

from src.tool_toxtransformer import tox_transformer_agent
from src.tool_deeptox import deeptox_agent
from src.workflows import analyze_chemical_workflow
# from src.tool_diffdock import diffdock_agent
# from src.main_agent import main_agent

today = datetime.now().strftime("%Y-%m-%d")

# Create the Playground instance with the agents
playground = Playground(agents=[deeptox_agent])

# Get the FastAPI app
app = playground.get_app()

# async def example_workflow():
#     """Example of using the chemical analysis workflow"""
#     chemical_name = "Amphotericin B"
#     toxicity_type = "nephrotoxicity"
#     result = await analyze_chemical_workflow(chemical_name, toxicity_type, tox_transformer_agent, deeptox_agent)
#     print(f"Analysis complete. Results saved to: {result.get('filename')}")

if __name__ == "__main__":
    # Run the example workflow
    # asyncio.run(example_workflow())
    
    # Start the playground app
    serve_playground_app("main:app", reload=True)
