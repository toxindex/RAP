import asyncio
import json
import os
import requests
import httpx
import pubchempy as pcp
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agno.tools import tool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants
CHEMPROP_PROPERTY_URL = (
    "http://chemprop-transformer-alb-2126755060."
    + "us-east-1.elb.amazonaws.com/predict?property_token=5042&"
)
def _extract_chemical_name(prompt: str) -> str:
    """Extract the chemical name from a user prompt

    Args:
        prompt: User prompt that mentions a chemical name

    Returns:
        The extracted chemical name or the original prompt if extraction fails
    """
    # Set up claude LLM
    extractor_llm = ChatOpenAI(
        model="gpt-4.1-2025-04-14", temperature=0, api_key=OPENAI_API_KEY
    )
    messages = [
        (
            "system",
            "Given a prompt, extract the chemical name. Return only the chemical name.",
        ),
        ("human", prompt),
    ]
    response = extractor_llm.invoke(messages)

    return response.content

async def _get_chemprop_request(inchi: str, retries: int = 3, delay: float = 2.0):
    """Make a request with the Chemprop API with inchi of the chemical name and return json of properties

    Args:
        inchi: InChI of the chemical name
        retries: Number of times to retry on failure
        delay: Initial delay between retries (seconds)
    """
    url = CHEMPROP_PROPERTY_URL + "inchi=" + inchi

    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                response = await client.get(url, timeout=500)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"All {retries} attempts failed.")
                    return "Chemprop API is temporarily unavailable. Please try again later."

def _format_chemprop_response(response: dict):
    """Use LLM to extract relevant chemical properties from Chemprop JSON."""
    try:
        # Prepare the predictions JSON
        if isinstance(response, str):
            return response
        entry = response[0] if isinstance(response, list) and response else response
        preds = entry.get("predictions", entry)
        preds_str = json.dumps(preds, separators=(",", ":"), ensure_ascii=False)
        # Instruction for Claude
        instruction = (
            "You are an expert chemist. Given the following JSON of extract chemical properties "
            "that correspond to the following short codes, include physiochemical, structural, "
            "environmental, and ADME properties and all other possible properties. Return a JSON "
            "object with the source included for each property."
        )
        # Call Claude
        extractor_llm = ChatOpenAI(
            model="gpt-4.1-2025-04-14", temperature=0, api_key=OPENAI_API_KEY
        )
        messages = [
            ("system", "You are a JSON extraction assistant specializing in chemistry."),
            ("human", instruction + "\n\nJSON:\n" + preds_str[:150000]),
        ]
        result = extractor_llm.invoke(messages)
        return result.content
    except Exception as e:
        return f"Error processing chemical properties: {str(e)}"

async def _get_inchi_pubchem(query_chemical: str):
    """
    Get the inchi for the given chemical name using PubChem API
    """
    try:
        return pcp.get_compounds(query_chemical, 'name')[0].inchi
    except Exception as e:
        print(f"An error occurred: {e}")
        
        try:
            print("trying to query using alternative names")
            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pccompound',
                'term': query_chemical,
                'retmode': 'json',
                'retmax': 5
            }
            r = requests.get(esearch_url, params=params)
            cids = r.json().get('esearchresult').get('idlist')

            records = []
            for cid in cids:
                try:
                    compound = pcp.get_compounds(cid, 'cid')[0]
                    if compound.inchi:
                        records.append((compound, compound.inchi))
                except Exception as e:
                    continue
            
            records.sort(key=lambda x: len(x[1]))
            best = records[0][0]
            return best.inchi
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error getting InChI for '{query_chemical}'"

@tool(description="This tool will allow you to look up information about chemical properties of a chemical. If asked about chemical properties, use this tool.")
def get_chemprop(query_chemical: str):
    """Get the chemical properties for a chemical

    Args:
        query_chemical: Chemical name provided by the user in a prompt
    """
    # Extract the chemical name from the user's prompt. this is turned off to handle mixtures.
    # query_chemical = _extract_chemical_name(query_chemical)
    # Get InChI
    inchi = asyncio.run(_get_inchi_pubchem(query_chemical))
    if not inchi:
        return f"No InChI found for '{query_chemical}'"

    # Get inchis if more than one inchi is in string
    inchis = inchi.split("\n")
    inchis = [inchi for inchi in inchis if inchi]
    if len(inchis) == 1:
        response = asyncio.run(_get_chemprop_request(inchi))
    else:
        # If a response is not valid, try the next inchi
        for inchi in inchis:
            response = asyncio.run(_get_chemprop_request(inchi))
            if response:
                break

    if not response:
        return f"No response from the Chemprop API for '{query_chemical}'"

    return _format_chemprop_response(response) 