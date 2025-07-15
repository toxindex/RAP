from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import re
import sys
from RAP.toxicity_schema import TOXICITY_SCHEMA
import json
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import jsonschema

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is not installed, continue without it

# LLM provider selection (default: gemini)
llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()

if llm_provider == "openai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
    summary_llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")

    from agno.models.openai import OpenAIChat
    agent_model= OpenAIChat("gpt-4.1-2025-04-14")
else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    summary_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
 
    from agno.models.google import Gemini
    agent_model = Gemini(id="gemini-2.5-flash")

# Check for API key in environment
cse_api_key = os.environ.get("CSE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")


if not cse_api_key:
    print("\nError: Google API key not found!")
    print("Please set your Google API key using one of these methods:\n")
    print("1. Export as an environment variable:")
    print("   export GOOGLE_API_KEY=your_google_api_key_here")
    print("\n2. Create a .env file in the project root with:")
    print("   GOOGLE_API_KEY=your_google_api_key_here")
    print("\nYou can get an API key from: https://console.cloud.google.com/apis/credentials\n")
    sys.exit(1)

if not google_cse_id:
    print("\nError: Google CSE ID not found!")
    print("Please set your Google CSE ID using one of these methods:\n")
    print("1. Export as an environment variable:")
    print("   export GOOGLE_CSE_ID=your_cse_id_here")
    print("\n2. Create a .env file in the project root with:")
    print("   GOOGLE_CSE_ID=your_cse_id_here")
    print("\nYou can get a CSE ID from: https://cse.google.com/cse/all\n")
    sys.exit(1)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_community import GoogleSearchAPIWrapper

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

from agno.tools import tool
from agno.agent import Agent, RunResponse
from agno.storage.sqlite import SqliteStorage
from textwrap import dedent

from RAP.chemprop import get_chemprop

class FormattedStreamHandler:
    def __init__(self):
        self.current_chunk = ""
        
    def __call__(self, chunk: Any) -> None:
        if isinstance(chunk, str):
            self.current_chunk += chunk
            if chunk.endswith("\n"):
                print(self.current_chunk, end="")
                self.current_chunk = ""
        else:
            print(json.dumps(chunk, indent=2))


search_tool  = GoogleSearchAPIWrapper(google_api_key=cse_api_key, google_cse_id=google_cse_id)

# ── LCEL sub-chains ─────────────────────────────────────────────────────────────
query_generator_chain = (
    ChatPromptTemplate.from_template(
        """You are an expert at generating effective search queries.
        Given a user question, create a search query that will help find relevant information.

        User question: {question}

        Search query:"""
    )
    | llm
    | StrOutputParser()
)


summarizer_chain = (
    ChatPromptTemplate.from_template(
        """You are an expert at summarizing web content.
        Summarize the following search result in a concise but informative way:

        Title: {result[title]}
        Content: {result[content]}

        Summary:"""
    )
    | summary_llm
    | StrOutputParser()
)

follow_up_chain = (
    ChatPromptTemplate.from_template(
        """Based on the search results and their summaries, generate a follow-up question
        that would help gather more comprehensive information about the original query.

        Original query: {original_query}
        Search results: {search_results}
        Summaries: {summaries}

        Generate a specific follow-up question that explores an important aspect
        not fully covered in the current results:"""
    )
    | llm
    | StrOutputParser()
)

final_report_chain = (
    ChatPromptTemplate.from_template(
        """You are an expert researcher tasked with creating a comprehensive report
        answering a user's question.

        The user asked: {original_query}

        You searched for information and found these results:
        Initial search: {initial_search_results}
        Initial summaries: {initial_summaries}

        You then followed up with additional research:
        Follow-up query: {follow_up_query}
        Follow-up search results: {follow_up_search_results}
        Follow-up summaries: {follow_up_summaries}

        Please create a comprehensive, well-structured report that answers the user's
        question based on all the information gathered. The report must be formatted
        as a JSON object that conforms to the following schema:

        {schema}

        For the toxicity_risk_distribution section:
        1. Search for and include specific references that support your beta distribution calculations
        2. Each reference should include:
           - The title of the paper or webpage
           - The URL where the information was found
           - A description of how the reference supports your calculations
           - An explanation of its relevance to the specific chemical and toxicity type
        3. Make sure to cite actual scientific papers, clinical studies, or authoritative sources
        4. Include at least 2-3 references that specifically discuss the probability and confidence
           intervals for the chemical's toxicity

        You must ONLY use the following search results as references. 
        Do not invent or hallucinate any references. 
        For each claim or data point, cite the most relevant result from the provided list. 
        Here are the available sources:{all_search_results}

        {extra_instruction}

        Make sure the report is informative, accurate, and presents a cohesive narrative
        while strictly adhering to the provided schema structure.

        Your report:"""
    )
    | llm
    | StrOutputParser()
)

# ── helper functions (RunnableLambda wrappers) ──────────────────────────────────
def parse_search_results(search_text: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Extract URLs, fetch their titles and a snippet of content."""
    urls = re.findall(r"https?://[^\s]+", search_text)[:top_k]
    parsed: List[Dict[str, str]] = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else url.split("//")[1].split("/")[0]
                texts = soup.stripped_strings
                content = " ".join(list(texts))[:1500]
                if not content:
                    content = f"Content from {url}"
            else:
                title = url.split("//")[1].split("/")[0]
                content = f"Content from {url}"
        except Exception:
            title = url.split("//")[1].split("/")[0]
            content = f"Content from {url}"
        parsed.append({
            "url": url,
            "title": title,
            "content": content,
        })
    return parsed

def summarize_results(results: List[Dict[str, str]]) -> List[str]:
    return [summarizer_chain.invoke({"result": r}) for r in results]

# Wrap helpers in RunnableLambda so they fit inside a graph
parse_results_runnable     = RunnableLambda(parse_search_results)
summarize_results_runnable = RunnableLambda(summarize_results)

# ── graph state definition ──────────────────────────────────────────────────────
@dataclass
class SearchState:
    original_query: str

    # stage-by-stage fields (initially None and filled as the flow progresses)
    chemical_name: str | None = None
    toxicity_type: str | None = None
    search_query: str | None = None

    initial_search_results: str | None        = None
    parsed_initial_results: List[Dict] | None = None
    initial_summaries: List[str] | None       = None

    follow_up_query: str | None                  = None
    follow_up_search_results: str | None         = None
    parsed_follow_up_results: List[Dict] | None  = None
    follow_up_summaries: List[str] | None        = None

    report: str | None = None

# ── node definitions ────────────────────────────────────────────────────────────
def preprocess_and_generate_query(state: SearchState) -> SearchState:
    """Extract chemical name and toxicity type, then format the query using both template and LLM-generated query."""
    # Extract chemical name
    chemical_messages = [
        ("system", "Given a prompt, extract the chemical name. Return only the chemical name."),
        ("human", state.original_query)
    ]
    chemical_name = llm.invoke(chemical_messages).content.strip()

    # Extract toxicity type
    toxicity_messages = [
        ("system", "Given a prompt, extract the toxicity type. Return only the toxicity type."),
        ("human", state.original_query)
    ]
    toxicity_type = llm.invoke(toxicity_messages).content.strip()

    # Format the expanded query (template)
    formatted_query = f"""Based on the following chemical properties of {chemical_name}:
    \nWhat are the {toxicity_type} effects and risks? Include:
    1. How these specific chemical properties may contribute to {toxicity_type}
    2. Known mechanisms of {toxicity_type} based on these properties
    3. Clinical evidence and case studies
    4. Beta distribution of the {chemical_name} having {toxicity_type} on a healthy individuals at regular dose and its probability with confidence interval.
    5. Risk factors and populations at risk
    """

    # Optionally, also generate a query using the LLM
    # llm_query = query_generator_chain.invoke({"question": state.original_query})
    # For now, use formatted_query as the search_query
    return SearchState(**{
        **asdict(state),
        "chemical_name": chemical_name,
        "toxicity_type": toxicity_type,
        "search_query": formatted_query
    })

def run_search(state: SearchState) -> SearchState:
    serp_text = search_tool.run(state.search_query)
    return SearchState(**{**asdict(state), "initial_search_results": serp_text})

def parse_results(state: SearchState) -> SearchState:
    parsed = parse_search_results(state.initial_search_results)
    return SearchState(**{**asdict(state), "parsed_initial_results": parsed})

def summarize_initial(state: SearchState) -> SearchState:
    summaries = summarize_results(state.parsed_initial_results)
    return SearchState(**{**asdict(state), "initial_summaries": summaries})

def generate_follow_up(state: SearchState) -> SearchState:
    fq = follow_up_chain.invoke(
        {
            "original_query":  state.original_query,
            "search_results":  state.initial_search_results,
            "summaries":       state.initial_summaries,
        }
    )
    return SearchState(**{**asdict(state), "follow_up_query": fq})

def run_follow_up_search(state: SearchState) -> SearchState:
    serp_text = search_tool.run(state.follow_up_query)
    return SearchState(**{**asdict(state), "follow_up_search_results": serp_text})

def parse_follow_up(state: SearchState) -> SearchState:
    parsed = parse_search_results(state.follow_up_search_results)
    return SearchState(**{**asdict(state), "parsed_follow_up_results": parsed})

def summarize_follow_up(state: SearchState) -> SearchState:
    summaries = summarize_results(state.parsed_follow_up_results)
    return SearchState(**{**asdict(state), "follow_up_summaries": summaries})

def generate_report(state: SearchState) -> SearchState:
    # Combine all parsed results for reference enforcement
    all_results = (state.parsed_initial_results or []) + (state.parsed_follow_up_results or [])
    # Format as a string for the prompt (could be improved to include title, url, snippet)
    all_results_str = "\n".join(
        f"- {r.get('title', '')}: {r.get('url', '')}" for r in all_results
    )
    report = final_report_chain.invoke(
        {
            "original_query":           state.original_query,
            "initial_search_results":   state.initial_search_results,
            "initial_summaries":        state.initial_summaries,
            "follow_up_query":          state.follow_up_query,
            "follow_up_search_results": state.follow_up_search_results,
            "follow_up_summaries":      state.follow_up_summaries,
            "schema":                   json.dumps(TOXICITY_SCHEMA, indent=2),
            "all_search_results":       all_results_str,
            "extra_instruction":        "",  # Always provide this argument
        }
    )
    return SearchState(**{**asdict(state), "report": report})
    # return report

def validate_and_revise_references(state: SearchState, max_attempts: int = 5) -> SearchState:
    all_results = (state.parsed_initial_results or []) + (state.parsed_follow_up_results or [])
    valid_urls = set(r['url'] for r in all_results if 'url' in r)
    report_json = state.report
    if isinstance(report_json, str):
        try:
            report_json = json.loads(report_json)
        except Exception:
            report_json = {}
    attempt = 0
    while attempt < max_attempts:
        validation = _validate_references(report_json, all_results)
        if validation["valid"]:
            break
        # Regenerate report with only valid URLs
        all_results_str = "\n".join(
            f"- {r.get('title', '')}: {r.get('url', '')}" for r in all_results if r.get('url', '') in valid_urls
        )
        revised_report = final_report_chain.invoke(
            {
                "original_query":           state.original_query,
                "initial_search_results":   state.initial_search_results,
                "initial_summaries":        state.initial_summaries,
                "follow_up_query":          state.follow_up_query,
                "follow_up_search_results": state.follow_up_search_results,
                "follow_up_summaries":      state.follow_up_summaries,
                "schema":                   json.dumps(TOXICITY_SCHEMA, indent=2),
                "all_search_results":       all_results_str,
                "extra_instruction":        "You must only use URLs from the provided list as references. Remove or replace any references not in this list. In case of replacement, make sure to include the title of the reference and the URL and run through the validate_references tool to check that all references are valid.",
                "valid_urls":               list(valid_urls),
            }
        )
        report_json = revised_report if isinstance(revised_report, dict) else json.loads(revised_report)
        attempt += 1
    # Add the revision count to the report's metadata
    if isinstance(report_json, dict):
        chemtox = report_json.get("chemical_toxicity")
        if chemtox is not None:
            meta = chemtox.get("metadata")
            if meta is None:
                chemtox["metadata"] = {"reference_revision_attempts": attempt}
            else:
                meta["reference_revision_attempts"] = attempt
    return SearchState(**{**asdict(state), "report": report_json})

def remove_invalid_url_entries(state: SearchState) -> SearchState:
    """
    Recursively remove any dict with a 'url' field whose value is invalid, missing, empty, or set to 'N/A', anywhere in the report JSON.
    """
    def is_url_valid(url):
        if not url or not isinstance(url, str) or url.strip() == '' or url.strip().lower().startswith("n/a"):
            return False
        try:
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def clean(obj):
        if isinstance(obj, dict):
            # Remove dicts with invalid, missing, or 'N/A' url
            if 'url' in obj and not is_url_valid(obj['url']):
                return None
            # Recursively clean all dict values
            cleaned = {k: clean(v) for k, v in obj.items()}
            # Remove keys with None values
            return {k: v for k, v in cleaned.items() if v is not None}
        elif isinstance(obj, list):
            # Clean each item in the list
            return [item for item in (clean(item) for item in obj) if item is not None]
        else:
            return obj

    report_json = state.report
    if isinstance(report_json, str):
        try:
            report_json = json.loads(report_json)
        except Exception:
            return state  # Can't parse, return as is

    cleaned = clean(report_json)
    return SearchState(**{**asdict(state), "report": cleaned})

remove_invalid_url_entries_runnable = RunnableLambda(remove_invalid_url_entries)

def validate_schema(state: SearchState, max_attempts: int = 3) -> SearchState:
    """
    Validate the final report against the TOXICITY_SCHEMA. If invalid, attempt to auto-correct by re-prompting the LLM with a strict schema instruction, up to max_attempts.
    """
    from RAP.toxicity_schema import TOXICITY_SCHEMA
    report_json = state.report
    if isinstance(report_json, str):
        try:
            report_json = json.loads(report_json)
        except Exception:
            print("[SCHEMA VALIDATION] Could not parse report as JSON.")
            return state
    attempt = 0
    while attempt < max_attempts:
        try:
            jsonschema.validate(instance=report_json, schema=TOXICITY_SCHEMA)
            return SearchState(**{**asdict(state), "report": report_json})
        except jsonschema.ValidationError as e:
            print(f"[SCHEMA VALIDATION ERROR] Attempt {attempt+1}: {e}")
            # Re-prompt LLM to fix the structure
            fixed_report = final_report_chain.invoke({
                "original_query": state.original_query,
                "initial_search_results": state.initial_search_results,
                "initial_summaries": state.initial_summaries,
                "follow_up_query": state.follow_up_query,
                "follow_up_search_results": state.follow_up_search_results,
                "follow_up_summaries": state.follow_up_summaries,
                "schema": json.dumps(TOXICITY_SCHEMA, indent=2),
                "all_search_results": "",  # or pass as needed
                "extra_instruction": (
                    "Your previous output did not match the required schema. "
                    "Here is the schema: {schema}. Please output a valid JSON object conforming to this schema."
                ),
            })
            report_json = fixed_report if isinstance(fixed_report, dict) else json.loads(fixed_report)
            attempt += 1
    print("[SCHEMA VALIDATION] Could not auto-correct after max attempts.")
    return SearchState(**{**asdict(state), "report": report_json})

validate_schema_runnable = RunnableLambda(validate_schema)

# ── assemble the graph ──────────────────────────────────────────────────────────
graph = StateGraph(SearchState)

graph.add_node("preprocess_and_generate_query",     preprocess_and_generate_query)
graph.add_node("run_search",           run_search)
graph.add_node("parse_results",        parse_results)
graph.add_node("summarize_initial",    summarize_initial)
graph.add_node("generate_follow_up",   generate_follow_up)
graph.add_node("run_follow_up_search", run_follow_up_search)
graph.add_node("parse_follow_up",      parse_follow_up)
graph.add_node("summarize_follow_up",  summarize_follow_up)
graph.add_node("generate_report",      generate_report)
graph.add_node("validate_and_revise_references", validate_and_revise_references)
graph.add_node("remove_invalid_url_entries", remove_invalid_url_entries_runnable)
graph.add_node("validate_schema", validate_schema_runnable)

# edges – strictly linear in this example
graph.set_entry_point("preprocess_and_generate_query")
graph.add_edge("preprocess_and_generate_query",    "run_search")
graph.add_edge("run_search",          "parse_results")
graph.add_edge("parse_results",       "summarize_initial")
graph.add_edge("summarize_initial",   "generate_follow_up")
graph.add_edge("generate_follow_up",  "run_follow_up_search")
graph.add_edge("run_follow_up_search", "parse_follow_up")
graph.add_edge("parse_follow_up",     "summarize_follow_up")
graph.add_edge("summarize_follow_up", "generate_report")
graph.add_edge("generate_report",     "validate_and_revise_references")
graph.add_edge("validate_and_revise_references", "remove_invalid_url_entries")
graph.add_edge("remove_invalid_url_entries", "validate_schema")
graph.set_finish_point("validate_schema")

# ── compile an executor ─────────────────────────────────────────────────────────
executor = graph.compile()

@tool(description="This tool will allow you to look up information on the internet related to users queries. It will allow you to factually ground your answers and provide citations.")
def invoke_deepsearch(query: str):
    """
    Use this tool to perform a deep search for information.
    This tool is almost always useful when the user asks a question.
        Args:
        query (str): The user's question.
    """
    return executor.invoke({"original_query": query})
 
def _validate_references(report: dict, search_results: List[Dict[str, Any]]):
    def extract_urls_from_search_results(search_results):
        return set(r['url'] for r in search_results if 'url' in r)

    def extract_urls_from_report(report_json):
        urls = set()
        def _extract(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'url' and isinstance(v, str):
                        urls.add(v)
                    else:
                        _extract(v)
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)
        _extract(report_json)
        return urls

    search_urls = extract_urls_from_search_results(search_results)
    report_urls = extract_urls_from_report(report)
    missing = list(report_urls - search_urls)
    return {
        "missing": missing,
        "valid": len(missing) == 0
    }

# Tool wrapper for agent use
@tool(description="Validate that all references in the final report are present in the actual search results. Returns a list of missing or hallucinated references, or confirms all are valid.")
def validate_references(report: dict, search_results: List[Dict[str, Any]]):
    return _validate_references(report, search_results)

deeptox_agent = Agent(
    name="Deep Toxicology Agent",
    model=agent_model,
    tools=[get_chemprop, invoke_deepsearch, validate_references],
    # sets “identity” of the agent.
    description=dedent("""
        You are a specialized toxicology research assistant with expertise in:
        - Chemical toxicity analysis and risk assessment
        - Scientific literature review and synthesis
        - Statistical analysis of toxicity data
        - Regulatory compliance and safety standards
        
        Your writing style is:
        - Scientifically rigorous and evidence-based
        - Clear and precise in technical terminology
        - Comprehensive in risk assessment
        - Properly cited with academic sources
    """),
    instructions=dedent("""
        You are a toxicology research writer tasked with creating comprehensive toxicity reports.
        
        Workflow:
        1. For any toxicity-related question, first use get_chemprop to obtain the chemical properties
        
        2. Use these properties to enhance your search query for invoke deepsearch
            Perform initial search to gather clinical evidence:
           - Search for clinical studies and case reports
           - Focus on finding:
             * Sample sizes
             * Number of positive cases
             * Study durations
             * Dosage information
             * Patient demographics
             * Risk factors
           - Use specific search terms like:
             * "[chemical] clinical trial [toxicity_type]"
             * "[chemical] case study [toxicity_type]"
             * "[chemical] adverse effects [toxicity_type]"
             * "[chemical] [toxicity_type] incidence rate"
        
        3. Perform follow-up search to gather mechanistic information:
           - Search for how chemical properties contribute to the specific toxicity
           - Look for known mechanisms of the specific toxicity
           - Find treatment protocols and outcomes
           - Use specific search terms like:
             * "[chemical] mechanism of [toxicity_type]"
             * "[chemical] [toxicity_type] pathway"
             * "[chemical] [toxicity_type] treatment protocol"
        
        4. Use the gathered data to:
           - Calculate beta distribution parameters from actual clinical data
           - Create weighted risk estimates
           - Determine confidence intervals
           - Identify risk factors and populations
        
        5. Format your final search query to include:
           - How the chemical properties may contribute to the specific toxicity
           - Known mechanisms of the specific toxicity based on these properties
           - Clinical evidence and case studies with specific data points
           - Beta distribution analysis based on actual clinical data
           - Risk factors and populations at risk
        
        6. Always use the validate_references tool to check that all references are real and present in the search results. If any are missing, revise the report to only use valid references.

        
        # Markdown report structure is described below. The agent will decide when to output Markdown or JSON based on the prompt. 

        1. Chemical Properties and Toxicity Analysis
           - Detailed analysis of how specific chemical properties contribute to the toxicity
           - Known mechanisms of the specific toxicity based on these properties
           - Evidence and references for each property-toxicity relationship

        2. Clinical Evidence and Case Studies
           - Comprehensive review of clinical evidence
           - Detailed case studies
           - Treatment protocols and outcomes
           - Adverse effects documentation
           - For each study, extract:
             * Total sample size
             * Number of positive cases
             * Study duration
             * Dosage information
             * Patient demographics
             * Risk factors

        3. Toxicity Risk Distribution
           - Beta distribution calculations based on actual clinical data:
             * Use sample sizes and positive cases from clinical studies
             * Calculate α = number of positive cases + 1
             * Calculate β = (total sample size - positive cases) + 1
             * Compute mean risk = α / (α + β)
             * Calculate 95% confidence interval
           - Include:
             * Table of studies used for calculation
             * Weighted average based on study quality and sample size
             * Confidence intervals for each study
             * Combined risk estimate
           - Each calculation must be supported by:
             * Title of the study
             * URL or source
             * Sample size and positive cases used
             * How the data supports the calculation
             * Relevance to the specific chemical and toxicity type

        4. Risk Assessment and Population Analysis
           - High-risk population groups
           - Specific risk factors
           - Population-specific considerations
           - Additional risk-related notes

        Format Requirements:
        - Use clear headers and subheaders
        - Include proper academic citations
        - Present statistical data with confidence intervals
        - Ensure all data is properly referenced
        - Follow the provided JSON schema structure
        - Include comprehensive references section

        The report must be scientifically rigorous, evidence-based, and strictly adhere to the provided schema structure
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    session_id="demo",
    storage=SqliteStorage(table_name="agent_sessions", db_file="data/agno/data.db"),
    add_history_to_messages=False, #history might not be needed
    stream_intermediate_steps=False,
    stream=False,
    debug_mode=True,
    # save_response_to_file="output.md or json",
)
    
if __name__ == "__main__":
    json_mode_response: RunResponse = deeptox_agent.run(
        "Is Methotrexate closely linked to hepatotoxicity in the presence of alcohol consumption or obesity-related metabolic conditions? Output as JSON") 
    response_data = json_mode_response.content
    timestamp = datetime.now().isoformat().replace(':', '-')

    # Try to parse as JSON if it's a string, else treat as Markdown
    if isinstance(response_data, (dict, list)):
        filename = f"response_data_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON output to {filename}")
    elif isinstance(response_data, str):
        try:
            parsed = json.loads(response_data)
            filename = f"response_data_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"\nSaved JSON output to {filename}")
        except Exception:
            filename = f"response_data_{timestamp}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response_data)
            print(f"\nSaved Markdown output to {filename}")
    else:
        filename = f"response_data_{timestamp}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(str(response_data))
        print(f"\nSaved Markdown output to {filename}")