from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import re
import sys
from RAP.toxicity_schema import TOXICITY_SCHEMA
import json
import os

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is not installed, continue without it

# Check for API key in environment
api_key = os.environ.get("OPENAI_API_KEY")


if not api_key:
    print("\nError: OpenAI API key not found!")
    print("Please set your OpenAI API key using one of these methods:\n")
    print("1. Export as an environment variable:")
    print("   export OPENAI_API_KEY=your_api_key_here")
    print("\n2. Create a .env file in the project root with:")
    print("   OPENAI_API_KEY=your_api_key_here")
    print("\nYou can get an API key from: https://platform.openai.com/api-keys\n")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

from agno.tools import tool
from agno.agent import Agent
from agno.storage.sqlite import SqliteStorage
from textwrap import dedent
from agno.models.openai import OpenAIChat

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

# ── LLMs & tools ────────────────────────────────────────────────────────────────
llm          = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
summary_llm  = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
search_tool  = DuckDuckGoSearchRun()
# search_tool  = GoogleSearchAPIWrapper()

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


    # search_query = f"""Based on the following chemical properties of {chemical_name}:
    # {properties}
    
    # What are the {toxicity_type} effects and risks? Include:
    # 1. How these specific chemical properties may contribute to {toxicity_type}
    # 2. Known mechanisms of {toxicity_type} based on these properties
    # 3. Clinical evidence and case studies
    # 4. Beta distribution of the {chemical_name} having {toxicity_type} on a healthy individuals at regular dose and its probability with confidence interval.
    # 5. Risk factors and populations at risk
    # """ 

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

        Make sure the report is informative, accurate, and presents a cohesive narrative
        while strictly adhering to the provided schema structure.

        Your report:"""
    )
    | llm
    | StrOutputParser()
)

# ── helper functions (RunnableLambda wrappers) ──────────────────────────────────
def parse_search_results(search_text: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Very light URL scraper for Google search result string."""
    urls = re.findall(r"https?://[^\s]+", search_text)[:top_k]
    parsed: List[Dict[str, str]] = []
    for url in urls:
        try:
            parsed.append(
                {
                    "url": url,
                    "title": url.split("//")[1].split("/")[0],  # crude domain title
                    "content": f"Content from {url}",            # placeholder
                }
            )
        except Exception:
            continue
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
def preprocess_query(state: SearchState) -> SearchState:
    """Extract chemical name and toxicity type, then format the query"""
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

    # Format the expanded query
    formatted_query = f"""Based on the following chemical properties of {chemical_name}:
    
    What are the {toxicity_type} effects and risks? Include:
    1. How these specific chemical properties may contribute to {toxicity_type}
    2. Known mechanisms of {toxicity_type} based on these properties
    3. Clinical evidence and case studies
    4. Beta distribution of the {chemical_name} having {toxicity_type} on a healthy individuals at regular dose and its probability with confidence interval.
    5. Risk factors and populations at risk
    """

    return SearchState(**{
        **asdict(state),
        "chemical_name": chemical_name,
        "toxicity_type": toxicity_type,
        "search_query": formatted_query
    })

def generate_query(state: SearchState) -> SearchState:
    q = query_generator_chain.invoke({"question": state.original_query})
    return SearchState(**{**asdict(state), "search_query": q})

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
    report = final_report_chain.invoke(
        {
            "original_query":           state.original_query,
            "initial_search_results":   state.initial_search_results,
            "initial_summaries":        state.initial_summaries,
            "follow_up_query":          state.follow_up_query,
            "follow_up_search_results": state.follow_up_search_results,
            "follow_up_summaries":      state.follow_up_summaries,
            "schema":                   json.dumps(TOXICITY_SCHEMA, indent=2)
        }
    )

    return SearchState(**{**asdict(state), "report": report})
    # return report

# ── assemble the graph ──────────────────────────────────────────────────────────
graph = StateGraph(SearchState)

graph.add_node("preprocess_query",     preprocess_query)
graph.add_node("generate_query",       generate_query)
graph.add_node("run_search",           run_search)
graph.add_node("parse_results",        parse_results)
graph.add_node("summarize_initial",    summarize_initial)
graph.add_node("generate_follow_up",   generate_follow_up)
graph.add_node("run_follow_up_search", run_follow_up_search)
graph.add_node("parse_follow_up",      parse_follow_up)
graph.add_node("summarize_follow_up",  summarize_follow_up)
graph.add_node("generate_report",      generate_report)

# edges – strictly linear in this example
graph.set_entry_point("preprocess_query")
graph.add_edge("preprocess_query",    "generate_query")
graph.add_edge("generate_query",      "run_search")
graph.add_edge("run_search",          "parse_results")
graph.add_edge("parse_results",       "summarize_initial")
graph.add_edge("summarize_initial",   "generate_follow_up")
graph.add_edge("generate_follow_up",  "run_follow_up_search")
graph.add_edge("run_follow_up_search", "parse_follow_up")
graph.add_edge("parse_follow_up",     "summarize_follow_up")
graph.add_edge("summarize_follow_up", "generate_report")
graph.set_finish_point("generate_report")

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

# # Usage:
# result_state = executor.invoke({"original_query": "What are the health benefits of green tea?"})
# print(result_state)

deeptox_agent = Agent(
    name="Deep Toxicology Agent",
    model=OpenAIChat(id="gpt-4.1-2025-04-14"),
    tools=[get_chemprop, invoke_deepsearch],
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
        
        When provided with a JSON report, create a detailed markdown report that follows this structure:

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

        The report must be scientifically rigorous, evidence-based, and strictly adhere to the provided schema structure.
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    session_id="demo",
    storage=SqliteStorage(table_name="agent_sessions", db_file="tmp/data.db"),
    add_history_to_messages=False, #history might not be needed
    stream_intermediate_steps=False,
    stream=False,
    debug_mode=True,
)
    
if __name__ == "__main__":
    deeptox_agent.print_response(
        """Based on the following chemical properties of gentamicin: 
        What are the nephrotoxic effects and risks? Include: 
        1. How these specific chemical properties may contribute to nephrotoxicity 
        2. Known mechanisms of nephrotoxicity based on these properties 
        3. Clinical evidence and case studies
        4. Beta distribution of the gentamicin having nephrotoxicity on a 
        healthy individuals at regular dose and its probability with confidence interval.
        5. Risk factors and populations at risk""") 