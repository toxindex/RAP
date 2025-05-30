"""
Celery tasks for the DeepTox agent.
"""
from typing import Dict, Any
from src.celery_app import celery_app
from src.deeptox import deeptox_agent

@celery_app.task
async def analyze_toxicity(query: str) -> Dict[str, Any]:
    """
    Analyze toxicity using the DeepTox agent.
    
    Args:
        query: Natural language query about chemical toxicity
    
    Returns:
        Dict containing the analysis results
    """
    try:
        result = await deeptox_agent.arun(query)
        
        return {
            'status': 'success',
            'result': str(result),
            'task_id': analyze_toxicity.request.id
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'task_id': analyze_toxicity.request.id
        }

@celery_app.task
async def get_chemical_properties(chemical_name: str) -> Dict[str, Any]:
    """
    Get chemical properties using the DeepTox agent.
    
    Args:
        chemical_name: Name of the chemical to analyze
    
    Returns:
        Dict containing the chemical properties
    """
    try:
        result = await deeptox_agent.arun(chemical_name)
        
        return {
            'status': 'success',
            'chemical_name': chemical_name,
            'properties': str(result),
            'task_id': get_chemical_properties.request.id
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'chemical_name': chemical_name,
            'error': str(e),
            'task_id': get_chemical_properties.request.id
        } 