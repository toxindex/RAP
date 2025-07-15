"""
Generalized schema for chemical toxicity assessment.
This module contains multiple versions of the toxicity assessment schema:
1. TOXICITY_SCHEMA_BASIC: Simple, minimal schema with essential fields
2. TOXICITY_SCHEMA_FLEXIBLE: Advanced schema with confidence levels and optional fields
3. TOXICITY_SCHEMA: Default schema (currently set to FLEXIBLE)
"""

# Basic schema with minimal required fields
TOXICITY_SCHEMA_BASIC = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ChemicalToxicityAssessmentBasic",
    "type": "object",
    "properties": {
        "chemical_toxicity": {
            "type": "object",
            "properties": {
                "chemical_properties": {
                    "type": "object",
                    "properties": {
                        "description": { "type": "string" },
                        "evidence": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["description", "evidence"]
                },
                "toxicity_mechanisms": {
                    "type": "object",
                    "properties": {
                        "description": { "type": "string" },
                        "mechanisms": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["description", "mechanisms"]
                },
                "clinical_evidence": {
                    "type": "object",
                    "properties": {
                        "description": { "type": "string" },
                        "evidence": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["description", "evidence"]
                },
                "toxicity_risk_distribution": {
                    "type": "object",
                    "properties": {
                        "explanation": { "type": "string" },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "alpha": { "type": "number" },
                                "beta": { "type": "number" },
                                "probability": { "type": "number" },
                                "confidence_interval": {
                                    "type": "object",
                                    "properties": {
                                        "lower": { "type": "number" },
                                        "upper": { "type": "number" },
                                        "confidence_level": { "type": "number" }
                                    },
                                    "required": ["lower", "upper", "confidence_level"]
                                }
                            },
                            "required": ["alpha", "beta", "probability", "confidence_interval"]
                        },
                        "interpretation": { "type": "string" },
                        "calculation_references": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": { "type": "string" },
                                    "url": { "type": "string" },
                                    "description": { "type": "string" },
                                    "relevance": { "type": "string" }
                                },
                                "required": ["title", "url", "description", "relevance"]
                            }
                        }
                    },
                    "required": ["explanation", "parameters", "interpretation", "calculation_references"]
                },
                "risk_factors": {
                    "type": "object",
                    "properties": {
                        "high_risk_groups": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "extra_notes": { "type": "string" }
                    },
                    "required": ["high_risk_groups", "extra_notes"]
                },
                "references": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": [
                "chemical_properties",
                "toxicity_mechanisms",
                "clinical_evidence",
                "toxicity_risk_distribution",
                "risk_factors",
                "references"
            ]
        }
    },
    "required": ["chemical_toxicity"]
}

# Flexible schema with confidence levels and optional fields
TOXICITY_SCHEMA_FLEXIBLE = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ChemicalToxicityAssessmentFlexible",
    "type": "object",
    "properties": {
        "chemical_toxicity": {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "data_completeness": {
                            "type": "object",
                            "properties": {
                                "overall_score": { "type": "number", "minimum": 0, "maximum": 1 },
                                "missing_fields": { "type": "array", "items": { "type": "string" } },
                                "confidence_level": { "type": "string", "enum": ["high", "medium", "low"] }
                            },
                            "required": ["overall_score", "confidence_level"]
                        },
                        "last_updated": { "type": "string", "format": "date-time" },
                        "data_sources": { "type": "array", "items": { "type": "string" } },
                        "reference_revision_attempts": { "type": "integer" }
                    },
                    "required": ["data_completeness", "last_updated"]
                },
                "chemical_properties": {
                    "type": "object",
                    "properties": {
                        "description": { "type": "string" },
                        "properties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "value": { "type": "string" },
                                    "relevance": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] }
                                },
                                "required": ["name", "value"]
                            }
                        },
                        "evidence": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["description"]
                },
                "toxicity_mechanisms": {
                    "type": "object",
                    "properties": {
                        "description": { "type": "string" },
                        "mechanisms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "description": { "type": "string" },
                                    "evidence": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] },
                                    "references": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    }
                                },
                                "required": ["name", "description"]
                            }
                        }
                    },
                    "required": ["description"]
                },
                "clinical_evidence": {
                    "type": "object",
                    "properties": {
                        "description": { "type": "string" },
                        "studies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": { "type": "string" },
                                    "type": { "type": "string" },
                                    "sample_size": { "type": "integer" },
                                    "positive_cases": { "type": "integer" },
                                    "duration": { "type": "string" },
                                    "dosage": { "type": "string" },
                                    "demographics": { "type": "string" },
                                    "findings": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] },
                                    "references": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    }
                                },
                                "required": ["title", "findings"]
                            }
                        },
                        "treatment_protocols": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "description": { "type": "string" },
                                    "effectiveness": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] },
                                    "references": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    }
                                },
                                "required": ["name", "description"]
                            }
                        }
                    },
                    "required": ["description"]
                },
                "toxicity_risk_distribution": {
                    "type": "object",
                    "properties": {
                        "explanation": { "type": "string" },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "alpha": { "type": "number" },
                                "beta": { "type": "number" },
                                "probability": { "type": "number" },
                                "confidence_interval": {
                                    "type": "object",
                                    "properties": {
                                        "lower": { "type": "number" },
                                        "upper": { "type": "number" },
                                        "confidence_level": { "type": "number" }
                                    }
                                }
                            }
                        },
                        "studies_used": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": { "type": "string" },
                                    "url": { "type": "string" },
                                    "sample_size": { "type": "integer" },
                                    "positive_cases": { "type": "integer" },
                                    "weight": { "type": "number" },
                                    "contribution": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] }
                                },
                                "required": ["title", "url", "contribution"]
                            }
                        },
                        "interpretation": { "type": "string" },
                        "limitations": { "type": "string" },
                        "confidence": { "type": "string", "enum": ["high", "medium", "low"] }
                    },
                    "required": ["explanation", "interpretation"]
                },
                "risk_factors": {
                    "type": "object",
                    "properties": {
                        "high_risk_groups": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "group": { "type": "string" },
                                    "risk_level": { "type": "string" },
                                    "explanation": { "type": "string" },
                                    "evidence": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] }
                                },
                                "required": ["group", "risk_level"]
                            }
                        },
                        "modifying_factors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "factor": { "type": "string" },
                                    "effect": { "type": "string" },
                                    "evidence": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] }
                                },
                                "required": ["factor"]
                            }
                        },
                        "preventive_measures": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "measure": { "type": "string" },
                                    "effectiveness": { "type": "string" },
                                    "evidence": { "type": "string" },
                                    "confidence": { "type": "string", "enum": ["high", "medium", "low"] }
                                },
                                "required": ["measure"]
                            }
                        },
                        "extra_notes": { "type": "string" }
                    }
                },
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": { "type": "string" },
                            "authors": { "type": "string" },
                            "year": { "type": "integer" },
                            "url": { "type": "string" },
                            "type": { "type": "string" },
                            "relevance": { "type": "string" }
                        },
                        "required": ["title"]
                    }
                }
            },
            "required": [
                "metadata",
                "chemical_properties",
                "toxicity_mechanisms",
                "clinical_evidence",
                "toxicity_risk_distribution",
                "risk_factors"
            ]
        }
    },
    "required": ["chemical_toxicity"]
}

# Set the default schema to FLEXIBLE
TOXICITY_SCHEMA = TOXICITY_SCHEMA_FLEXIBLE 