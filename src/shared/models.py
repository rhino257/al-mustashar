"""Shared Pydantic models and data structures for the AlMustashar project."""

import json
from typing import List, Literal, Optional, Dict, Any # Added Dict, Any
from pydantic import BaseModel, Field

class YemeniLegalQueryAnalysis(BaseModel):
    """
    Pydantic model for the structured output of the Yemeni legal query analysis LLM call.
    """
    classification: Literal[
        "conversational",
        "legal_query_direct_lookup",
        "legal_query_conceptual_search",
        "other",
    ] = Field(..., description="Classification of the user's query.")
    raw_query: str = Field(..., description="The original user query.")
    hypothetical_answer_for_embedding: Optional[str] = Field(
        default=None,
        description="A dense, factual paragraph in ARABIC summarizing core legal principles for conceptual search. Null otherwise.",
    )
    intent: Optional[
        Literal["specific_article_lookup", "conceptual_search", "unknown"]
    ] = Field(
        default=None,
        description="The intent behind a legal query. Null for non-legal or unclassified queries.",
    )
    law_name: Optional[str] = Field(
        default=None, description="The name of the Yemeni law or regulation mentioned, if any."
    )
    article_number: Optional[str] = Field(
        default=None, description="The specific article number mentioned, if any."
    )
    keywords_for_search: List[str] = Field(
        default_factory=list,
        description="Relevant Arabic keywords from the query for keyword-based search.",
    )
    error_message: Optional[str] = Field(
        default=None, 
        description="An error message if parsing or generation of this analysis object itself failed."
    )
    # New fields for enhanced metadata utilization - Re-added June 7, 2025
    identified_law_categories: Optional[List[str]] = Field(
        default_factory=list,
        description="Law categories identified from the user query, matching the predefined taxonomy."
    )
    identified_tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Potential topical tags identified from the user query."
    )
    query_intent_details: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Structured details about the user's specific intent, e.g., seeking definition, conditions, penalties, procedure."
    )
    filter_logic: Optional[Literal['AND', 'OR']] = Field(
        default='AND', 
        description="Logic to apply between different filter types (categories, tags). Default to AND for precision."
    )
    confidence_scores: Optional[Dict[str, float]] = Field(
        default_factory=dict, 
        description="Confidence scores for identified categories, tags, or intent."
    )

    model_config = {
        "json_schema_extra": {
            "description": "Schema for understanding and analyzing a user's query related to Yemeni law."
        }
    }

    def __str__(self):
        """Returns a JSON string representation of the model for safe logging."""
        return json.dumps(self.model_dump(), indent=2, ensure_ascii=False)
