"""
Pydantic models for structuring the output of LLM-assisted
classification and tagging of laws and law articles.
These models are intended for use by database processing scripts.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class LawClassificationOutput(BaseModel):
    """
    Represents the LLM's suggested classification for a law.
    """
    law_name: str = Field(description="The original law name provided to the LLM.")
    suggested_categories: List[str] = Field(
        default_factory=list,
        description="A list of suggested standard categories from the predefined taxonomy (e.g., ['مدني', 'تجاري'])."
    )
    sharia_influence: bool = Field(
        default=False, # Provide a non-None default
        description="Indicates if the law is significantly derived from or implements Sharia principles."
    )
    confidence_score: float = Field(
        default=0.0,  # Provide a non-None default
        description="LLM's confidence in its suggestions (0.0-1.0)."
    )
    reasoning: str = Field(
        default="",   # Provide a non-None default
        description="Brief reasoning for the suggested categories and Sharia influence assessment."
    )
    error_message: str = Field(
        default="", # Provide a non-None default
        description="Any error message if the LLM failed to process this law."
    )

class ArticleTaggingOutput(BaseModel):
    """
    Represents the LLM's suggested tags for a law article.
    """
    article_identifier: str = Field(description="Identifier for the article (e.g., law_name + article_number).")
    suggested_tags: List[str] = Field(
        default_factory=list,
        description="A list of suggested tags from a predefined vocabulary or newly identified relevant tags (e.g., ['مدة الطعن', 'استئناف'])."
    )
    confidence_score: float = Field(
        default=0.0,  # Provide a non-None default
        description="LLM's confidence in its suggestions (0.0-1.0)."
    )
    reasoning: str = Field(
        default="",   # Provide a non-None default
        description="Brief reasoning for the suggested tags."
    )
    error_message: str = Field(
        default="", # Provide a non-None default
        description="Any error message if the LLM failed to process this article."
    )

class CommentTaggingOutput(BaseModel):
    """
    Represents the LLM's suggested tags for a comment.
    """
    comment_identifier: str = Field(description="Identifier for the comment (e.g., comment title or first few words if no title).")
    suggested_tags: List[str] = Field(
        default_factory=list,
        description="A list of suggested tags from a predefined vocabulary or newly identified relevant tags."
    )
    confidence_score: float = Field(
        default=0.0,
        description="LLM's confidence in its suggestions (0.0-1.0)."
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the suggested tags."
    )
    error_message: str = Field(
        default="",
        description="Any error message if the LLM failed to process this comment."
    )
