"""
Script to fetch laws with low classification confidence (or needing review)
and a sample of their articles for further review or processing by an advanced LLM.
"""
import asyncio
import os
import json
import random
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from supabase import create_client, Client as SupabaseClient
from tqdm.asyncio import tqdm as async_tqdm

# --- Pydantic Models for Structured Output ---
class ArticleSample(BaseModel):
    article_id: str
    article_number: Optional[str] = None
    article_text: Optional[str] = None

class LawDataForReview(BaseModel):
    law_id: str
    law_name: Optional[str] = None
    description: Optional[str] = None
    current_suggested_classification: Optional[List[str]] = Field(default_factory=list)
    current_classification_confidence: Optional[float] = None
    current_classification_reasoning: Optional[str] = None
    sample_articles: List[ArticleSample] = Field(default_factory=list)

# --- Configuration ---
# Load environment variables from .env file in the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(project_root, '.env'))

KNOWLEDGE_SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL")
KNOWLEDGE_SUPABASE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY")

CONFIDENCE_THRESHOLD = 0.9
ARTICLE_SAMPLE_SIZE = 20
OUTPUT_FILENAME = "data_for_advanced_llm_review.json"

# --- Database Interaction Functions ---
async def fetch_laws_for_review(db_client: SupabaseClient) -> List[Dict[str, Any]]: # Removed confidence_threshold argument
    target_model_name = "gemini-2.5-preview"
    print(f"Fetching laws where classification_llm_model is NOT '{target_model_name}' or is NULL...")
    try:
        response = await asyncio.to_thread(
            db_client.table("law")
            .select("law_id, law_name, description, suggested_classification, classification_confidence, classification_reasoning, classification_llm_model")
            .or_(
                f"classification_llm_model.not.eq.{target_model_name},"
                "classification_llm_model.is.null"
            )
            .execute
        )
        if response.data:
            print(f"  Fetched {len(response.data)} laws for review.")
        else:
            print("  No laws found matching review criteria.")
        return response.data or []
    except Exception as e:
        print(f"  [DB ERROR] Error fetching laws for review: {e}") # Keep this error message generic
        return []

async def fetch_random_articles_for_law(db_client: SupabaseClient, law_id: str, sample_size: int) -> List[Dict[str, Any]]:
    print(f"  Fetching up to {sample_size} random articles for law_id: {law_id}...")
    try:
        # Using ORDER BY random() can be inefficient on large tables, but is simplest for sampling.
        # For very large article sets per law, a multi-step approach might be better.
        response = await asyncio.to_thread(
            db_client.table("law_articles")
            .select("article_id, article_number, article_text")
            .eq("law_id", law_id)
            .order("article_id", desc=False) # Order by article_id instead of non-existent id
            .limit(sample_size * 5) # Fetch more than needed to allow for random sampling if random() is not available or performant
            .execute
        )
        
        if response.data:
            # If random() is not directly supported or to ensure client-side randomness from a larger pool:
            if len(response.data) > sample_size:
                sampled_articles = random.sample(response.data, sample_size)
            else:
                sampled_articles = response.data
            print(f"    Fetched {len(sampled_articles)} sample articles for law_id: {law_id}.")
            return sampled_articles
        else:
            print(f"    No articles found for law_id: {law_id}.")
            return []
    except Exception as e:
        print(f"  [DB ERROR] Error fetching articles for law_id {law_id}: {e}")
        return []

# --- Main Logic ---
async def main():
    if not (KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY):
        print("Supabase URL or Key not found in .env file. Exiting.")
        return

    try:
        supabase_client: SupabaseClient = create_client(KNOWLEDGE_SUPABASE_URL, KNOWLEDGE_SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}. Exiting.")
        return

    laws_to_review_data = await fetch_laws_for_review(supabase_client) # Removed CONFIDENCE_THRESHOLD
    
    all_data_for_llm: List[LawDataForReview] = []

    if not laws_to_review_data:
        print("No laws met the criteria for review. Exiting.")
        return

    for law_data in async_tqdm(laws_to_review_data, desc="Processing laws"):
        law_id = law_data.get("law_id")
        if not law_id:
            print(f"  Skipping law due to missing law_id: {law_data.get('law_name')}")
            continue

        sample_articles_data = await fetch_random_articles_for_law(supabase_client, law_id, ARTICLE_SAMPLE_SIZE)
        
        article_samples = [
            ArticleSample(
                article_id=art.get("article_id"),
                article_number=str(art.get("article_number")) if art.get("article_number") is not None else None, # Convert to string
                article_text=art.get("article_text")
            ) for art in sample_articles_data
        ]

        law_review_entry = LawDataForReview(
            law_id=law_id,
            law_name=law_data.get("law_name"),
            description=law_data.get("description"),
            current_suggested_classification=law_data.get("suggested_classification"),
            current_classification_confidence=law_data.get("classification_confidence"),
            current_classification_reasoning=law_data.get("classification_reasoning"),
            sample_articles=article_samples
        )
        all_data_for_llm.append(law_review_entry)

    # Determine Desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop_path):
        print(f"Warning: Desktop path '{desktop_path}' not found. Saving to current directory instead.")
        desktop_path = "." # Fallback to current directory

    output_file_path = os.path.join(desktop_path, OUTPUT_FILENAME)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(all_data_for_llm):
                # Convert each LawDataForReview object to a JSON string
                json_string = entry.model_dump_json(indent=4, exclude_none=True)
                f.write(json_string)
                f.write("\n") # Add a newline after the JSON object
                
                # Add separator after each entry, except for the last one
                if i < len(all_data_for_llm) - 1:
                    f.write("____________\n\n")
                    
        print(f"Successfully wrote data for {len(all_data_for_llm)} laws to: {output_file_path}")
    except IOError as e:
        print(f"Error writing output file to {output_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")

if __name__ == "__main__":
    asyncio.run(main())
