"""
Script to update law classifications in Supabase from a reviewed JSON file.
The JSON file is expected to contain individual JSON objects separated by '____________'.
"""
import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from supabase import create_client, Client as SupabaseClient
from tqdm import tqdm # Using synchronous tqdm for the main loop

# --- Pydantic Model for Input Validation ---
class RefinedLawData(BaseModel):
    law_id: str
    revised_suggested_classification: Optional[List[str]] = Field(default_factory=list)
    revised_sharia_influence: Optional[bool] = None
    revised_classification_confidence: Optional[float] = None
    revised_classification_reasoning: Optional[str] = None
    # classification_llm_model from LLM output is optional, will be overridden
    classification_llm_model: Optional[str] = None 
    # classification_needs_review from LLM output is optional, will be overridden
    classification_needs_review: Optional[bool] = None


# --- Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(project_root, '.env'))

KNOWLEDGE_SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL")
KNOWLEDGE_SUPABASE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY")

# INPUT_FILE_PATH = r"C:\Users\MT\Desktop\reviewed.txt" 
# Use a relative path or ensure this path is correct for the execution environment
# For flexibility, let's make it configurable or default to a path within the project
INPUT_FILE_PATH = os.getenv("REVIEWED_CLASSIFICATIONS_FILE_PATH", os.path.join(os.path.expanduser("~"), "Desktop", "reviewed.txt"))
NEW_LLM_MODEL_NAME = "gemini-2.5-preview"

# --- Helper Functions ---
def load_and_parse_reviewed_data(file_path: str) -> List[RefinedLawData]:
    """
    Loads data from a text file containing one or more concatenated JSON objects,
    potentially separated by whitespace or newlines.
    Parses and validates each JSON object.
    """
    print(f"Loading and parsing reviewed data from: {file_path}")
    valid_law_data_list: List[RefinedLawData] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        decoder = json.JSONDecoder()
        idx = 0
        obj_count = 0
        while idx < len(content):
            # Find the beginning of the next JSON object
            try:
                # Attempt to decode a JSON object from the current position
                # after stripping leading whitespace.
                obj, end_idx = decoder.raw_decode(content[idx:])
                # Advance idx by the number of characters consumed by raw_decode from content[idx:]
                idx += end_idx 
                obj_count += 1
                try:
                    validated_data = RefinedLawData(**obj)
                    valid_law_data_list.append(validated_data)
                except PydanticValidationError as e:
                    print(f"Warning: Pydantic validation error for object {obj_count} (law_id: {obj.get('law_id', 'UNKNOWN')}). Error: {e}.")
            except json.JSONDecodeError as e:
                # This error means no valid JSON object started at content[idx:]
                # It might be trailing whitespace, or a malformed object.
                # If there's non-whitespace left, it's an issue. Otherwise, we might be done.
                remaining_content = content[idx:].strip()
                if remaining_content:
                    # Calculate global character position for better error reporting
                    # The string passed to raw_decode was content[idx:]
                    # e.pos is relative to the start of that slice.
                    actual_error_start_in_content = idx 
                    context_start = actual_error_start_in_content + (e.pos - 30 if e.pos > 30 else 0)
                    context_end = actual_error_start_in_content + e.pos + 30
                    
                    print(f"Warning: Could not decode JSON for object {obj_count+1} (attempted parse starting near global char {idx}). Specific error: {e}.") # obj_count not incremented yet on error
                    print(f"         Context around error (char {e.pos} in segment): '{content[context_start : context_end]}'")
                    
                    # Attempt to find the next '{' to recover, or break if none.
                    # Search from after the point where the error occurred within the current attempt.
                    search_from_idx = idx + e.pos + 1 
                    next_brace_pos = content.find('{', search_from_idx)
                    if next_brace_pos == -1:
                        print("  No further JSON objects found after decode error.")
                        break 
                    idx = next_brace_pos # Move to the next potential start
                    print(f"  Attempting to recover by jumping to next '{{' at char {idx}")
                else:
                    # Only whitespace remaining, we are done.
                    break
        
        print(f"Attempted to parse {obj_count} JSON structures. Successfully validated and parsed {len(valid_law_data_list)} law entries.")
        return valid_law_data_list
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading or parsing the file: {e}")
        return []

async def update_law_in_db(db_client: SupabaseClient, law_data: RefinedLawData) -> bool:
    """
    Updates a single law record in the Supabase database.
    """
    print(f"  Updating law_id: {law_data.law_id}...")
    try:
        update_payload = {
            "suggested_classification": law_data.revised_suggested_classification,
            "suggested_sharia_influence": law_data.revised_sharia_influence,
            "classification_confidence": law_data.revised_classification_confidence,
            "classification_reasoning": law_data.revised_classification_reasoning,
            "classification_llm_model": NEW_LLM_MODEL_NAME,
            "classification_needs_review": False, # Trusting the advanced LLM's review
            "classification_processed_at": datetime.now(timezone.utc).isoformat()
        }

        # Remove keys with None values to avoid overwriting DB fields with NULL
        # if the advanced LLM didn't provide them (though our Pydantic model has defaults)
        update_payload = {k: v for k, v in update_payload.items() if v is not None}
        
        if not update_payload:
            print(f"    Skipping update for law_id {law_data.law_id} as payload is empty after None removal.")
            return False

        await asyncio.to_thread(
            db_client.table("law")
            .update(update_payload)
            .eq("law_id", law_data.law_id)
            .execute
        )
        print(f"    Successfully updated law_id: {law_data.law_id}")
        return True
    except Exception as e:
        print(f"    [DB UPDATE ERROR] Failed to update law_id {law_data.law_id}: {e}")
        return False

# --- Main Logic ---
async def main():
    if not (KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY):
        print("Supabase URL or Key not found in .env file. Exiting.")
        return

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Input file '{INPUT_FILE_PATH}' not found. Please check the path. You can also set REVIEWED_CLASSIFICATIONS_FILE_PATH environment variable.")
        return
        
    try:
        supabase_client: SupabaseClient = create_client(KNOWLEDGE_SUPABASE_URL, KNOWLEDGE_SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}. Exiting.")
        return

    refined_laws = load_and_parse_reviewed_data(INPUT_FILE_PATH)

    if not refined_laws:
        print("No valid law data to process from the input file. Exiting.")
        return

    successful_updates = 0
    failed_updates = 0

    # Unsafe mode is not typically managed this way for direct Supabase client operations.
    # The client uses a service key which usually has necessary permissions.
    # Removing the placeholder MCP call.
    print("Proceeding with database updates...")

    for law_data in tqdm(refined_laws, desc="Updating law classifications"):
        success = await update_law_in_db(supabase_client, law_data)
        if success:
            successful_updates += 1
        else:
            failed_updates += 1
    
    print("\nDatabase updates processing complete.")

    print(f"\n--- Update Summary ---")
    print(f"Total laws processed from file: {len(refined_laws)}")
    print(f"Successfully updated: {successful_updates}")
    print(f"Failed updates: {failed_updates}")

if __name__ == "__main__":
    asyncio.run(main())
