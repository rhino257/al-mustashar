import os
import sys
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

try:
    from custom_providers.matryoshka_arabic_embeddings import MatryoshkaArabicEmbeddings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure your PYTHONPATH is set correctly or run this script from the project root.")
    sys.exit(1)

load_dotenv()

SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY environment variables must be set.")
    sys.exit(1)

# Configuration
TABLE_NAME = "comments"
TEXT_COLUMN = "processed_text"  # Assuming comments also have a 'processed_text' column
ID_COLUMN = "article_id"        # Primary key of the 'comments' table
NEW_EMBEDDING_COLUMN = "embedding_matryoshka_v2"
BATCH_SIZE = 50
EMBEDDING_DIMENSION = 768

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Initialize Matryoshka Embeddings
try:
    embedder = MatryoshkaArabicEmbeddings(truncate_dim=EMBEDDING_DIMENSION, normalize_text=True)
except Exception as e:
    print(f"Error initializing MatryoshkaArabicEmbeddings: {e}")
    sys.exit(1)

async def fetch_total_items_to_process() -> int:
    """Fetches the total count of items in the table that need embedding."""
    try:
        response = (
            supabase.table(TABLE_NAME)
            .select(ID_COLUMN, count="exact")
            .is_(NEW_EMBEDDING_COLUMN, "null")
            .execute()
        )
        return response.count if response.count is not None else 0
    except Exception as e:
        print(f"Error fetching total count from Supabase for table {TABLE_NAME}: {e}")
        return 0

async def fetch_batch_to_process(offset: int) -> List[Dict[str, Any]]:
    """Fetches a batch of items that do not yet have the new embedding."""
    try:
        response = (
            supabase.table(TABLE_NAME)
            .select(f"{ID_COLUMN}, {TEXT_COLUMN}")
            .is_(NEW_EMBEDDING_COLUMN, "null")
            .limit(BATCH_SIZE)
            .offset(offset)
            .execute()
        )
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching batch from Supabase for table {TABLE_NAME}: {e}")
        return []

async def update_embeddings_for_batch(updates: List[Dict[str, Any]]) -> int:
    """Updates a batch of items with their new embeddings."""
    updated_count = 0
    try:
        for update_item in updates:
            embedding_list = [float(x) for x in update_item[NEW_EMBEDDING_COLUMN]]
            try:
                (supabase.table(TABLE_NAME)
                 .update({NEW_EMBEDDING_COLUMN: embedding_list})
                 .eq(ID_COLUMN, update_item[ID_COLUMN])
                 .execute())
                updated_count += 1
            except Exception as e_item:
                print(f"Error updating item {update_item[ID_COLUMN]} in table {TABLE_NAME}: {e_item}")
        
    except Exception as e:
        print(f"General error during batch update in Supabase for table {TABLE_NAME}: {e}")
    return updated_count


async def main():
    print(f"Starting script to populate Matryoshka embeddings for table '{TABLE_NAME}'...")
    
    total_to_process = await fetch_total_items_to_process()
    if total_to_process == 0:
        print(f"No items found in '{TABLE_NAME}' needing embedding, or error fetching count. Exiting.")
        return

    print(f"Total items in '{TABLE_NAME}' to process: {total_to_process}")
    
    processed_successfully = 0
    offset = 0

    with tqdm_asyncio(total=total_to_process, unit="item", desc=f"Processing {TABLE_NAME}") as pbar:
        while processed_successfully < total_to_process:
            items_batch = await fetch_batch_to_process(offset)

            if not items_batch:
                if offset == 0 and processed_successfully == 0:
                     pbar.set_description(f"No items found in '{TABLE_NAME}' to process at start.")
                else:
                     pbar.set_description(f"No more items found in '{TABLE_NAME}'.")
                break
            
            texts_to_embed = []
            valid_items_in_batch = []

            for item in items_batch:
                text_content = item.get(TEXT_COLUMN)
                if text_content and isinstance(text_content, str) and text_content.strip():
                    texts_to_embed.append(text_content)
                    valid_items_in_batch.append(item)
                else:
                    print(f"Item {item.get(ID_COLUMN)} in '{TABLE_NAME}' has no valid text in '{TEXT_COLUMN}'. Skipping.")

            if not texts_to_embed:
                if len(items_batch) < BATCH_SIZE:
                    break
                offset += len(items_batch) 
                continue

            try:
                embeddings = embedder.embed_documents(texts_to_embed)
            except Exception as e:
                print(f"Error generating embeddings for batch from '{TABLE_NAME}': {e}")
                offset += len(items_batch)
                continue
            
            updates_for_supabase = []
            for i, item_dict in enumerate(valid_items_in_batch):
                if i < len(embeddings):
                    updates_for_supabase.append({
                        ID_COLUMN: item_dict[ID_COLUMN],
                        NEW_EMBEDDING_COLUMN: embeddings[i]
                    })
                else:
                    print(f"Error: Mismatch in text/embedding list for item {item_dict[ID_COLUMN]} in '{TABLE_NAME}'")

            if updates_for_supabase:
                num_updated_in_batch = await update_embeddings_for_batch(updates_for_supabase)
                processed_successfully += num_updated_in_batch
                pbar.update(num_updated_in_batch)
            
            if len(items_batch) < BATCH_SIZE:
                pbar.set_description(f"Processed last batch for '{TABLE_NAME}'.")
                break
    
    pbar.close()
    print(f"\nScript finished for '{TABLE_NAME}'. Total items successfully processed and updated: {processed_successfully}")

if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL", SUPABASE_URL)
        SUPABASE_SERVICE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY", SUPABASE_SERVICE_KEY)
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set in .env or environment.")
            sys.exit(1)
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print(f"Supabase client re-initialized with URL: {SUPABASE_URL[:30]}...")
    elif not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set via .env or environment variables.")
        sys.exit(1)
    
    asyncio.run(main())
