import os
import sys
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any, Optional # Added Optional
from tqdm.asyncio import tqdm_asyncio # For async iteration
# from tqdm import tqdm # Keep for synchronous parts if any future need

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

try:
    from custom_providers.matryoshka_arabic_embeddings import MatryoshkaArabicEmbeddings
    # normalize_arabic_text_for_embedding is used within MatryoshkaArabicEmbeddings if normalize_text=True
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
TABLE_NAME = "law_articles"
TEXT_COLUMN = "processed_text"
ID_COLUMN = "article_id"
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

async def fetch_total_articles_to_process() -> int:
    """Fetches the total count of articles that need embedding."""
    try:
        # Supabase-py uses postgrest-py, which returns a CountResponse for .count()
        response = (
            supabase.table(TABLE_NAME)
            .select(ID_COLUMN, count="exact") # Request exact count
            .is_(NEW_EMBEDDING_COLUMN, "null")
            .execute()
        )
        return response.count if response.count is not None else 0
    except Exception as e:
        print(f"Error fetching total count from Supabase: {e}")
        return 0

async def fetch_batch_to_process(offset: int) -> List[Dict[str, Any]]:
    """Fetches a batch of articles that do not yet have the new embedding."""
    try:
        response = (
            supabase.table(TABLE_NAME)
            .select(f"{ID_COLUMN}, {TEXT_COLUMN}")
            .is_(NEW_EMBEDDING_COLUMN, "null")
            .limit(BATCH_SIZE)
            .offset(offset) # Offset is needed if we are not using a cursor/keyset pagination
            .execute()
        )
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching batch from Supabase: {e}")
        return []

async def update_embeddings_for_batch(updates: List[Dict[str, Any]], pbar: Optional[tqdm_asyncio] = None) -> int:
    """Updates a batch of articles with their new embeddings."""
    updated_count = 0
    try:
        # Using tqdm_asyncio.gather for concurrent updates if supabase client supports true async operations for update
        # However, supabase-py's update is synchronous under the hood for each call.
        # So, we'll iterate and update, but can wrap the iteration in tqdm for progress on updates.
        
        # If pbar is for the outer loop (processing batches), we might not need a nested pbar here,
        # or we can use a simple print. For now, just printing success/failure per batch.
        
        # Let's process updates one by one to manage errors better for individual rows
        for update_item in updates: # Can wrap this with tqdm(updates, desc="Updating batch") if desired
            embedding_list = [float(x) for x in update_item[NEW_EMBEDDING_COLUMN]]
            try:
                (supabase.table(TABLE_NAME)
                 .update({NEW_EMBEDDING_COLUMN: embedding_list})
                 .eq(ID_COLUMN, update_item[ID_COLUMN])
                 .execute())
                updated_count += 1
            except Exception as e_item:
                print(f"Error updating article {update_item[ID_COLUMN]}: {e_item}")
        
        if updated_count > 0:
            # This print might be too verbose if tqdm is used effectively in the main loop
            # print(f"Successfully updated {updated_count} articles in this batch.")
            pass

    except Exception as e:
        print(f"General error during batch update in Supabase: {e}")
    return updated_count


async def main():
    print("Starting script to populate Matryoshka embeddings...")
    
    total_to_process = await fetch_total_articles_to_process()
    if total_to_process == 0:
        print("No articles found needing embedding, or error fetching count. Exiting.")
        return

    print(f"Total articles to process: {total_to_process}")
    
    processed_successfully = 0
    offset = 0 # offset for fetching batches

    # Initialize tqdm progress bar for the main loop over batches
    # The number of iterations will be total_to_process / BATCH_SIZE, rounded up.
    # Or, we can update it dynamically. Let's use total_to_process as the total for items.
    with tqdm_asyncio(total=total_to_process, unit="article", desc="Processing articles") as pbar:
        while processed_successfully < total_to_process:
            # print(f"\nFetching batch of articles (offset {offset})...") # tqdm will show progress
            articles_batch = await fetch_batch_to_process(offset) # We still need offset for fetching

            if not articles_batch:
                if offset == 0 and processed_successfully == 0 : # No articles found at all on first try
                     pbar.set_description("No articles found to process at start.")
                else: # No more articles found after some processing
                     pbar.set_description("No more articles found.")
                break
            
            # pbar.set_postfix_str(f"Fetched {len(articles_batch)} for current batch")

            texts_to_embed = []
            valid_articles_in_batch = [] # Keep track of articles that have text

            for article in articles_batch:
                text_content = article.get(TEXT_COLUMN)
                if text_content and isinstance(text_content, str) and text_content.strip():
                    texts_to_embed.append(text_content)
                    valid_articles_in_batch.append(article) # Store the whole article dict
                else:
                    print(f"Article {article.get(ID_COLUMN)} has no valid text in '{TEXT_COLUMN}'. Skipping.")
                    # This article won't be processed, so it won't contribute to pbar.update()
                    # If we want the main pbar to reflect skipped items, we'd need to adjust total or pbar.update here.
                    # For now, pbar reflects successfully processed items.

            if not texts_to_embed:
                # pbar.set_postfix_str("No valid texts in batch.")
                if len(articles_batch) < BATCH_SIZE: # Reached the end
                    break
                offset += len(articles_batch) # Advance offset if entire batch was bad, to avoid loop
                continue

            # pbar.set_postfix_str(f"Embedding {len(texts_to_embed)} texts...")
            try:
                embeddings = embedder.embed_documents(texts_to_embed)
            except Exception as e:
                print(f"Error generating embeddings for batch: {e}")
                offset += len(articles_batch) # Advance offset to skip this problematic batch
                continue
            
            updates_for_supabase = []
            for i, article_dict in enumerate(valid_articles_in_batch):
                if i < len(embeddings):
                    updates_for_supabase.append({
                        ID_COLUMN: article_dict[ID_COLUMN],
                        NEW_EMBEDDING_COLUMN: embeddings[i]
                    })
                else: # Should not happen if lists are managed correctly
                    print(f"Error: Mismatch in text/embedding list for article {article_dict[ID_COLUMN]}")

            if updates_for_supabase:
                # pbar.set_postfix_str(f"Updating {len(updates_for_supabase)} in DB...")
                num_updated_in_batch = await update_embeddings_for_batch(updates_for_supabase)
                processed_successfully += num_updated_in_batch
                pbar.update(num_updated_in_batch)
            
            # If not all articles in the fetched batch were processed (e.g. due to missing text),
            # the offset logic needs to be robust.
            # Since we query for `is_(NEW_EMBEDDING_COLUMN, "null")`, successfully processed ones are excluded.
            # The `offset` is mainly to ensure we don't re-fetch the *same initial set* of unprocessed items if some fail non-fatally.
            # However, if fetch_batch_to_process always gets the "next" available NULLs, offset might not be strictly needed
            # or should be managed differently. For now, keeping it simple:
            # If a batch is fully processed, the next fetch will get new NULLs.
            # If a batch has errors and items remain NULL, they might be fetched again if offset doesn't advance past them.
            # The current fetch_batch_without_embedding uses offset, so we must increment it if we want to skip.
            # But since we are filtering by NULL, we don't strictly need to increment offset.
            # Let's remove offset increment, as the NULL check should handle fetching new items.
            # offset += len(articles_batch) # This line is removed.

            if len(articles_batch) < BATCH_SIZE: # Reached the end of available data
                pbar.set_description("Processed last batch.")
                break
    
    pbar.close()
    print(f"\nScript finished. Total articles successfully processed and updated: {processed_successfully}")

if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        # Re-check env vars after loading .env
        SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL", SUPABASE_URL)
        SUPABASE_SERVICE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY", SUPABASE_SERVICE_KEY)
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set in .env or environment.")
            sys.exit(1)
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) # Re-initialize with potentially new values
        print(f"Supabase client re-initialized with URL: {SUPABASE_URL[:30]}...")
    elif not (SUPABASE_URL and SUPABASE_SERVICE_KEY): # If not loaded from .env and also not pre-set
        print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set via .env or environment variables.")
        sys.exit(1)
    
    asyncio.run(main())
