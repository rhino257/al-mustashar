import os
import sys
import asyncio
import time # Added for timing
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

try:
    from custom_providers.bge_m3_law_embeddings import BgeM3LawEmbeddings
except ImportError as e:
    print(f"Error importing BgeM3LawEmbeddings: {e}")
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
NEW_EMBEDDING_COLUMN = "embedding_bge_m3_law" 
BATCH_SIZE = 20 
EMBEDDING_DIMENSION = 1024

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

try:
    embedder = BgeM3LawEmbeddings(normalize_text=True) 
except Exception as e:
    print(f"Error initializing BgeM3LawEmbeddings: {e}")
    sys.exit(1)

async def fetch_total_articles_to_process() -> int:
    print(f"[{time.strftime('%H:%M:%S')}] Attempting to fetch total articles to process...")
    try:
        response = (
            supabase.table(TABLE_NAME)
            .select(ID_COLUMN, count="exact")
            .is_(NEW_EMBEDDING_COLUMN, "null")
            .execute()
        )
        count = response.count if response.count is not None else 0
        print(f"[{time.strftime('%H:%M:%S')}] Fetched total: {count} articles.")
        return count
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error fetching total count for BGE-M3-Law from Supabase: {e}")
        return 0

async def fetch_batch_to_process() -> List[Dict[str, Any]]:
    print(f"[{time.strftime('%H:%M:%S')}] Attempting to fetch batch...")
    try:
        response = (
            supabase.table(TABLE_NAME)
            .select(f"{ID_COLUMN}, {TEXT_COLUMN}")
            .is_(NEW_EMBEDDING_COLUMN, "null")
            .limit(BATCH_SIZE)
            .execute()
        )
        data = response.data if response.data else []
        print(f"[{time.strftime('%H:%M:%S')}] Fetched batch of {len(data)} articles.")
        return data
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error fetching batch for BGE-M3-Law from Supabase: {e}")
        return []

async def update_embeddings_for_batch(updates: List[Dict[str, Any]]) -> int:
    updated_count = 0
    if not updates:
        return 0
    for update_item in updates:
        embedding_list = [float(x) for x in update_item[NEW_EMBEDDING_COLUMN]]
        try:
            (supabase.table(TABLE_NAME)
             .update({NEW_EMBEDDING_COLUMN: embedding_list})
             .eq(ID_COLUMN, update_item[ID_COLUMN])
             .execute())
            updated_count += 1
        except Exception as e_item:
            print(f"Error updating article {update_item[ID_COLUMN]} with BGE-M3-Law embedding: {e_item}")
    return updated_count

async def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting script to populate BGE-M3-Law embeddings (original update logic)...")
    
    total_to_process = await fetch_total_articles_to_process()
    if total_to_process == 0:
        print(f"[{time.strftime('%H:%M:%S')}] No articles found needing BGE-M3-Law embedding, or error fetching count. Exiting.")
        return

    print(f"[{time.strftime('%H:%M:%S')}] Total articles to process with BGE-M3-Law: {total_to_process}")
    
    processed_successfully = 0
    batch_num = 0
    
    print(f"[{time.strftime('%H:%M:%S')}] Initializing tqdm progress bar...")
    with tqdm_asyncio(total=total_to_process, unit="article", desc="Processing BGE-M3-Law") as pbar:
        print(f"[{time.strftime('%H:%M:%S')}] tqdm initialized. Entering main processing loop.")
        while processed_successfully < total_to_process:
            batch_num += 1
            print(f"\n[{time.strftime('%H:%M:%S')}] --- Batch {batch_num} ---")
            
            articles_batch = await fetch_batch_to_process() # This now has internal prints

            if not articles_batch:
                print(f"[{time.strftime('%H:%M:%S')}] No more articles in batch. Ending.")
                pbar.set_description("No more articles found for BGE-M3-Law.")
                break
            
            texts_to_embed = []
            valid_articles_in_batch = []
            # print(f"[{time.strftime('%H:%M:%S')}] Preparing texts for batch {batch_num}...")
            for article in articles_batch:
                text_content = article.get(TEXT_COLUMN)
                if text_content and isinstance(text_content, str) and text_content.strip():
                    texts_to_embed.append(text_content)
                    valid_articles_in_batch.append(article)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Article {article.get(ID_COLUMN)} has no valid text. Skipping.")
            # print(f"[{time.strftime('%H:%M:%S')}] Text preparation complete for batch {batch_num}. {len(texts_to_embed)} valid texts.")


            if not texts_to_embed:
                if len(articles_batch) < BATCH_SIZE: break 
                print(f"[{time.strftime('%H:%M:%S')}] No valid texts in batch {batch_num} to embed. Continuing.")
                continue 

            # print(f"[{time.strftime('%H:%M:%S')}] Generating embeddings for batch {batch_num}...")
            try:
                embeddings = embedder.embed_documents(texts_to_embed)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error generating BGE-M3-Law embeddings for batch {batch_num}: {e}")
                import traceback; traceback.print_exc()
                await asyncio.sleep(1) 
                continue 
            # print(f"[{time.strftime('%H:%M:%S')}] Embeddings generated for batch {batch_num}.")
            
            updates_for_supabase = []
            for i, article_dict in enumerate(valid_articles_in_batch):
                if i < len(embeddings):
                    updates_for_supabase.append({
                        ID_COLUMN: article_dict[ID_COLUMN],
                        NEW_EMBEDDING_COLUMN: embeddings[i]
                    })
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Error: Mismatch in BGE-M3-Law text/embedding list for article {article_dict[ID_COLUMN]}")

            if updates_for_supabase:
                # print(f"[{time.strftime('%H:%M:%S')}] Updating Supabase for batch {batch_num}...")
                num_updated_in_batch = await update_embeddings_for_batch(updates_for_supabase)
                # print(f"[{time.strftime('%H:%M:%S')}] Supabase update complete for batch {batch_num}. Updated {num_updated_in_batch} articles.")
                processed_successfully += num_updated_in_batch
                pbar.update(num_updated_in_batch)
            
            if len(articles_batch) < BATCH_SIZE:
                print(f"[{time.strftime('%H:%M:%S')}] Fetched less than BATCH_SIZE, assuming end of data.")
                pbar.set_description("Processed last batch for BGE-M3-Law.")
                break
    
    pbar.close()
    print(f"\n[{time.strftime('%H:%M:%S')}] BGE-M3-Law embedding script finished. Total articles successfully processed: {processed_successfully}")

if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL", SUPABASE_URL)
        SUPABASE_SERVICE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY", SUPABASE_SERVICE_KEY)
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set.")
            sys.exit(1)
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    elif not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set.")
        sys.exit(1)
    
    asyncio.run(main())
