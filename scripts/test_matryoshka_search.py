import os
import sys
import io 
import asyncio
import time 
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import numpy as np 

# Reconfigure stdout to use UTF-8 if not already
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        print("Successfully reconfigured sys.stdout to UTF-8.")
    except Exception as e:
        print(f"Warning: Could not reconfigure sys.stdout to UTF-8: {e}. Output might have encoding issues.")

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

try:
    from custom_providers.matryoshka_arabic_embeddings import MatryoshkaArabicEmbeddings
    from custom_providers.bge_m3_law_embeddings import BgeM3LawEmbeddings 
except ImportError as e:
    print(f"Error importing embedding providers: {e}")
    sys.exit(1)

dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env file from: {dotenv_path}")
else:
    load_dotenv() 
    if os.path.exists(".env"):
        print("Loaded .env file from current working directory.")
    else:
        print("Warning: .env file not found. Relying on environment variables if set.")

SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Error: KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set.")
    sys.exit(1)

# --- Test Parameters ---
QUERIES = [
    "كم مدة الطعن في الاستئناف",
    "ما هو تعريف الوصية في القانون اليمني؟",
    "ما هي إجراءات رفع دعوى إخلاء مأجور في اليمن؟",
    "ما هي شروط الحضانة للأم بعد الطلاق في القانون اليمني؟",
    "ما هي التزامات المستأجر بموجب عقد الإيجار اليمني؟",
    "هل يوجد مفهوم للتعويض عن الضرر المعنوي في القانون المدني اليمني؟",
    "ما هي أركان وعقوبة جريمة خيانة الأمانة في اليمن؟",
    "كيف يتم إشهار إفلاس التاجر في القانون التجاري اليمني؟",
    "كم هي مدة الإجازة السنوية للعامل في قانون العمل اليمني؟",
    "إذا توفي شخص وترك ديوناً ولم يترك تركة، هل يلزم الورثة بسدادها؟",
    "ما الفرق بين البطلان والفسخ في العقود حسب القانون اليمني؟"
]

# Parameters for individual Hybrid RPCs
RPC_SEMANTIC_THRESHOLD_MATRYOSHKA = 0.5
RPC_SEMANTIC_THRESHOLD_BGE = 0.25
RPC_RRF_K_VAL = 60
RPC_FINAL_MATCH_COUNT = 5 # How many results each Hybrid RPC should return

# RPC Names
HYBRID_RPC_ARTICLES_MATRYOSHKA = 'hybrid_search_articles_smart_rrf_matryoshka'
HYBRID_RPC_ARTICLES_BGE = 'hybrid_search_articles_smart_rrf_bge'
HYBRID_RPC_COMMENTS_MATRYOSHKA = 'hybrid_search_comments_smart_rrf_matryoshka'
HYBRID_RPC_COMMENTS_BGE = 'hybrid_search_comments_smart_rrf_bge'

# Parameters for Client-Side "Meta-Hybrid" RRF
COUNT_FROM_EACH_HYBRID_RPC_FOR_META_RRF = RPC_FINAL_MATCH_COUNT
META_RRF_K_VAL = 60
# CLIENT_SIDE_RRF_PRE_FILTER_SEMANTIC_THRESHOLD = 0.4 # Replaced by model-specific thresholds
FINAL_TOP_N_FOR_LLM = 5 # How many results to display/consider for each approach


try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print(f"Supabase client initialized for URL: {SUPABASE_URL[:30]}...")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    sys.exit(1)

matryoshka_embedder: Optional[MatryoshkaArabicEmbeddings] = None
bge_m3_law_embedder: Optional[BgeM3LawEmbeddings] = None

try:
    matryoshka_embedder = MatryoshkaArabicEmbeddings(normalize_text=True)
    print(f"MatryoshkaArabicEmbeddings initialized.")
except Exception as e:
    print(f"Error initializing MatryoshkaArabicEmbeddings: {e}")

try:
    bge_m3_law_embedder = BgeM3LawEmbeddings(normalize_text=True) 
    print(f"BgeM3LawEmbeddings initialized.")
except Exception as e:
    print(f"Error initializing BgeM3LawEmbeddings: {e}")


def get_ranked_results_from_hybrid_rpc(
    supabase_client: Client, 
    rpc_name: str, 
    query_text: str, # Keyword query
    query_embedding_vector: List[float], # Semantic query
    semantic_threshold: float, 
    final_match_count: int, 
    rrf_k_value: int
) -> List[Dict[str, Any]]:
    """
    Calls a Hybrid RPC and returns a list of ranked document objects.
    Assumes the RPC returns results already ordered by its internal RRF.
    """
    try:
        if isinstance(query_embedding_vector, np.ndarray):
            embedding_list = query_embedding_vector.tolist()
        else:
            embedding_list = list(query_embedding_vector)

        rpc_params = {
            'p_query_embedding': embedding_list,
            'p_keyword_query': query_text,
            'p_match_threshold': semantic_threshold,
            'p_match_count': final_match_count,
            'p_rrf_k_val': rrf_k_value
        }
        # print(f"Calling RPC {rpc_name} with params: {rpc_params}") # For debugging
        response = supabase_client.rpc(rpc_name, rpc_params).execute()

        if hasattr(response, 'data') and response.data:
            return response.data
        elif hasattr(response, 'error') and response.error:
            print(f"Error from Hybrid RPC {rpc_name}: {response.error}")
            return []
        else:
            # print(f"No results or unexpected response from Hybrid RPC {rpc_name}.")
            return []
    except Exception as e:
        print(f"Error calling Hybrid RPC {rpc_name}: {e}")
        import traceback; traceback.print_exc()
        return []

def client_side_reciprocal_rank_fusion(list_of_ranked_doc_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    fused_scores: Dict[str, float] = {} 
    doc_map: Dict[str, Dict[str, Any]] = {}

    for ranked_doc_list in list_of_ranked_doc_lists:
        if not ranked_doc_list:
            continue
        for rank, doc in enumerate(ranked_doc_list, 1):
            original_semantic_score = doc.get('similarity', 0.0)
            source_model_type = doc.get('source_model_type', 'unknown') # Get the source model type

            # Model-specific Pre-RRF Semantic Filtering
            passed_filter = True
            if source_model_type == 'matryoshka':
                if original_semantic_score < RPC_SEMANTIC_THRESHOLD_MATRYOSHKA:
                    passed_filter = False
            elif source_model_type == 'bge':
                if original_semantic_score < RPC_SEMANTIC_THRESHOLD_BGE:
                    passed_filter = False
            # else: # if unknown source_model_type, pass it through or apply a default? For now, pass.
                # print(f"Debug: Doc ID {doc.get('article_id')} from unknown model type {source_model_type}. Passing pre-filter.")
                
            if not passed_filter:
                # print(f"Debug: Doc ID {doc.get('article_id')} from {doc.get('item_type')}/{source_model_type} with original_semantic_score {original_semantic_score:.4f} below its model's threshold. Skipping for RRF.")
                continue

            doc_id = doc.get('article_id') 
            if not doc_id:
                doc_id = doc.get('comment_id') 
                if not doc_id:
                    print(f"Warning: Document found without 'article_id' or 'comment_id' (and passed pre-filter): {doc}")
                    continue
            
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            elif not doc_map[doc_id].get('processed_text') and doc.get('processed_text'): # Prefer doc with text
                doc_map[doc_id] = doc
            if 'item_type' not in doc_map[doc_id]: # Ensure item_type is set
                 doc_map[doc_id]['item_type'] = doc.get('item_type', 'unknown')


            score_contribution = 1.0 / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score_contribution
            
    reranked_results_with_scores = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    
    final_fused_docs = []
    for doc_id, score in reranked_results_with_scores:
        doc_detail = doc_map.get(doc_id)
        if doc_detail:
            doc_detail_copy = doc_detail.copy() 
            doc_detail_copy['meta_rrf_score'] = score # New score from this client-side RRF
            final_fused_docs.append(doc_detail_copy)
            
    return final_fused_docs

def print_individual_hybrid_rpc_results(results: List[Dict[str, Any]], model_description: str, item_type: str, query: str, threshold: float, rpc_match_count: int, rrf_k: int):
    print(f"\n\n======================================================================")
    print(f"RESULTS FOR QUERY: \"{query}\" FROM HYBRID RPC ({model_description} Semantic + Keyword) - Target: {item_type.upper()}S")
    print(f"(RPC Params: Sem_Th: {threshold}, MatchCount: {rpc_match_count}, RRF_K: {rrf_k})")
    print(f"======================================================================")
    if results:
        print(f"Retrieved {len(results)} {item_type}s (RRF'd by RPC):")
        for i, item in enumerate(results[:rpc_match_count], 1): 
            print(f"\n  --- Result {i} ({model_description} Hybrid RPC for {item_type}s) ---")
            item_id = item.get('article_id') # Assuming 'article_id' is PK for comments too
            print(f"    {item_type.capitalize()} ID: {item_id}")
            if item_type == 'article':
                print(f"    Law Name: {item.get('law_name')}")
                print(f"    Article Number: {item.get('article_number')}")
            elif item_type == 'comment':
                print(f"    Comment Title: {item.get('title', 'N/A')}")
                print(f"    Author: {item.get('author', 'N/A')}")

            sem_sim_val = item.get('similarity') 
            rrf_score_val = item.get('rrf_score')
            sem_sim_str = f"{sem_sim_val:.4f}" if isinstance(sem_sim_val, float) else "N/A"
            rrf_score_str = f"{rrf_score_val:.4f}" if isinstance(rrf_score_val, float) else "N/A"
            print(f"    Original Semantic Similarity: {sem_sim_str}")
            print(f"    RPC RRF Score: {rrf_score_str}")
            processed_text_content = item.get('processed_text', '')
            print(f"    Processed Text: {processed_text_content}") # Display full processed_text
    else:
        print(f"No results from Supabase Hybrid RPC for {model_description} on {item_type}s.")

def print_meta_fused_results(results: List[Dict[str, Any]], query: str, top_n: int):
    print(f"\n\n======================================================================")
    print(f"META-HYBRID RRF RESULTS FOR QUERY: \"{query}\" (Top {top_n})")
    print(f"(Fused from All Hybrid RPC outputs: Articles & Comments)")
    print(f"======================================================================")
    if results:
        print(f"Displaying top {min(len(results), top_n)} Meta-Hybrid fused items:")
        for i, item in enumerate(results[:top_n], 1):
            item_type = item.get('item_type', 'unknown')
            item_id = item.get('article_id') # Assuming 'article_id' is PK for comments too
            print(f"\n  --- Meta-Fused Result {i} (Type: {item_type.capitalize()}) ---")
            print(f"    ID: {item_id}")
            if item_type == 'article':
                print(f"    Law Name: {item.get('law_name')}")
                print(f"    Article Number: {item.get('article_number')}")
            elif item_type == 'comment':
                print(f"    Comment Title: {item.get('title', 'N/A')}")
                print(f"    Author: {item.get('author', 'N/A')}")
            
            meta_rrf_score = item.get('meta_rrf_score', 'N/A')
            meta_rrf_score_str = f"{meta_rrf_score:.4f}" if isinstance(meta_rrf_score, float) else "N/A"
            print(f"    Meta RRF Score: {meta_rrf_score_str}")
            processed_text_content = item.get('processed_text', '')
            print(f"    Processed Text: {processed_text_content}") # Display full processed_text
    else:
        print(f"No results after Meta-Hybrid RRF fusion for this query.")


async def run_all_tests(test_query: str):
    print(f"\n\n######################################################################")
    print(f"PROCESSING QUERY: \"{test_query}\"")
    print(f"######################################################################")

    results_articles_mat_hybrid_rpc: List[Dict[str, Any]] = []
    results_articles_bge_hybrid_rpc: List[Dict[str, Any]] = []
    results_comments_mat_hybrid_rpc: List[Dict[str, Any]] = []
    results_comments_bge_hybrid_rpc: List[Dict[str, Any]] = []

    print("\n--- TESTING COMMENTS TABLE ONLY ---")

    # --- COMMENTS: Matryoshka Hybrid RPC ---
    if matryoshka_embedder:
        mat_query_embedding = matryoshka_embedder.embed_query(test_query)
        results_comments_mat_hybrid_rpc = get_ranked_results_from_hybrid_rpc(
            supabase, HYBRID_RPC_COMMENTS_MATRYOSHKA,
            test_query, mat_query_embedding,
            RPC_SEMANTIC_THRESHOLD_MATRYOSHKA, RPC_FINAL_MATCH_COUNT, RPC_RRF_K_VAL
        )
        for r in results_comments_mat_hybrid_rpc: 
            r['item_type'] = 'comment'
            r['source_model_type'] = 'matryoshka' # Tag with source model
        print_individual_hybrid_rpc_results(results_comments_mat_hybrid_rpc, "Matryoshka", "comment", test_query,
                                            RPC_SEMANTIC_THRESHOLD_MATRYOSHKA, RPC_FINAL_MATCH_COUNT, RPC_RRF_K_VAL)

    # --- COMMENTS: BGE-M3-Law Hybrid RPC ---
    if bge_m3_law_embedder:
        bge_query_embedding = bge_m3_law_embedder.embed_query(test_query) # Re-use embedding or re-generate
        results_comments_bge_hybrid_rpc = get_ranked_results_from_hybrid_rpc(
            supabase, HYBRID_RPC_COMMENTS_BGE,
            test_query, bge_query_embedding,
            RPC_SEMANTIC_THRESHOLD_BGE, RPC_FINAL_MATCH_COUNT, RPC_RRF_K_VAL
        )
        for r in results_comments_bge_hybrid_rpc: 
            r['item_type'] = 'comment'
            r['source_model_type'] = 'bge' # Tag with source model
        print_individual_hybrid_rpc_results(results_comments_bge_hybrid_rpc, "BGE-M3-Law", "comment", test_query,
                                           RPC_SEMANTIC_THRESHOLD_BGE, RPC_FINAL_MATCH_COUNT, RPC_RRF_K_VAL)

    # --- Scenario 3: Meta-Hybrid (Client-Side RRF of COMMENT Hybrid RPC outputs) ---
    meta_hybrid_candidate_lists = []
    # Only include comment results for fusion
    if results_comments_mat_hybrid_rpc:
        meta_hybrid_candidate_lists.append(results_comments_mat_hybrid_rpc[:COUNT_FROM_EACH_HYBRID_RPC_FOR_META_RRF])
    if results_comments_bge_hybrid_rpc:
        meta_hybrid_candidate_lists.append(results_comments_bge_hybrid_rpc[:COUNT_FROM_EACH_HYBRID_RPC_FOR_META_RRF])
        
    fused_meta_hybrid_docs: List[Dict[str, Any]] = []
    if not meta_hybrid_candidate_lists:
        print("\nNo results from any Hybrid RPC to fuse for Meta-Hybrid. Skipping.")
    else:
        print("\n--- Performing Client-Side RRF of Hybrid RPC Outputs (Meta-Hybrid) ---")
        fused_meta_hybrid_docs = client_side_reciprocal_rank_fusion(
            meta_hybrid_candidate_lists, k=META_RRF_K_VAL
        )
        print(f"Meta-Hybrid RRF performed. Total unique docs after meta-fusion: {len(fused_meta_hybrid_docs)}")
        if fused_meta_hybrid_docs:
             print(f"Top few Meta-Hybrid RRF scores: {[ (doc.get('article_id'), doc.get('meta_rrf_score')) for doc in fused_meta_hybrid_docs[:5]] }")
    
    print_meta_fused_results(fused_meta_hybrid_docs, test_query, FINAL_TOP_N_FOR_LLM)


async def main_test_loop():
    if not matryoshka_embedder and not bge_m3_law_embedder:
        print("Neither Matryoshka nor BGE-M3-Law embedders initialized. Exiting.")
        return

    original_stdout = sys.stdout
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_file_path = os.path.join(desktop_path, "hybrid_search_test_results.txt")

    try:
        print(f"Redirecting output to: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            sys.stdout = f
            for query in QUERIES: # Using QUERIES from your instructions
                await run_all_tests(query) # Changed function name
                print("\n-------------------------------------------------------\n")
            print(f"\n\nOutput successfully written to {output_file_path}")
    except Exception as e:
        # If redirection fails, print error to original stdout
        sys.stdout = original_stdout
        print(f"Error during script execution or output redirection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout # Restore stdout
        print(f"Script finished. Results saved to {output_file_path} (if successful).")


if __name__ == "__main__":
    print("Starting Meta-Hybrid RPC comparison script...")
    # Ensure embedders are loaded before asyncio.run
    if not matryoshka_embedder: print("Matryoshka embedder failed to initialize. Check logs.")
    if not bge_m3_law_embedder: print("BGE-M3-Law embedder failed to initialize. Check logs.")
    
    asyncio.run(main_test_loop()) # Changed function name
    print("\nMeta-Hybrid RPC Comparison script finished.")
