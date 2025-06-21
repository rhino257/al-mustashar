import logging
import uuid
import json
import traceback 
import dataclasses 
import copy 
import re # Import regex module
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from langgraph.graph.graph import CompiledGraph
from typing import AsyncGenerator, Dict, Any, Optional, List

from ..models.api_models import RagQueryRequest, SSEMetadata, SSEStreamInitiated, SSEMessageUpdate, SSEMessageFinalized, ChatIdInfo, ErrorDetails, SourceDocument
from langchain_core.documents import Document as LangchainDocument
from ..core.auth import get_current_user
from ..core.rate_limiter import user_rate_limiter # Import the rate limiter
from ...retrieval_graph.state import AgentState
from gotrue.types import User 
from ..models.api_models import NewAgentChatRequest
from ...retrieval_graph.graph import graph as compiled_graph_app 
import os
import httpx
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag", 
    tags=["RAG Agent"],
    dependencies=[Depends(get_current_user), Depends(user_rate_limiter)] 
)

async def save_ai_message_to_supabase_async(
    chat_id: str,
    user_id: str,
    ai_message_text: str,
    actual_ai_message_id: str 
):
    supabase_url = os.getenv("USERS_SUPABASE_URL") 
    supabase_key_env = os.getenv("USERS_SUPABASE_KEY") 
    edge_function_name = "add-assistant-message" 

    if supabase_key_env:
        logger.info(f"[SaveMessage] Loaded USERS_SUPABASE_KEY. Starts with: '{supabase_key_env[:5]}', ends with: '{supabase_key_env[-5:]}'. Length: {len(supabase_key_env)}")
        supabase_key = supabase_key_env.strip() 
    else:
        logger.error("[SaveMessage] CRITICAL: USERS_SUPABASE_KEY environment variable is NOT SET or is empty!")
        return 

    if not all([supabase_url, supabase_key, edge_function_name]): 
        logger.error("Supabase config missing for saving AI message (USERS_SUPABASE_URL or USERS_SUPABASE_KEY (after strip) or edge_function_name).")
        return

    payload_to_save = {
        "chat_id": chat_id,
        "message_text": ai_message_text,
        "user_id": user_id,
        "ai_message_id": actual_ai_message_id 
    }
    headers = {
        "Authorization": f"Bearer {supabase_key.strip()}", 
        "Content-Type": "application/json"
    }
    deno_function_url = f"{supabase_url.rstrip('/')}/functions/v1/{edge_function_name}" 

    try:
        async with httpx.AsyncClient() as client: 
            response = await client.post(deno_function_url, json=payload_to_save, headers=headers, timeout=10.0)
            if response.status_code == 201 or response.status_code == 200: 
                logger.info(f"AI Message saved to Supabase: chat_id={chat_id}, ai_message_id={actual_ai_message_id}")
            else:
                logger.warning(f"Failed to save AI message to Supabase, status: {response.status_code}, response: {response.text}")
    except Exception as e:
        logger.error(f"Exception during Supabase AI message save: {e}", exc_info=True)

def parse_and_match_citations(
    full_answer: str, 
    potential_sources: List[LangchainDocument]
) -> List[SourceDocument]:
    cited_source_documents: List[SourceDocument] = []
    # Regex to find citations like "(المادة 123 من قانون كذا)" or "(مادة ٤٥ من القانون المدني)"
    # It captures article number (Arabic or Western numerals) and law name.
    article_citation_pattern = re.compile(r"\(المادة\s+([\d\u0660-\u0669]+)\s+من\s+(?:قانون\s+)?(.+?)\)")
    
    # Normalize Arabic numerals to Western numerals for easier comparison if needed
    def normalize_arabic_numerals(num_str):
        mapping = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        return num_str.translate(mapping)

    found_citations = article_citation_pattern.findall(full_answer)
    
    # Keep track of added document IDs to avoid duplicates if cited multiple times
    added_doc_ids = set()

    for raw_article_num, law_name_cited in found_citations:
        try:
            article_num_cited = normalize_arabic_numerals(raw_article_num.strip())
            law_name_cited_normalized = law_name_cited.strip().replace("ال", "").replace(" ", "") # Basic normalization

            for doc in potential_sources:
                doc_metadata = doc.metadata or {}
                doc_id = str(doc_metadata.get("article_id", doc_metadata.get("id", uuid.uuid4()))) # Prefer article_id

                if doc_id in added_doc_ids:
                    continue

                item_type = doc_metadata.get("item_type", "unknown")
                
                if item_type == "article":
                    doc_article_num = str(doc_metadata.get("article_number", "")).strip()
                    doc_law_name = str(doc_metadata.get("law_name", "")).strip()
                    doc_law_name_normalized = doc_law_name.replace("ال", "").replace(" ", "")

                    if doc_article_num == article_num_cited and \
                       (law_name_cited_normalized in doc_law_name_normalized or \
                        doc_law_name_normalized in law_name_cited_normalized): # Flexible matching for law name
                        
                        cited_source_documents.append(SourceDocument(
                            id=doc_id,
                            content=doc.page_content,
                            metadata=doc_metadata
                        ))
                        added_doc_ids.add(doc_id)
                        logger.info(f"Matched cited article: Law='{doc_law_name}', Article='{doc_article_num}' to a source document.")
                        logger.info(f"Retrieved document content: {doc.page_content}")
                        break # Move to next citation once matched
        except Exception as e:
            logger.error(f"Error processing citation ('{raw_article_num}', '{law_name_cited}'): {e}", exc_info=True)

    # TODO: Add logic for parsing and matching comment citations if a pattern emerges
    # For now, if no article citations are found, or to supplement, we could consider
    # a simpler heuristic: if the answer is short and few documents were provided,
    # assume all were relevant. Or, if the LLM mentions specific comment titles/authors.
    # This part needs further refinement based on how comments are cited.

    if not found_citations and not cited_source_documents and len(potential_sources) <= 3:
         logger.info(f"No explicit article citations found in answer, and few potential sources ({len(potential_sources)}). Assuming all are relevant.")
         for doc in potential_sources:
            doc_id = str(doc.metadata.get("article_id", doc.metadata.get("id", uuid.uuid4())))
            if doc_id not in added_doc_ids:
                 cited_source_documents.append(SourceDocument(
                    id=doc_id,
                    content=doc.page_content,
                    metadata=doc.metadata
                ))
                 added_doc_ids.add(doc_id)
    
    logger.info(f"Final list of {len(cited_source_documents)} cited/used source documents prepared for SSE.")
    return cited_source_documents


async def _rag_response_stream(
    llm_stream: Optional[AsyncGenerator[Any, None]],
    synthesized_answer: Optional[str],
    chat_id: str,
    user_id: str,
    ai_message_id_for_sse: str,
    # source_documents: Optional[List[SourceDocument]] = None # This will now be derived
    final_documents_for_synthesis: Optional[List[LangchainDocument]] = None # All docs given to LLM
) -> AsyncGenerator[str, None]:
    persistent_ai_message_id = str(uuid.uuid4()) 
    full_answer_accumulated = ""
    logger.info(f"ENTERING [_rag_response_stream] for chat_id: {chat_id}, SSE_ID: {ai_message_id_for_sse}")
    # logger.info(f"[_rag_response_stream] Received llm_stream: {type(llm_stream)}, synthesized_answer: '{str(synthesized_answer)[:100]}' (type: {type(synthesized_answer)})") # Commented out

    try:
        metadata_event_data = {"chat_id": chat_id, "ai_message_id": ai_message_id_for_sse, "file_processing_errors": []}
        yield f"event: metadata\ndata: {json.dumps(metadata_event_data)}\n\n"
        # logger.info(f"[_rag_response_stream] YIELDED metadata.") # Commented out

        initiated_data = {"message_id": ai_message_id_for_sse, "status": "processing", "isFinal": False}
        yield f"event: stream_initiated\ndata: {json.dumps(initiated_data)}\n\n"
        # logger.info(f"[_rag_response_stream] YIELDED stream_initiated.") # Commented out

        stream_processed_flag = False
        if llm_stream:
            # logger.info(f"[_rag_response_stream] Attempting to process llm_stream.") # Commented out
            chunk_count = 0
            async for chunk_obj in llm_stream: 
                chunk_count += 1
                # logger.info(f"[_rag_response_stream] Received chunk_obj #{chunk_count}: type={type(chunk_obj)}, content='{str(chunk_obj)[:100]}'") # Commented out 
                
                text_chunk = ""
                if isinstance(chunk_obj, str):
                    text_chunk = chunk_obj
                elif hasattr(chunk_obj, 'content'): 
                    content_attr = getattr(chunk_obj, 'content')
                    if isinstance(content_attr, str):
                        text_chunk = content_attr
                
                # logger.info(f"[_rag_response_stream] Extracted text_chunk: '{text_chunk}'") # Commented out

                if text_chunk: 
                    full_answer_accumulated += text_chunk
                    update_data = {
                        "message_id": ai_message_id_for_sse, "delta": text_chunk,
                        "cumulative_text": full_answer_accumulated, "status": "streaming", "isFinal": False,
                    }
                    yield f"event: message_update\ndata: {json.dumps(update_data)}\n\n"
                    # logger.info(f"[_rag_response_stream] YIELDED message_update with delta: '{text_chunk[:50]}'") # Commented out
                    stream_processed_flag = True
                # else: # Commented out
                    # logger.info(f"[_rag_response_stream] text_chunk is empty or None for chunk_obj #{chunk_count}.") # Commented out
                await asyncio.sleep(0.01) 
            
            if chunk_count == 0:
                logger.warning(f"[_rag_response_stream] llm_stream was provided but yielded no chunks.")
            # else: # Commented out
                # logger.info(f"[_rag_response_stream] Finished iterating llm_stream. Total chunks: {chunk_count}.") # Commented out

        if not stream_processed_flag and synthesized_answer:
            # logger.info(f"[_rag_response_stream] llm_stream did not yield data, or was None. Using synthesized_answer: '{synthesized_answer[:100]}'") # Commented out
            full_answer_accumulated = synthesized_answer 
            update_data = {
                "message_id": ai_message_id_for_sse, "delta": synthesized_answer, 
                "cumulative_text": synthesized_answer, "status": "streaming", "isFinal": False,
            }
            yield f"event: message_update\ndata: {json.dumps(update_data)}\n\n"
            logger.info(f"[_rag_response_stream] YIELDED message_update with full synthesized_answer.")
            stream_processed_flag = True 
        elif not stream_processed_flag: 
            logger.warning(f"[_rag_response_stream] No content from llm_stream and synthesized_answer is empty/None. full_answer_accumulated: '{full_answer_accumulated}'")
            if not full_answer_accumulated:
                full_answer_accumulated = "أ抱歉، لم أتمكن من إنشاء رد." 
                logger.info(f"[_rag_response_stream] Setting default error message.")
        
        # Log the final synthesized answer before citation parsing
        logger.info(f"""
--- Final Synthesized Answer ---
{full_answer_accumulated}
--------------------------------
""")

        # Parse citations and determine actual sources used
        actually_cited_sources: List[SourceDocument] = []
        if full_answer_accumulated and final_documents_for_synthesis:
            logger.info(f"Full answer for citation parsing: {full_answer_accumulated}")
            logger.info(f"Potential sources for citation parsing: {[doc.metadata for doc in final_documents_for_synthesis]}")
            actually_cited_sources = parse_and_match_citations(full_answer_accumulated, final_documents_for_synthesis)
        
        final_data_payload: Dict[str, Any] = {
            "message_id": ai_message_id_for_sse,
            "persistent_ai_message_id": persistent_ai_message_id, 
            "full_content": full_answer_accumulated,
            "status": "complete",
            "isFinal": True,
            "chat_id_info": {"chat_id": chat_id, "is_new_chat": False}, 
            "metadata": {}
        }
        if actually_cited_sources: # Use the filtered list
            final_data_payload["metadata"]["sources"] = [s.dict(by_alias=True, exclude_none=True) for s in actually_cited_sources]
        
        yield f"event: message_finalized\ndata: {json.dumps(final_data_payload)}\n\n"
        logger.info(f"[_rag_response_stream] YIELDED message_finalized. Full content sent: '{full_answer_accumulated[:100]}'. Sources sent: {len(actually_cited_sources)}")

        if full_answer_accumulated and full_answer_accumulated != "أ抱歉، لم أتمكن من إنشاء رد.":
            asyncio.create_task(
                save_ai_message_to_supabase_async(
                    chat_id=chat_id, user_id=user_id,
                    ai_message_text=full_answer_accumulated,
                    actual_ai_message_id=persistent_ai_message_id
                )
            )
            logger.info(f"[_rag_response_stream] Task created to save AI message for persistent_id: {persistent_ai_message_id}")
        else:
            logger.info(f"[_rag_response_stream] No content or default error in full_answer_accumulated, skipping save for persistent_id: {persistent_ai_message_id}")

    except Exception as e_stream: 
        logger.error(f"Error in _rag_response_stream for chat_id {chat_id}: {e_stream}", exc_info=True)
        if 'persistent_ai_message_id' not in locals():
            persistent_ai_message_id = str(uuid.uuid4()) 

        error_final_data: Dict[str, Any] = {
            "message_id": ai_message_id_for_sse, 
            "persistent_ai_message_id": persistent_ai_message_id,
            "status": "error", "isFinal": True,
            "error_details": {"error": "Stream processing error", "details": str(e_stream),
                              "user_facing_message": "Sorry, an error occurred while generating your response."},
            "full_content": full_answer_accumulated, 
            "chat_id_info": {"chat_id": chat_id, "is_new_chat": False }
        }
        yield f"event: message_finalized\ndata: {json.dumps(error_final_data)}\n\n"
        logger.info(f"[_rag_response_stream] YIELDED message_finalized (error).")
    finally:
        logger.info(f"EXITING [_rag_response_stream] for chat_id: {chat_id}, SSE_ID: {ai_message_id_for_sse}")


@router.get("/query", response_class=StreamingResponse) 
async def query_rag_agent_get( 
    request: Request, 
    query: str = Query(..., description="The user's input query.", max_length=4096),
    chat_id_query: str = Query(..., alias="chat_id", description="The ID of the chat session.", max_length=128), 
    pipeline_name: Optional[str] = Query("default", description="Name of the RAG pipeline/agent configuration to use.", max_length=128),
    ai_message_id_query: Optional[str] = Query(None, alias="ai_message_id", description="Client-provided AI message ID.", max_length=128),
    use_reranker: bool = Query(False, description="Whether to use a reranker."),
    agent_persona_query: Optional[str] = Query("almustashar", alias="agent_persona", description="The desired agent persona.", max_length=128), # Added agent_persona
    current_user: User = Depends(get_current_user)
):
    logger.info(f"GET /rag/query received - Query length: {len(query)}, Chat ID length: {len(chat_id_query)}, "
                f"Pipeline name length: {len(pipeline_name) if pipeline_name else 'N/A'}, "
                f"AI Message ID length: {len(ai_message_id_query) if ai_message_id_query else 'N/A'}, "
                f"Agent Persona length: {len(agent_persona_query) if agent_persona_query else 'N/A'}")

    almustashar_agent: CompiledGraph = request.app.state.almustashar_agent
    if not almustashar_agent:
        logger.error("Almustashar agent not initialized in app state for GET /query.")
        raise HTTPException(status_code=503, detail="AI service is currently unavailable.")

    user_id = current_user.id
    user_name = getattr(current_user, 'user_metadata', {}).get('full_name') or \
                getattr(current_user, 'user_metadata', {}).get('name')
    phone_number = getattr(current_user, 'phone', None)
    logger.info(f"Extracted user details: user_name='{user_name}', phone_number='{phone_number}'")

    current_chat_id = chat_id_query 
    ai_message_id_for_sse = ai_message_id_query or str(uuid.uuid4().hex) 

    logger.info(f"Received GET /rag/query for chat_id: {current_chat_id} from user: {user_id} with query: {query}")

    from langchain_core.messages import HumanMessage 
    
    agent_input_dict: Dict[str, Any] = {
        "user_query": query,
        "chat_id": current_chat_id,
        "user_id": user_id,
        "user_name": user_name,
        "phone_number": phone_number,
        "messages": [HumanMessage(content=query)], 
        "use_reranker": use_reranker,
        "router": {'type': 'general', 'logic': ''}, 
        "steps": [],
        "documents": [],
        "query_embedding": None,
        "query_analysis_result": None,
        "query_classification": None,
        "text_for_embedding": None,
        "identified_law_name": None,
        "identified_article_number": None,
        "search_keywords": [],
        "query_intent": None,
        "error_message": None,
        "error_node": None,
        "retrieved_documents": [],
        "reranked_documents": [],
        "preliminary_answer": None,
        "final_documents_for_synthesis": [],
        "final_answer": None,
        "llm_output_stream": None, 
        "conversational_response": None,
        "direct_lookup_mcp_args": None,
        "direct_lookup_mcp_response": None,
    }

    # Prepare base configurable dictionary
    configurable_dict: Dict[str, Any] = {
        "thread_id": current_chat_id,
        "user_id": user_id,
        "agent_persona": agent_persona_query,
        # Default values from AgentConfiguration can be overridden here if needed per persona
        # For example, if "الذكي" should always use a specific threshold for its Matryoshka RPC
    }

    # Conditionally set embedding_model and other persona-specific parameters
    if agent_persona_query == "المستشار":
        configurable_dict["embedding_model"] = "matryoshka_arabic/default_768dim"
        configurable_dict["hybrid_search_threshold"] = 0.5 # Threshold for Matryoshka semantic part in its hybrid RPC
        configurable_dict["DEFAULT_RRF_K_VAL"] = 60 # RRF K for its hybrid RPC
        logger.info(f"Configuring 'المستشار' persona with HYBRID retrieval strategy.")
    elif agent_persona_query == "الذكي":
        configurable_dict["embedding_model"] = "matryoshka_arabic/default_768dim"
        configurable_dict["hybrid_search_threshold"] = 0.5 # Threshold for Matryoshka semantic part in its hybrid RPC
        configurable_dict["DEFAULT_RRF_K_VAL"] = 60 # RRF K for its hybrid RPC
        logger.info(f"Configuring 'الذكي' persona with Matryoshka embeddings and specific thresholds.")
    elif agent_persona_query == "الغبي":
        configurable_dict["embedding_model"] = "matryoshka_arabic/default_768dim" # Ensure الغبي uses Matryoshka
        # Set a default semantic threshold for الغبي's Matryoshka semantic search if needed, e.g.,
        # configurable_dict["hybrid_search_threshold"] = 0.5 # Or a new specific config field
        logger.info(f"Configuring 'الغبي' persona with Matryoshka embeddings.")
    # Add other personas as needed

    config = {"configurable": configurable_dict}
    logger.info(f"Invoking RAG agent with config: {config}")
    
    final_state_dict: Optional[Dict[str, Any]] = None
    try:
        # log_input = {k: (f"[{len(v)} messages]" if k == "messages" and v else v) for k,v in agent_input_dict.items()} # Commented out
        # logger.info(f"Invoking RAG agent (GET /query) with ainvoke. Input (sanitized): {log_input}, Config: {config}") # Commented out
        final_state_dict = await almustashar_agent.ainvoke(agent_input_dict, config=config)
        # logger.info(f"RAG agent ainvoke (GET /query) completed. Keys: {list(final_state_dict.keys()) if final_state_dict else 'None'}") # Commented out

    except Exception as e_ainvoke:
        logger.error(f"Error during almustashar_agent.ainvoke (GET /query): {e_ainvoke}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e_ainvoke)}")

    if not final_state_dict: 
        logger.error(f"LangGraph ainvoke returned None for GET /query, chat_id {current_chat_id}.")
        raise HTTPException(status_code=500, detail="Agent did not return a final state.")

    llm_stream_from_state = final_state_dict.get("llm_output_stream")
    synthesized_answer_from_state = final_state_dict.get("final_answer") or \
                                    final_state_dict.get("conversational_response")
    
    # Pass the full list of documents that were sent to synthesis
    docs_for_synthesis_raw = final_state_dict.get("final_documents_for_synthesis", [])
    
    sse_generator = _rag_response_stream(
        llm_stream=llm_stream_from_state,
        synthesized_answer=synthesized_answer_from_state,
        chat_id=current_chat_id, 
        user_id=user_id, 
        ai_message_id_for_sse=ai_message_id_for_sse, 
        final_documents_for_synthesis=docs_for_synthesis_raw # Pass the full list here
    )
    
    return StreamingResponse(sse_generator, media_type="text/event-stream")

@router.post("/chat-query", response_class=StreamingResponse)
async def new_agent_chat_query(
    request: Request, 
    request_data: NewAgentChatRequest,
    current_user: User = Depends(get_current_user) 
):
    logger.info(f"POST /rag/chat-query received - Query length: {len(request_data.query)}, "
                f"Chat ID length: {len(request_data.chat_id) if request_data.chat_id else 'N/A'}, "
                f"Agent Persona length: {len(request_data.agent_persona) if request_data.agent_persona else 'N/A'}")

    graph_app_to_use: CompiledGraph = request.app.state.almustashar_agent

    if not graph_app_to_use:
        logger.error("Almustashar agent (compiled_graph_app) not initialized for /chat-query.")
        raise HTTPException(status_code=503, detail="AI service is currently unavailable.")

    user_id = current_user.id 
    user_name = getattr(current_user, 'user_metadata', {}).get('full_name') or \
                getattr(current_user, 'user_metadata', {}).get('name')
    phone_number = getattr(current_user, 'phone', None)
    logger.info(f"Extracted user details for POST: user_name='{user_name}', phone_number='{phone_number}'")

    current_chat_id = request_data.chat_id or str(uuid.uuid4().hex)
    ai_message_id_for_sse = str(uuid.uuid4().hex) 

    from langchain_core.messages import HumanMessage 
    
    initial_graph_input: Dict[str, Any] = {
        "user_query": request_data.query,
        "chat_id": current_chat_id,
        "user_id": user_id,
        "user_name": user_name,
        "phone_number": phone_number,
        "messages": [HumanMessage(content=request_data.query)],
        # agent_persona will be passed via config
        # Initialize other state keys as in the GET endpoint for consistency
        "use_reranker": request_data.use_reranker if hasattr(request_data, 'use_reranker') else False, # Assuming use_reranker might come from request
        "router": {'type': 'general', 'logic': ''}, 
        "steps": [],
        "documents": [],
        "query_embedding": None,
        "query_analysis_result": None,
        "query_classification": None,
        "text_for_embedding": None,
        "identified_law_name": None,
        "identified_article_number": None,
        "search_keywords": [],
        "query_intent": None,
        "error_message": None,
        "error_node": None,
        "retrieved_documents": [],
        "reranked_documents": [],
        "preliminary_answer": None,
        "final_documents_for_synthesis": [],
        "final_answer": None,
        "llm_output_stream": None, 
        "conversational_response": None,
        "direct_lookup_mcp_args": None,
        "direct_lookup_mcp_response": None,
    }
    
    # Prepare base configurable dictionary
    configurable_dict_post: Dict[str, Any] = {
        "thread_id": current_chat_id,
        "user_id": user_id,
        "agent_persona": request_data.agent_persona
    }

    # Conditionally set embedding_model and other persona-specific parameters
    if request_data.agent_persona == "المستشار":
        configurable_dict_post["embedding_model"] = "matryoshka_arabic/default_768dim"
        configurable_dict_post["hybrid_search_threshold"] = 0.5 # Threshold for Matryoshka semantic part in its hybrid RPC
        configurable_dict_post["DEFAULT_RRF_K_VAL"] = 60 # RRF K for its hybrid RPC
        logger.info(f"Configuring 'المستشار' persona for POST with HYBRID retrieval strategy.")
    elif request_data.agent_persona == "الذكي":
        configurable_dict_post["embedding_model"] = "matryoshka_arabic/default_768dim"
        configurable_dict_post["hybrid_search_threshold"] = 0.5 # Threshold for Matryoshka semantic part in its hybrid RPC
        configurable_dict_post["DEFAULT_RRF_K_VAL"] = 60 # RRF K for its hybrid RPC
        logger.info(f"Configuring 'الذكي' persona for POST with Matryoshka embeddings and specific thresholds.")
    elif request_data.agent_persona == "الغبي":
        configurable_dict_post["embedding_model"] = "matryoshka_arabic/default_768dim" # Ensure الغبي uses Matryoshka
        # configurable_dict_post["hybrid_search_threshold"] = 0.5 # Or a new specific config field
        logger.info(f"Configuring 'الغبي' persona for POST with Matryoshka embeddings.")
    # Add other personas as needed

    config_for_ainvoke = {"configurable": configurable_dict_post}
    logger.info(f"Invoking RAG agent with config: {config_for_ainvoke}")

    final_state_dict: Optional[Dict[str, Any]] = None
    try:
        # log_input = {k: v for k, v in initial_graph_input.items() if k != "messages" or not v} # Commented out
        # if "messages" in initial_graph_input and initial_graph_input["messages"]: # Commented out
            # log_input["messages"] = f"[{len(initial_graph_input['messages'])} messages]" # Commented out

        # logger.info(f"Invoking LangGraph agent with ainvoke. Input (sanitized): {log_input}, Config: {config_for_ainvoke}") # Commented out
        
        final_state_dict = await graph_app_to_use.ainvoke(initial_graph_input, config=config_for_ainvoke)
        
        # logger.info(f"LangGraph ainvoke successful. Final state keys: {list(final_state_dict.keys()) if final_state_dict else 'None'}") # Commented out

    except Exception as e:
        logger.error(f"Error during LangGraph ainvoke for chat_id {current_chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing your request with the agent: {str(e)}")

    if not final_state_dict:
        logger.error(f"LangGraph ainvoke returned None for chat_id {current_chat_id}.")
        raise HTTPException(status_code=500, detail="Agent did not return a final state.")

    llm_stream_from_state = final_state_dict.get("llm_output_stream")
    
    synthesized_answer_from_state = final_state_dict.get("final_answer") or \
                                    final_state_dict.get("conversational_response")

    # Pass the full list of documents that were sent to synthesis
    docs_for_synthesis_raw = final_state_dict.get("final_documents_for_synthesis", [])
    
    # if llm_stream_from_state: # Commented out
        # logger.info("llm_output_stream successfully retrieved from graph final state.") # Commented out
    # elif synthesized_answer_from_state: # Commented out
        # logger.info("No llm_output_stream, but synthesized_answer found in graph final state.") # Commented out
    # else: # Commented out
        # logger.warning("Neither llm_output_stream nor a synthesized answer found in graph final state for /chat-query.") # Commented out

    sse_generator = _rag_response_stream( 
        llm_stream=llm_stream_from_state,
        synthesized_answer=synthesized_answer_from_state,
        chat_id=current_chat_id,
        user_id=user_id,
        ai_message_id_for_sse=ai_message_id_for_sse,
        final_documents_for_synthesis=docs_for_synthesis_raw # Pass the full list here
    )
    
    logger.info(f"Returning StreamingResponse for /chat-query, chat_id: {current_chat_id}")
    return StreamingResponse(sse_generator, media_type="text/event-stream")
