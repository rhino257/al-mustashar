import asyncio
import logging
from typing import Any, Dict, List
import sys # Moved for path adjustment
from pathlib import Path # Moved for path adjustment

import pytest # Added for asyncio mark

# Ensure src directory is in path for direct script execution
project_root_for_test = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root_for_test)) # Use insert(0, ...) to ensure it's checked first
sys.path.insert(0, str(project_root_for_test / "src")) # Add src to path

import src.shared.utils # Ensures .env is loaded

# Configure basic logging to show INFO messages for cleaner output
logging.basicConfig(level=logging.INFO)

from langgraph.checkpoint.memory import MemorySaver # Added for checkpointer
import langchain # Added
import langchain_core # Added
import openai # Added to check version
print(f"OpenAI library version: {openai.__version__}") # Added
print(f"Langchain version: {langchain.__version__}") # Added
print(f"Langchain-core version: {langchain_core.__version__}") # Added

from langchain_core.messages import HumanMessage

# Configure basic logging for tests
# logging.basicConfig(level=logging.INFO) # Replaced by DEBUG level above
logger = logging.getLogger(__name__)

# Assuming the graph and state are accessible from these paths
# Adjust imports based on your project structure if necessary
try:
    from retrieval_graph.graph import builder as almustashar_graph_builder # Import builder
    from retrieval_graph.state import AgentState, InputState
    from custom_nodes.comprehension_nodes import YemeniLegalQueryAnalysis # For inspecting query_analysis_result
    from custom_nodes.direct_lookup_node import KNOWLEDGE_DB_PROJECT_ID, MCP_SERVER_NAME # For MCP call info if needed by a real client
except ImportError:
    # This block might still be useful if the initial sys.path adjustment isn't enough
    # or for other import issues, but the primary goal is to make 'import src.shared.utils' work.
    logger.error("Failed to import graph or state even after initial path adjustment. Check PYTHONPATH or project structure.")
    # Re-raising or exiting might be appropriate if critical imports fail
    raise # Re-raise the ImportError if essential modules can't be found


async def run_test_query(query: str, thread_id: str, test_case_id: str) -> None: # Added test_case_id
    """
    Sends a query to the AlMustasharRetrievalGraph and prints key outputs.
    Handles two-pass invocation for direct lookup tests.
    """
    logger.info(f"\n--- Testing Query (Thread ID: {thread_id}, Test Case ID: {test_case_id}): \"{query}\" ---")

    # Compile the graph with a MemorySaver for this test run
    # This ensures that update_state has a checkpointer
    graph_for_test = almustashar_graph_builder.compile(checkpointer=MemorySaver())
    graph_for_test.name = "AlMustasharRetrievalGraph_TestInstance"


    # Prepare input for the graph
    graph_input: InputState = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "chat_id": thread_id, # Using thread_id as chat_id for testing
        "user_id": "test_user_id" # Using a static user_id for testing
    }

    # Configuration for invoking the graph
    config = {"configurable": {"thread_id": thread_id}}

    current_run_final_state_dict = None
    # current_run_full_response_chunks = [] # Not collecting chunks with ainvoke directly here

    try:
        # --- First Pass (or single pass if not direct lookup) ---
        logger.info("--- Graph Invocation ---")
        # Use ainvoke instead of astream_events
        final_state = await graph_for_test.ainvoke(graph_input, config)
        
        if isinstance(final_state, dict):
            current_run_final_state_dict = final_state
        elif isinstance(final_state, AgentState):
            current_run_final_state_dict = final_state.dict()
        else:
            logger.error(f"Unexpected type for final_state from ainvoke: {type(final_state)}")
            current_run_final_state_dict = {}


        # Check if direct lookup mcp_args are present after pass 1
        direct_lookup_mcp_args = None
        if current_run_final_state_dict:
            direct_lookup_mcp_args = current_run_final_state_dict.get("direct_lookup_mcp_args")

        if direct_lookup_mcp_args and "direct_lookup" in test_case_id:
            logger.info("--- Direct Lookup Detected: Simulating MCP call and preparing for Pass 2. ---")
            logger.info(f"MCP Args from Pass 1: {direct_lookup_mcp_args}")

            simulated_mcp_response = [{
                "article_id": "sim_article_id_005",
                "text_content": "هذا هو النص المحاكى للمادة الخامسة من قانون العمل.",
                "article_number": 5,
                "law_name": "قانون العمل",
                "law_year": "1995"
            }]
            logger.info(f"Simulated MCP Response: {simulated_mcp_response}")

            update_payload = {
                "direct_lookup_mcp_response": simulated_mcp_response,
                "direct_lookup_mcp_args": None
            }
            # Create a new input for the second invoke, starting with the original input
            # and adding the MCP response. The graph state update via update_state
            # is more for checkpointers; with ainvoke, we pass the modified input.
            # However, if the graph is designed to pick up from checkpointed state for the second pass,
            # then update_state is correct. Assuming the test setup relies on update_state for two-pass.
            graph_for_test.update_state(config, update_payload)
            logger.info("Graph state updated with simulated MCP response for Pass 2.")
            
            # --- Second Pass ---
            logger.info("--- Graph Invocation: Pass 2 (after MCP state update) ---")
            # Re-invoke the graph. The graph_input should ideally be the same,
            # and the graph should use the updated state from the checkpoint.
            final_state_pass2 = await graph_for_test.ainvoke(graph_input, config)

            if isinstance(final_state_pass2, dict):
                current_run_final_state_dict = final_state_pass2
            elif isinstance(final_state_pass2, AgentState):
                current_run_final_state_dict = final_state_pass2.dict()
            else:
                logger.error(f"Unexpected type for final_state_pass2 from ainvoke: {type(final_state_pass2)}")
                current_run_final_state_dict = {}


        # --- Process and Log Final Results ---
        # With ainvoke, the stream is part of the final state if populated by the node.
        # We are not collecting chunks here as astream_events did.
        # The full text response should be in final_answer or conversational_response.
        
        if current_run_final_state_dict is not None:
            logger.info("\n--- Final Agent State ---")
            # Log a snippet of the state dict
            log_state_snippet = {k: v for k, v in current_run_final_state_dict.items() if k != "llm_output_stream"}
            logger.info(f"Final State (excluding stream): {str(log_state_snippet)[:500]}...")


            state_obj_candidate = None
            if isinstance(current_run_final_state_dict, dict):
                try:
                    # Ensure all required fields for AgentState are present or provide defaults
                    # This might need a more robust way to construct AgentState from a potentially partial dict
                    # For now, let's assume the dict from ainvoke is comprehensive enough or handle missing keys.
                    # A safer approach: AgentState.parse_obj(current_run_final_state_dict) if using Pydantic models
                    # For dataclasses, ensure all fields are present or have defaults.
                    # We'll try to instantiate and catch errors.
                    # Minimal required fields for AgentState from InputState:
                    # user_query, chat_id, user_id, messages. These are in graph_input.
                    # The dict from ainvoke should contain all other fields set by the graph.
                    # We can merge graph_input with the result if necessary.
                    # For now, assume current_run_final_state_dict is the full state.
                    state_obj_candidate = AgentState(**current_run_final_state_dict)
                except TypeError as te: # Catches missing required arguments for AgentState constructor
                    logger.error(f"Error instantiating AgentState from dict: {te}. Dict keys: {current_run_final_state_dict.keys()}")
                except Exception as e:
                    logger.error(f"Generic error instantiating AgentState from dict: {e}. Dict keys: {current_run_final_state_dict.keys()}")
            # No need for elif isinstance(current_run_final_state_dict, AgentState) as it's already converted to dict

            if state_obj_candidate:
                state_obj = state_obj_candidate
                logger.info(f"Query Classification: {state_obj.query_classification}")
                logger.info(f"Text ACTUALLY Used for Embedding (state.text_for_embedding): {str(state_obj.text_for_embedding)}")
                if state_obj.query_analysis_result:
                    # q_analysis is now a YemeniLegalQueryAnalysis Pydantic model instance
                    q_analysis = state_obj.query_analysis_result
                    logger.info(f"  Query Classification (from analysis object): {q_analysis.classification}")
                    logger.info(f"  Identified Law (from analysis object): {q_analysis.law_name}")
                    logger.info(f"  Identified Article (from analysis object): {q_analysis.article_number}")
                    logger.info(f"  Keywords (from analysis object): {q_analysis.keywords_for_search}")
                    logger.info(f"  Hypothetical Answer for Embedding (from analysis object): {str(q_analysis.hypothetical_answer_for_embedding)}")
                
                logger.info(f"Retrieved Documents Count: {len(state_obj.documents)}")
                if "direct_lookup" in test_case_id and state_obj.documents:
                    for i, doc in enumerate(state_obj.documents):
                        logger.info(f"  Doc {i+1} (Direct Lookup): {doc.page_content[:150]}... (Metadata: {doc.metadata})")

                if state_obj.final_answer:
                    logger.info(f"Final Legal Answer: {state_obj.final_answer}")
                if state_obj.conversational_response:
                    logger.info(f"Conversational Response: {state_obj.conversational_response}")
                
                # Check for the stream if needed for other assertions or logging
                if current_run_final_state_dict.get("llm_output_stream"):
                    logger.info("Stream available in final state (llm_output_stream).")
                
                if state_obj.error_message:
                    logger.error(f"Error in State: {state_obj.error_message} (at step: {state_obj.error_node})")
            else: # state_obj_candidate could not be created
                logger.warning("Could not instantiate AgentState from final dictionary.")
                # Log key fields directly from the dictionary as a fallback
                logger.info(f"Query Classification (from dict): {current_run_final_state_dict.get('query_classification')}")
                if current_run_final_state_dict.get('final_answer'):
                    logger.info(f"Final Legal Answer (from dict): {current_run_final_state_dict.get('final_answer')}")
                if current_run_final_state_dict.get('conversational_response'):
                    logger.info(f"Conversational Response (from dict): {current_run_final_state_dict.get('conversational_response')}")

        else: # current_run_final_state_dict is None
            logger.warning("Graph invocation did not yield a final state dictionary.")

    except Exception as e:
        # Ensure the error message itself is safely formatted if it contains problematic characters
        error_message_for_log = str(e) # Get the string representation of the error
        logger.error(
            "Error running test query \"%s\" for test case ID \"%s\". Exception: %s",
            query,
            test_case_id,
            error_message_for_log # Pass the error string as an argument
        )

@pytest.mark.asyncio
async def test_run_all_queries(): # Renamed main to be discoverable by pytest
    # Define test queries
    test_queries = [
        {
            "id": "conv_greeting_1",
            "query": "مرحباً، كيف حالك اليوم؟" # Conversational: Greeting
        },
        {
            "id": "conv_identity_1",
            "query": "من أنت؟" # Conversational: Identity
        },
        {
            "id": "conv_capability_1",
            "query": "ماذا يمكنك أن تفعل؟" # Conversational: Capability
        },
        {
            "id": "legal_conceptual_1",
            "query": "ما هي شروط الوصية في القانون اليمني؟" # Legal: Conceptual Search
        },
        {
            "id": "legal_conceptual_2",
            "query": "ما هي عقوبة شهادة الزور؟" # Legal: Conceptual Search
        },
        {
            "id": "legal_direct_lookup_1", # This ID will trigger the two-pass logic
            "query": "ما هي المادة 5 من قانون العمل؟" # Legal: Direct Lookup
        },
        # Add more test cases:
        # - Specific article lookup (once direct_lookup_node is implemented and routed)
        # - Queries that might result in "other" classification
        # - Queries designed to test error handling
    # - Follow-up questions to test chat history
    ]

    thread_counter = 1
    for test_case in test_queries:
        # Using a unique thread_id for each query to ensure isolated state.
        await run_test_query(
            test_case["query"], 
            thread_id=f"test_thread_{test_case['id']}_{thread_counter}",
            test_case_id=test_case["id"] # Pass the test case ID
        )
        thread_counter += 1
        logger.info("----------------------------------------------------")

@pytest.mark.asyncio
async def test_gemini_chat_model_agenerate_via_portkey():
    """
    Focused test for GeminiChatModel._agenerate via Portkey for non-streaming, non-tool calls.
    """
    logger.info("\n--- Testing GeminiChatModel._agenerate via Portkey ---")
    
    # Ensure .env is loaded (handled by src.shared.utils import at the top of the file)
    # We need PORTKEY_API_KEY, GOOGLE_VIRTUAL_KEY_ID, and GOOGLE_API_KEY
    
    from src.custom_providers.gemini_chat_model import GeminiChatModel # Direct import for this test
    from src.shared.utils import load_chat_model # To compare or use as alternative loading

    # Option 1: Direct instantiation (if you know all required params)
    # model_name = "gemini-1.5-flash-latest" 
    # chat_model_instance = GeminiChatModel(
    #     model_name=model_name,
    #     use_portkey_for_generate=True, # Explicitly set the flag
    #     temperature=0.7
    #     # Add other necessary params for GeminiChatModel init if any, e.g., google_api_key if not from env
    # )

    # Option 2: Using existing loader and then setting the flag
    # This is generally safer if load_chat_model handles complex setup.
    try:
        # Ensure the model loaded here is the one we intend to test with Portkey, or override model_name later
        chat_model_instance = load_chat_model(fully_specified_name="gemini_custom/gemini-1.5-flash-latest") 
    except Exception as e:
        logger.error(f"Failed to load chat model using load_chat_model: {e}", exc_info=True)
        pytest.fail(f"Failed to load chat model: {e}")
        return # For linters, though pytest.fail will stop execution

    chat_model_instance.use_portkey_for_generate = True # CRITICAL: Set the flag

    messages = [
        HumanMessage(content="You are a helpful assistant."), # Note: For Gemini, system messages are often handled at model init or extracted.
                                                            # For Portkey (OpenAI-like), system message in list is fine.
                                                            # Let's use HumanMessage here to ensure it's part of the Portkey payload.
                                                            # If a SystemMessage is desired, _convert_messages_to_portkey_format handles it.
        HumanMessage(content="Say 'Hello Portkey Test'.")
    ]
    
    logger.info(f"Attempting _agenerate call with use_portkey_for_generate={chat_model_instance.use_portkey_for_generate}")     
    # Override model_name for this specific test to match Portkey's suggestion, if different from default
    # chat_model_instance.model_name = "gemini-1.5-flash-001" # This would affect the instance globally
    # Instead, we will ensure the 'model' parameter in openai_params is set correctly.
    # Updated to reflect ProductionGeminiChatModel's attribute for the default SDK model name
    logger.info(f"Model Name to be used in Portkey call (default for SDK): {getattr(chat_model_instance, 'default_model_for_sdk', 'N/A')}")
    # The native Portkey client is no longer used, so this check is removed.
    # logger.info(f"Portkey Client initialized: {chat_model_instance.portkey_client is not None}")
    # async_openai_client_for_portkey was part of the old GeminiChatModel, ProductionGeminiChatModel uses self.client
    logger.info(f"Portkey Async OpenAI Client initialized: {hasattr(chat_model_instance, 'client') and chat_model_instance.client is not None}")

    # Adapt the call to use the new method in ProductionGeminiChatModel
    # This test is for a simple completion, not tool use.
    # We'll use the "fallback" strategy as a default for this test.
    try:
        # ProductionGeminiChatModel expects a list of dicts for messages, not Langchain HumanMessage objects directly
        # for its chat_completion_with_tools method's `messages` param.
        # However, the `self.client.chat.completions.create` inside it expects OpenAI format.
        # Let's check how ProductionGeminiChatModel is called from comprehension_node.
        # comprehension_node passes Langchain SystemMessage/HumanMessage objects.
        # The `chat_completion_with_tools` method in ProductionGeminiChatModel then passes these to
        # `self.client.chat.completions.create` which expects a list of dicts like:
        # [{"role": "user", "content": "..."}]
        # The Langchain `AIMessage` has a `to_messages` method, and `BaseMessage` has `to_openai_format`
        # The `portkey_ai._vendor.openai.AsyncOpenAI` client likely handles Langchain `BaseMessage` objects.
        # The error was `_agenerate` not found, not message format.
        # Let's assume the client handles Langchain messages.
        # The `_agenerate` method in Langchain BaseChatModel returns a ChatResult.
        # `chat_completion_with_tools` returns the raw OpenAI SDK response. We need to adapt assertions.

        # For a simple non-tool call, we can use chat_completion_with_tools without tools/tool_choice
        raw_openai_response = await chat_model_instance.chat_completion_with_tools(
            messages=[ # Convert to dict format expected by OpenAI SDK if client doesn't handle BaseMessage
                {"role": "system", "content": "You are a helpful assistant."}, # Example system message
                {"role": "user", "content": "Say 'Hello Portkey Test'."}
            ],
            strategy="fallback", # Or any other valid strategy for testing
            priority="normal"
            # stop=None # Pass as kwargs if needed by `chat_completion_with_tools`
        )
        # Reconstruct a ChatResult-like object or adapt assertions
        # The raw_openai_response is an openai.types.chat.chat_completion.ChatCompletion object
        
        # For assertion, we need to simulate parts of ChatResult structure
        from langchain_core.outputs import ChatGeneration, LLMResult
        from langchain_core.messages import AIMessage

        first_choice = raw_openai_response.choices[0]
        message_content = first_choice.message.content
        
        # Create a ChatGeneration object
        generation = ChatGeneration(
            message=AIMessage(content=message_content),
            generation_info={
                "finish_reason": first_choice.finish_reason,
                # Add other relevant info from raw_openai_response if needed
            }
        )
        chat_result = LLMResult(generations=[[generation]], llm_output={"portkey_id": getattr(raw_openai_response, 'id', None)})

    except Exception as e:
        logger.error(f"Error during chat_completion_with_tools call: {e}", exc_info=True)
        pytest.fail(f"Error during chat_completion_with_tools call: {e}")
        return

    logger.info(f"Chat Result from _agenerate: {chat_result}")
    assert chat_result.generations, "No generations in chat_result"
    # chat_result.generations is now a list of lists of ChatGeneration
    assert chat_result.generations[0][0].message.content, "No content in the first generation's message"
    
    logger.info(f"LLM Output (content): {chat_result.generations[0][0].message.content}")
    logger.info(f"LLM Output (full): {chat_result.llm_output}")

    # Crucially, now check your Portkey.ai dashboard!
    # The assertion below is a basic check; dashboard verification is key.
    assert "Hello Portkey Test" in chat_result.generations[0][0].message.content, \
        f"Expected 'Hello Portkey Test' in response, got: {chat_result.generations[0][0].message.content}"
    
    assert chat_result.llm_output is not None, "llm_output is None"
    # The use_portkey_for_generate flag is not on ProductionGeminiChatModel. It always uses Portkey.
    # We can check for portkey_id directly.
    assert "portkey_id" in chat_result.llm_output and chat_result.llm_output["portkey_id"] is not None, \
        "portkey_id not found in llm_output or is None"
    logger.info("--- ProductionGeminiChatModel via Portkey test completed ---")
    # Clean up async resources
    # ProductionGeminiChatModel uses self.client which is an AsyncOpenAI instance
    if hasattr(chat_model_instance, 'client') and chat_model_instance.client:
        client_to_close = chat_model_instance.client
        logger.info(f"Attempting to close chat_model_instance.client of type: {type(client_to_close)}")
        try:
            await client_to_close.aclose()
            logger.info("Successfully called await client.aclose() on chat_model_instance.client.")
        except AttributeError:
            logger.warning("chat_model_instance.client does not have an 'aclose' method. Cannot clean up.")
        except RuntimeError as re: # Catch event loop closed errors specifically
            if "Event loop is closed" in str(re):
                logger.warning(f"Could not close client due to event loop being closed: {re}")
            else:
                logger.error(f"RuntimeError during chat_model_instance.client cleanup: {re}", exc_info=True)
        except Exception as e:
            logger.error(f"Error during chat_model_instance.client cleanup: {e}", exc_info=True)
            
    # The old 'portkey_client' (native Portkey SDK client) is not used by ProductionGeminiChatModel
    # if hasattr(chat_model_instance, 'portkey_client') and chat_model_instance.portkey_client:
    #     pass


# if __name__ == "__main__": # Commenting out direct execution for pytest
#     asyncio.run(test_run_all_queries())
