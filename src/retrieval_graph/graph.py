"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""
import logging
import time # Added for timing
from typing import Any, Literal, Optional, TypedDict, cast

logger = logging.getLogger(__name__) # Initialize logger at module level

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver # Import MemorySaver (will be replaced)
from supabase import create_client # Import Supabase client, removed AsyncClient
from almustashar_api.config import USERS_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY # Import Supabase credentials

# Custom node for Yemeni legal query understanding
from custom_nodes.comprehension_nodes import understand_yemeni_legal_query_node
# Import the new multi-embedding node
from custom_nodes.embedding_nodes import generate_multi_embeddings_node
# Existing custom nodes
from custom_nodes.retrieval_nodes import (
    generate_query_embedding_node,
    # pinecone_semantic_retriever_node, # Removed
    supabase_hybrid_retriever_node, 
)
# Import the new synthesis and conversational nodes
from custom_nodes.response_nodes import (
    synthesize_yemeni_legal_answer_node,
    handle_conversational_query_node,
    prepare_synthesis_node,
)
# Import new direct lookup nodes
from custom_nodes.direct_lookup_node import (
    execute_direct_supabase_lookup_node, # Changed from prepare_direct_lookup_node
    # process_direct_lookup_result_node, # This node is now removed/merged
)

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router # Router might be deprecated soon
from shared.utils import format_docs, load_chat_model


async def analyze_and_route_query( # This will likely be deprecated or repurposed
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model, configuration) # Pass full configuration
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"] # This uses the old Router TypedDict
    if _type == "langchain": # This classification is from the old router
        return "create_research_plan" # This path will be disconnected
    elif _type == "more-info":
        return "ask_for_more_info" # This path will be disconnected
    elif _type == "general":
        return "respond_to_general_query" # This path will be disconnected
    else:
        # This path should ideally not be hit if understand_yemeni_legal_query_node is the entry.
        # If it is, it means state.router was somehow populated by an old mechanism.
        # For safety, we can route to a generic response or END.
        # However, the new routing logic below should take precedence.
        raise ValueError(f"Unknown router type from old router: {_type}")

def initial_intake_router(state: AgentState) -> Literal[
    "process_direct_lookup_result_node",
    "understand_yemeni_legal_query_node",
    "handle_conversational_query_node"  # Added new possible route
]:
    """
    Initial router to check for simple greetings or if the query needs full understanding.
    The check for pending MCP responses for direct lookup is removed as direct lookup is now synchronous within the graph.
    """
    # Define simple Arabic greetings
    simple_greetings = [
        "السلام عليكم",
        "سلام عليكم",
        "سلام",
        "مرحبا",
        "أهلا",
        "اهلا",
        "أهلا وسهلا",
        "اهلا وسهلا",
        "صباح الخير",
        "مساء الخير",
    ]
    # Normalize query for comparison (e.g., remove punctuation, extra spaces if needed)
    last_message_content = ""
    if state.messages and isinstance(state.messages[-1].content, str):
        last_message_content = state.messages[-1].content.strip()

    if last_message_content in simple_greetings:
        logger.info(f"Simple greeting '{last_message_content}' detected. Routing directly to handle_conversational_query_node.")
        return "handle_conversational_query_node"

    logger.info("No simple greeting, routing to understand_yemeni_legal_query_node for full query processing.")
    return "understand_yemeni_legal_query_node"

def route_yemeni_legal_query(state: AgentState, *, config: RunnableConfig) -> Literal[
    "generate_query_embedding_node",
    "handle_conversational_query_node",
    "execute_direct_supabase_lookup_node",
    "error_handler_node",
    "END"
]:
    """
    Determines the next step based on the classification from understand_yemeni_legal_query_node.
    """
    classification = state.query_classification
    analysis_result = state.query_analysis_result
    
    app_config = AgentConfiguration.from_runnable_config(config)
    agent_persona = app_config.agent_persona

    logger.info(f"Routing based on Yemeni legal query classification: {classification} for persona: {agent_persona}")

    if state.error_message and state.error_node == "understand_yemeni_legal_query_node":
        logger.error(f"Error in understand_yemeni_legal_query_node: {state.error_message}. Routing to END.")
        return "END"

    if classification == "legal_query_conceptual_search":
        if agent_persona == "المستشار":
            logger.info(f"Query is conceptual search for 'المستشار' persona. Routing to generate_multi_embeddings_node.")
            return "generate_multi_embeddings_node"
        else:
            logger.info(f"Query is conceptual search for '{agent_persona}' persona. Routing to generate_query_embedding_node.")
            return "generate_query_embedding_node"
            
    elif classification == "legal_query_direct_lookup":
        if analysis_result and analysis_result.law_name and analysis_result.article_number:
            logger.info("Query classified as legal_query_direct_lookup. Routing to execute_direct_supabase_lookup_node.")
            return "execute_direct_supabase_lookup_node"
        else:
            logger.warning("Classification is legal_query_direct_lookup, but missing law_name/article_number. Falling back to conceptual search.")
            return "generate_query_embedding_node"

    elif classification == "conversational" or classification == "other":
        logger.info(f"Query classified as '{classification}'. Routing to handle_conversational_query_node.")
        return "handle_conversational_query_node"
        
    else:
        logger.warning(f"Unknown or None query classification: '{classification}'. Routing to handle_conversational_query_node as a fallback.")
        return "handle_conversational_query_node"


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model, configuration)
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model, configuration)
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model, configuration).with_structured_output(Plan)
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    response = cast(Plan, await model.ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model, configuration)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)

# Add nodes
builder.add_node("understand_yemeni_legal_query_node", understand_yemeni_legal_query_node)
builder.add_node("execute_direct_supabase_lookup_node", execute_direct_supabase_lookup_node)
builder.add_node("generate_query_embedding_node", generate_query_embedding_node)
builder.add_node("generate_multi_embeddings_node", generate_multi_embeddings_node)
builder.add_node("supabase_hybrid_retriever_node", supabase_hybrid_retriever_node)
builder.add_node("prepare_synthesis_node", prepare_synthesis_node)
builder.add_node("synthesize_yemeni_legal_answer_node", synthesize_yemeni_legal_answer_node)
builder.add_node("handle_conversational_query_node", handle_conversational_query_node)

# Define Edges
builder.set_conditional_entry_point(
    initial_intake_router,
    {
        "understand_yemeni_legal_query_node": "understand_yemeni_legal_query_node",
        "handle_conversational_query_node": "handle_conversational_query_node",
    },
)

builder.add_conditional_edges(
    "understand_yemeni_legal_query_node",
    route_yemeni_legal_query,
    {
        "generate_query_embedding_node": "generate_query_embedding_node",
        "generate_multi_embeddings_node": "generate_multi_embeddings_node",
        "handle_conversational_query_node": "handle_conversational_query_node",
        "execute_direct_supabase_lookup_node": "execute_direct_supabase_lookup_node",
        "END": END,
    }
)

builder.add_edge("execute_direct_supabase_lookup_node", "prepare_synthesis_node")
builder.add_edge("generate_query_embedding_node", "supabase_hybrid_retriever_node")
builder.add_edge("generate_multi_embeddings_node", "supabase_hybrid_retriever_node")
builder.add_edge("supabase_hybrid_retriever_node", "prepare_synthesis_node")
builder.add_edge("prepare_synthesis_node", "synthesize_yemeni_legal_answer_node")
builder.add_edge("synthesize_yemeni_legal_answer_node", END)
builder.add_edge("handle_conversational_query_node", END)

graph = builder.compile()
graph.name = "AlMustasharRetrievalGraph"
