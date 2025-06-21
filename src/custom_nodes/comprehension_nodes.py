"""Custom LangGraph nodes for query comprehension and analysis for "المستشار" project."""

import json
import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Literal 
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field 
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.embeddings import Embeddings # Added for type checking

from custom_providers.production_gemini_chat_model import ProductionGeminiChatModel
from custom_providers.openai_custom_embeddings import OpenAICustomEmbeddings 
from shared.retrieval import make_text_encoder 
from shared.configuration import BaseConfiguration 
from retrieval_graph.configuration import AgentConfiguration 
from shared.utils import load_chat_model 

try:
    from utils.strategy_selector import StrategySelector
except ImportError:
    from src.utils.strategy_selector import StrategySelector 

try:
    from shared.legal_vocabularies import HIERARCHICAL_LAW_CLASSIFICATIONS, HIERARCHICAL_ARTICLE_TAGS
except ImportError:
    from src.shared.legal_vocabularies import HIERARCHICAL_LAW_CLASSIFICATIONS, HIERARCHICAL_ARTICLE_TAGS


from retrieval_graph.state import AgentState 
from shared.models import YemeniLegalQueryAnalysis 

logger = logging.getLogger(__name__)

YEMENI_LEGAL_ANALYSIS_TOOL_NAME = "analyze_yemeni_legal_query"
SIMPLE_CLASSIFICATION_TOOL_NAME = "simple_classify_query" 

LEGAL_SYNONYM_DICTIONARY: Dict[str, List[str]] = {
    "مدة": ["ميعاد", "أجل", "فترة"],
    "ميعاد": ["مدة", "أجل", "فترة"],
    "أجل": ["مدة", "ميعاد", "فترة"],
    "عقوبة": ["جزاء", "عقاب"],
    "جزاء": ["عقوبة", "عقاب"],
    "قانون": ["تشريع", "نظام"],
    "تشريع": ["قانون", "نظام"],
    "استئناف": ["طعن بالاستئناف"],
    "طعن": ["اعتراض", "استئناف", "نقض", "تمييز"], 
}

class SimpleClassificationTool(BaseModel):
    classification: Literal["conversational", "other"] = Field(
        ...,
        description="Classify the query. Use 'conversational' for general chat, greetings, or simple questions not requiring legal knowledge. Use 'other' if it might need more detailed analysis or involves legal terms."
    )

def _clean_tool_schema_for_google(tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tool_schema, dict) or tool_schema.get("type") != "function":
        return tool_schema
    function_def = tool_schema.get("function")
    if not isinstance(function_def, dict):
        return tool_schema
    parameters = function_def.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        return tool_schema
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return tool_schema
    
    cleaned_tool_schema = json.loads(json.dumps(tool_schema))
    cleaned_properties = cleaned_tool_schema["function"]["parameters"]["properties"]
    
    cleaned_something = False
    for param_name, param_schema in properties.items():
        if isinstance(param_schema, dict) and "anyOf" in param_schema:
            cleaned_something = True
            cleaned_properties[param_name] = {"anyOf": param_schema["anyOf"]}
    return cleaned_tool_schema if cleaned_something else tool_schema

async def _quick_query_analysis(query: str) -> Dict[str, Any]:
    logger.debug(f"Performing quick analysis for query: {query[:100]}...")
    analysis = {
        "complexity": "medium", 
        "urgency": "normal",    
        "query_type": "general" 
    }
    query_lower = query.lower().strip()
    legal_keywords = [
        "قانون", "مادة", "محكمة", "عقوبة", "جريمة", "حقوق", "تشريع", "دستور",
        "محامي", "قاضي", "قضية", "دعوى", "استئناف", "نقض", "لائحة", "نظام", "حكم", "إجراءات",
        "شهادة", "شهود", "نصاب", "حد", "حدود", "قصاص", "دية", "إثبات", "بينة", "يمين",
        "law", "article", "court", "penalty", "crime", "rights", "legislation", "constitution",
        "lawyer", "judge", "case", "procedure", "ruling", "decree", "testimony", "witness", "evidence"
    ]
    is_legal_query_heuristic = any(keyword in query_lower for keyword in legal_keywords)

    if is_legal_query_heuristic:
        analysis["query_type"] = "legal_keyword_detected"
    else:
        purely_conversational_phrases = [
            "من انت", "عرفني على نفسك", "ما اسمك", "ماذا تفعل", "كيف حالك", "ايش الاخبار", "ايش العلوم",
            "مرحبا بك", "أهلا بك", "شكرا لك", "شكرا جزيلا", "عفوا", "تسلم", "الله يسلمك",
            "who are you", "introduce yourself", "what is your name", "what do you do", "how are you", "what's up",
            "thank you", "thanks", "you're welcome",
            "ماذا يمكنك ان تفعل لي", "ايش تقدر تسوي", "ايش خدماتك"
        ]
        general_question_starters = ["هل يمكنك", "هل تستطيع", "تقدر", "ممكن"]
        if any(phrase == query_lower for phrase in purely_conversational_phrases) or \
           (len(query.split()) < 7 and any(query_lower.startswith(starter) for starter in general_question_starters)) or \
           (len(query.split()) < 5 and not is_legal_query_heuristic): 
            analysis["query_type"] = "purely_conversational_heuristic"; analysis["complexity"] = "low" 

    if analysis["query_type"] != "purely_conversational_heuristic":
        if any(indicator in query_lower for indicator in ["compare", "analyze", "complex", "multiple", "detailed", "explain in depth", "implications of", "ما هي الآثار المترتبة", "اشرح بالتفصيل"]):
            analysis["complexity"] = "high"
        if any(indicator in query_lower for indicator in ["urgent", "deadline", "court hearing", "immediate", "emergency", "asap", "مستعجل", "جلسة محكمة", "ضروري جدا"]):
            analysis["urgency"] = "critical"
            if analysis["query_type"] in ["general", "legal_keyword_detected"]: analysis["query_type"] = "court_deadline_hint" 
        elif any(indicator in query_lower for indicator in ["important", "priority", "مهم"]):
            analysis["urgency"] = "high"
        if any(indicator in query_lower for indicator in ["what is", "define", "meaning of", "definition of", "tell me about article", "ما هو", "عرف", "معنى"]) and analysis["complexity"] != "high":
            analysis["complexity"] = "low"
            if analysis["query_type"] in ["general", "legal_keyword_detected"]: analysis["query_type"] = "simple_definition"
        if re.search(r"(article|المادة)\s+\d+", query_lower) and re.search(r"(law|code|قانون|نظام)", query_lower) and analysis["complexity"] != "high":
            analysis["query_type"] = "direct_lookup_hint"; analysis["complexity"] = "low"
        if len(query.split()) < 5 and analysis["complexity"] != "high" and analysis["query_type"] in ["general", "legal_keyword_detected"]:
            analysis["complexity"] = "low"; analysis["query_type"] = "short_query" 
            
    logger.debug(f"Quick analysis result: {analysis}")
    return analysis

def _parse_llm_tool_response(response_message: AIMessage, tool_name: str) -> Optional[YemeniLegalQueryAnalysis]:
    if response_message and response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.type == "function" and tool_call.function and tool_call.function.name == tool_name:
                try:
                    args_data_str = tool_call.function.arguments
                    if isinstance(args_data_str, str):
                        args_dict = json.loads(args_data_str)
                        logger.info(f"Successfully parsed {tool_name} from AIMessage.tool_calls (function.arguments). Args: {args_dict}")
                        return YemeniLegalQueryAnalysis(**args_dict)
                    else: logger.error(f"Tool call function.arguments for {tool_name} is not a string: {type(args_data_str)}. Value: {args_data_str}")
                except json.JSONDecodeError as json_e: logger.error(f"JSONDecodeError parsing tool_call function.arguments for {tool_name}: {args_data_str}. Error: {json_e}")
                except Exception as e: logger.error(f"Failed to instantiate YemeniLegalQueryAnalysis from tool_call function.arguments: {args_data_str}. Error: {e}")
            else: logger.debug(f"Skipping tool_call with type '{tool_call.type}' and name '{getattr(tool_call.function, 'name', 'N/A')}'")
    return None

def _parse_llm_content_response(content: str, tool_name_in_json: str) -> Optional[YemeniLegalQueryAnalysis]:
    if not content: return None
    cleaned_content = content.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_content, re.DOTALL)
    if match: cleaned_content = match.group(1).strip()
    if not cleaned_content: return None
    try:
        parsed_data = json.loads(cleaned_content)
        args_dict = None
        if "name" in parsed_data and parsed_data.get("name") == tool_name_in_json and "args" in parsed_data and isinstance(parsed_data["args"], dict):
            args_dict = parsed_data["args"]
        elif all(key in parsed_data for key in YemeniLegalQueryAnalysis.model_fields.keys() if YemeniLegalQueryAnalysis.model_fields[key].is_required()):
            args_dict = parsed_data
        if args_dict:
            logger.debug(f"Successfully parsed YemeniLegalQueryAnalysis from .content fallback using tool name '{tool_name_in_json}'.")
            return YemeniLegalQueryAnalysis(**args_dict)
        else:
            logger.error(f"Parsed JSON from .content does not match expected 'args' structure for tool '{tool_name_in_json}' or direct fields.")
            return None
    except json.JSONDecodeError as e: logger.warning(f"Failed to parse JSON from .content: {e}. Raw content snippet for parsing: {cleaned_content[:500]}")
    except Exception as e: logger.error(f"Error instantiating YemeniLegalQueryAnalysis from parsed .content: {e}. Data: {cleaned_content[:500]}")
    return None

async def understand_yemeni_legal_query_node(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, Any]:
    node_name = "understand_yemeni_legal_query_node"
    logger.info(f"--- Executing {node_name} ---")
    start_time = time.perf_counter()

    strategy_selector = StrategySelector() 
    app_config = AgentConfiguration.from_runnable_config(config)
    llm = load_chat_model(app_config.query_model, app_config) 

    user_query = ""
    if state.messages and isinstance(state.messages[-1], HumanMessage):
        user_query = state.messages[-1].content
    else:
        logger.error("No user query (last HumanMessage) found in state.messages.")
        error_analysis = YemeniLegalQueryAnalysis(classification="other", raw_query="Error: No user query found", error_message="UnderstandQueryNode: No user query found")
        return {"query_analysis_result": error_analysis, "query_classification": "other", "text_for_embedding": "Error: No user query found", "error_message": "UnderstandQueryNode: No user query found in state.messages", "error_node": node_name, "identified_law_name": None, "identified_article_number": None, "search_keywords": [], "query_intent": None, "portkey_id": None, "strategy_used": "unknown", "priority_used": "unknown", "original_query_embedding": None}

    formatted_chat_history = "None"
    if state.messages and len(state.messages) > 1:
        history_messages = [f"{msg.type}: {str(msg.content).replace('{', '{{').replace('}', '}}')}" for msg in state.messages[:-1]]
        if history_messages: formatted_chat_history = "\\n".join(history_messages)

    final_query_analysis_result: Optional[YemeniLegalQueryAnalysis] = None
    final_classification: str = "other"; final_text_for_embedding: str = user_query
    final_identified_law_name: Optional[str] = None; final_identified_article_number: Optional[str] = None
    final_search_keywords: List[str] = []; final_query_intent: Optional[str] = None
    final_original_query_embedding: Optional[List[float]] = None
    portkey_id_from_response: Optional[str] = None
    strategy_used: str = "unknown"; priority_used: str = "unknown"
    error_message_to_return: Optional[str] = None; error_node_to_return: Optional[str] = None

    try:
        # Removed embedding of the original user query
        logger.info(f"[{node_name}] Original user query embedding step has been removed for optimization.")

        quick_analysis_results = await _quick_query_analysis(user_query)
        strategy_used, priority_used = strategy_selector.select_strategy(quick_analysis_results)
        timeout_config = strategy_selector.get_timeout_config(strategy_used)
        logger.info(f"Selected strategy: '{strategy_used}', priority: '{priority_used}' for query: {user_query[:100]}. Quick analysis type: {quick_analysis_results.get('query_type')}")

        tool_to_use_schema: Dict[str, Any]; tool_name_for_llm: str; system_prompt_content: str
        is_simple_classification_path = False

        if quick_analysis_results.get("query_type") == "purely_conversational_heuristic":
            logger.info("Query identified as purely_conversational_heuristic. Using SimpleClassificationTool.")
            is_simple_classification_path = True
            tool_name_for_llm = SIMPLE_CLASSIFICATION_TOOL_NAME
            raw_simple_tool_schema = convert_to_openai_tool(SimpleClassificationTool)
            raw_simple_tool_schema['function']['name'] = tool_name_for_llm
            tool_to_use_schema = _clean_tool_schema_for_google(raw_simple_tool_schema)
            system_prompt_content = f"""You are an AI assistant. Your task is to classify the user's query. You MUST respond by calling the function `{tool_name_for_llm}`. The arguments for the function must conform to the following JSON schema: ```json\n{json.dumps(tool_to_use_schema['function']['parameters'], indent=2, ensure_ascii=False)}\n```\nCurrent User Query: {user_query}\nClassify this query. Use 'conversational' for general chat, greetings, or simple questions not requiring legal knowledge. Use 'other' if it might need more detailed analysis or involves legal terms."""
        else:
            logger.info("Query not purely_conversational_heuristic. Using full YemeniLegalQueryAnalysis tool.")
            is_simple_classification_path = False
            tool_name_for_llm = YEMENI_LEGAL_ANALYSIS_TOOL_NAME
            raw_full_tool_schema = convert_to_openai_tool(YemeniLegalQueryAnalysis)
            raw_full_tool_schema['function']['name'] = tool_name_for_llm
            tool_to_use_schema = _clean_tool_schema_for_google(raw_full_tool_schema)
            
            system_prompt_content = f"""
You are an AI assistant that analyzes queries about Yemeni law.
You MUST respond by calling the function `{YEMENI_LEGAL_ANALYSIS_TOOL_NAME}`.
The arguments for the function must conform to the following JSON schema:
```json
{json.dumps(tool_to_use_schema['function']['parameters'], indent=2, ensure_ascii=False)}
```
Current User Query: {user_query}
Chat History (if any): {formatted_chat_history}

    Provide your analysis by calling the `{YEMENI_LEGAL_ANALYSIS_TOOL_NAME}` function with the appropriate arguments.

**General Instructions for Analysis:**
*   **Classification:** Determine if the query is 'conversational', 'legal_query_direct_lookup', 'legal_query_conceptual_search', or 'other'.
*   **Law & Article Identification:** If the query mentions a specific law name or article number, extract them into `law_name` and `article_number`.
*   **Keywords:** Extract relevant Arabic keywords for search into `keywords_for_search`. These should be precise terms from the query. (Keyword expansion will be handled separately).
*   **Intent:** Determine the user's primary intent (e.g., 'specific_article_lookup', 'conceptual_search', 'definition_seeking', 'general_overview').

**NEW Instructions for Enhanced Metadata Fields (Populate these if applicable):**
*   **`identified_law_categories`**: If the query implies specific legal domains (e.g., "commercial law", "criminal procedure"), list the corresponding categories. You MUST choose one or more exclusively from the 'Official Law Classifications' list provided below. Select the most specific relevant hierarchical path(s).
    Official Law Classifications:
    ```json
    {json.dumps(HIERARCHICAL_LAW_CLASSIFICATIONS, ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}
    ```
*   **`identified_tags`**: Based on the user query, identify key legal concepts. Then, from the 'Official Article Tag Vocabulary' provided below, select 1 to 5 of the most relevant and specific **complete tag strings or valid prefix paths** that correspond to these concepts. A valid prefix means it's the beginning part of one or more tags in the vocabulary (e.g., if 'القانون_المدني/العقود/عقد_البيع' is in the vocabulary, then 'القانون_المدني/العقود' would be a valid prefix if the query is about civil contracts generally).
    Official Article Tag Vocabulary:
    ```json
    {json.dumps(HIERARCHICAL_ARTICLE_TAGS, ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}
    ```
*   **`query_intent_details`**: Provide a structured dictionary detailing the specific nature of the query if it's more complex than a simple lookup or classification. For example: `{{'type': 'comparison', 'items': ['law A', 'law B'], 'aspect': 'penalties'}}` or `{{'type': 'procedure_inquiry', 'subject': 'company registration'}}`.
*   **`filter_logic`**: Suggest 'AND' if the query implies multiple strict conditions that must all be met. Suggest 'OR' if the query is broader or exploring multiple related concepts. Default to 'AND'.
*   **`confidence_scores`**: Provide a dictionary of your confidence (0.0 to 1.0) for each piece of information you extract for the new fields (e.g., `{{{{'identified_law_categories': 0.8, 'identified_tags': 0.7, 'query_intent_details': 0.6}}}}`).

**Instructions for 'hypothetical_answer_for_embedding' field (CRITICAL FOR ACCURATE RETRIEVAL - Context-Anchored HyDE):**
1.  **Language:** Generate this paragraph **exclusively in ARABIC**.
2.  **Style:** Write as if it's an excerpt from a definitive Yemeni legal text or a highly authoritative legal summary. It must be formal, objective, and fact-centric.
3.  **Context Anchoring (NEW & IMPORTANT):**
    *   Begin the hypothetical answer by explicitly mentioning the primary law category you've identified for the query.
    *   Subtly weave in 1-2 of the most specific `identified_tags` into the answer if they fit naturally and add precision.
4.  **Content Focus & Adaptability to Query Type (NEW & IMPORTANT - Strive for Legal Accuracy and Completeness):**
    *   **Directly address the user's query.** Summarize the core legal principles, rules, or information that would comprehensively answer it.
    *   **If the query seeks a singular, direct piece of factual information**: The hypothetical answer should be **concise and directly state this factual information**.
    *   **If the query bridges substantive and procedural/evidentiary law**: The hypothetical answer should try to incorporate key identifying terms from **both** legal domains.
    *   **If the query is more general or seeks an overview, or asks for a specific duration/period (e.g., "كم مدة الطعن")**: The hypothetical answer MUST start by stating the **most general AND DIRECT applicable rule or principle.** If a specific article (e.g., "المادة 275") is known or strongly suspected to provide a common, general numerical value for such a duration (e.g., "ستون يوماً"), the hypothetical answer **MUST be phrased as if directly quoting or very closely paraphrasing that article's core provision regarding the duration and article number (e.g., 'المادة 275 تنص على أن ميعاد الطعن ستون يوماً').** Avoid indirect phrasing like 'may be referred to'. It MAY then briefly mention exceptions or related articles for specific cases ONLY AFTER stating the primary rule directly.
    *   **If the query is specific about complex conditions or penalties**: The hypothetical answer MUST be highly specific.
    *   **If the query seeks a definition**: Provide a formal legal definition.
    *   **Incorporate relevant Yemeni legal terms and concepts accurately throughout.**
    *   Mention relevant (even if hypothetical but plausible) Yemeni law names or article numbers if appropriate.
    *   **Avoid generic procedural language** unless the query is *explicitly* about those aspects.
5.  **Purpose:** This text is NOT for the user. It is a "pseudo-document" for semantic search.
6.  **Conciseness & Density:** Be concise but information-dense. Aim for a single, well-structured paragraph.
7.  **No Conversational Elements:** Do NOT include conversational phrases.
8.  **If Query is Vague/Broad:** Provide a concise summary of the core legal topic.
"""
        
        logger.debug(f"Tool schema for LLM: {json.dumps(tool_to_use_schema, indent=2, ensure_ascii=False)}")

        messages_for_llm_lc_format = [SystemMessage(content=system_prompt_content), HumanMessage(content=user_query)]
        openai_formatted_messages = [{"role": ("user" if msg.type == "human" else "assistant" if msg.type == "ai" else msg.type), "content": str(msg.content)} for msg in messages_for_llm_lc_format]
        
        logger.info(f"Sending to ProductionGeminiChatModel with tool '{tool_name_for_llm}', strategy '{strategy_used}', priority '{priority_used}'. Persona: {app_config.agent_persona}")
        portkey_user_metadata = {"user_id": state.user_id, "_user": state.user_id} if state.user_id else None
        if portkey_user_metadata: logger.info(f"Passing user_metadata to Portkey: {portkey_user_metadata}")
        else: logger.warning("No user_id found in AgentState for Portkey metadata.")

        llm_response_raw = await llm.chat_completion_with_tools(messages=openai_formatted_messages, tools=[tool_to_use_schema], tool_choice={"type": "function", "function": {"name": tool_name_for_llm}}, strategy=strategy_used, priority=priority_used, user_metadata=portkey_user_metadata, timeout=timeout_config.get("timeout", 60), temperature=0.1)
        portkey_id_from_response = getattr(llm_response_raw, 'id', None)
        
        if llm_response_raw and llm_response_raw.choices:
            response_message = llm_response_raw.choices[0].message
            if is_simple_classification_path:
                simple_classification_tool_call_found = False
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.type == "function" and tool_call.function and tool_call.function.name == SIMPLE_CLASSIFICATION_TOOL_NAME:
                            simple_classification_tool_call_found = True
                            try:
                                args_dict = json.loads(tool_call.function.arguments); simple_classification_obj = SimpleClassificationTool(**args_dict)
                                final_classification = simple_classification_obj.classification; logger.info(f"Simple classification successful: {final_classification}")
                            except Exception as e: logger.error(f"Failed to parse SimpleClassificationTool: {e}. Args: {tool_call.function.arguments}"); final_classification = "other"; error_message_to_return = "Failed to parse simple classification tool"; error_node_to_return = node_name
                            break 
                if not simple_classification_tool_call_found: logger.warning("SimpleClassificationTool call expected but not found. Defaulting classification to 'other'."); final_classification = "other"; error_message_to_return = "Simple classification tool call not found in LLM response"; error_node_to_return = node_name
                final_query_analysis_result = YemeniLegalQueryAnalysis(raw_query=user_query, classification=final_classification, error_message=error_message_to_return)
                final_text_for_embedding = user_query
            else: 
                ai_response_object_full = _parse_llm_tool_response(response_message, YEMENI_LEGAL_ANALYSIS_TOOL_NAME)
                if not ai_response_object_full:
                    logger.info("Tool call not found in AIMessage.tool_calls for full analysis, attempting to parse from .content.")
                    raw_content = getattr(response_message, "content", ""); logger.info(f"RAW LLM Response Content (for full analysis fallback parsing):\n'{raw_content}'")
                    ai_response_object_full = _parse_llm_content_response(raw_content, YEMENI_LEGAL_ANALYSIS_TOOL_NAME)
                if not ai_response_object_full:
                    logger.error(f"Failed to get structured YemeniLegalQueryAnalysis from LLM for query: {user_query[:100]}")
                    final_query_analysis_result = YemeniLegalQueryAnalysis(classification="other", raw_query=user_query, error_message="Failed to parse LLM response for query analysis.")
                    error_message_to_return = final_query_analysis_result.error_message; error_node_to_return = node_name
                else: final_query_analysis_result = ai_response_object_full
                if final_query_analysis_result: 
                    if not final_query_analysis_result.raw_query: final_query_analysis_result.raw_query = user_query
                    if final_query_analysis_result.hypothetical_answer_for_embedding: final_query_analysis_result.hypothetical_answer_for_embedding = final_query_analysis_result.hypothetical_answer_for_embedding.replace('{', '{{').replace('}', '}}')
                    final_classification = final_query_analysis_result.classification
                    final_text_for_embedding = final_query_analysis_result.hypothetical_answer_for_embedding if final_query_analysis_result.hypothetical_answer_for_embedding else user_query
                    final_identified_law_name = final_query_analysis_result.law_name; final_identified_article_number = final_query_analysis_result.article_number
                    original_keywords = final_query_analysis_result.keywords_for_search or []
                    expanded_keywords = set(original_keywords)
                    for keyword in original_keywords:
                        if keyword in LEGAL_SYNONYM_DICTIONARY: expanded_keywords.update(LEGAL_SYNONYM_DICTIONARY[keyword])
                    final_search_keywords = list(expanded_keywords)
                    if len(final_search_keywords) > len(original_keywords): logger.info(f"Keywords expanded from {original_keywords} to {final_search_keywords}")
                    else: final_search_keywords = original_keywords
                    final_query_intent = final_query_analysis_result.intent
                    if final_query_analysis_result.error_message and not error_message_to_return: error_message_to_return = final_query_analysis_result.error_message; error_node_to_return = node_name
        else: 
            logger.error(f"LLM response had no choices or was None for query: {user_query[:100]}")
            error_message_to_return = "LLM response was empty or malformed."; error_node_to_return = node_name
            final_classification = "other"; final_query_analysis_result = YemeniLegalQueryAnalysis(classification=final_classification, raw_query=user_query, error_message=error_message_to_return)
        
        if not final_query_analysis_result:
             final_query_analysis_result = YemeniLegalQueryAnalysis(classification=final_classification, raw_query=user_query, error_message=error_message_to_return or "Analysis result object not created due to an earlier error.")
             if not error_message_to_return: error_message_to_return = "Analysis result object not created."; error_node_to_return = node_name
        
        if final_query_analysis_result and final_query_analysis_result.error_message and not error_message_to_return:
            error_message_to_return = final_query_analysis_result.error_message; error_node_to_return = node_name

        # Keyword Augmentation for General Appeal Duration
        if (final_query_analysis_result and
            final_query_analysis_result.classification == "legal_query_conceptual_search" and
            "كم مدة" in user_query and "الاستئناف" in user_query and
            not final_query_analysis_result.law_name and
            not final_query_analysis_result.article_number):
            
            augmented_keywords = set(final_search_keywords)
            specific_terms = ["ستون يوما", "ميعاد الطعن ستون يوما", "المادة 275"]
            for term in specific_terms:
                if term not in augmented_keywords: augmented_keywords.add(term)
            if len(augmented_keywords) > len(final_search_keywords):
                logger.info(f"[{node_name}] Augmented keywords for general appeal duration query. Original: {final_search_keywords}, Augmented: {list(augmented_keywords)}")
                final_search_keywords = list(augmented_keywords)
                if final_query_analysis_result: final_query_analysis_result.keywords_for_search = final_search_keywords 
        
        logger.warning(f"""
--- [{node_name}] Final Comprehension Results ---
- Classification: {final_classification}
- Hypothetical Answer (for embedding): {final_text_for_embedding}
- Identified Law Categories: {final_query_analysis_result.identified_law_categories if final_query_analysis_result else 'N/A'}
- Identified Tags: {final_query_analysis_result.identified_tags if final_query_analysis_result else 'N/A'}
- Search Keywords: {final_search_keywords}
- Law Name: {final_identified_law_name}
- Article Number: {final_identified_article_number}
----------------------------------------------------
""")

        return {
            "query_analysis_result": final_query_analysis_result, "query_classification": final_classification,
            "text_for_embedding": final_text_for_embedding, "identified_law_name": final_identified_law_name,
            "identified_article_number": final_identified_article_number, "search_keywords": final_search_keywords,
            "query_intent": final_query_intent, "original_query_embedding": final_original_query_embedding,
            "error_message": error_message_to_return, "error_node": error_node_to_return,
            "portkey_id": portkey_id_from_response, "strategy_used": strategy_used, "priority_used": priority_used
        }
    except Exception as e:
        raw_exception_text = str(e); full_traceback = traceback.format_exc()
        logger.error("Critical error in node %s for query '%s':\n%s\nFull traceback:\n%s", node_name, user_query[:100], raw_exception_text, full_traceback)
        error_analysis_obj = YemeniLegalQueryAnalysis(classification="other", raw_query=user_query, error_message=f"Node critical error: {raw_exception_text}")
        return {
            "query_analysis_result": error_analysis_obj, "query_classification": "other", "text_for_embedding": user_query,
            "identified_law_name": None, "identified_article_number": None, "search_keywords": [], "query_intent": "unknown",
            "original_query_embedding": final_original_query_embedding, 
            "error_message": f"{node_name}: Critical error (see server logs - {raw_exception_text})", "error_node": node_name,
            "portkey_id": portkey_id_from_response, "strategy_used": strategy_used, "priority_used": priority_used
        }
    finally:
        end_time = time.perf_counter()
        logger.info(f"--- {node_name} execution time: {end_time - start_time:.4f} seconds ---")
