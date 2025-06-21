"""
Proof-of-Concept script for LLM-assisted classification of laws
and tagging of law articles from the Supabase database.
Now also includes tagging for comments.

This script will:
1. Connect to the Supabase database.
2. Fetch laws, articles, and comments that need metadata suggestions.
3. For each, prepare a prompt and call an LLM.
4. Write the LLM's suggestions to new 'suggested_*' columns in the database.
   These suggestions are marked for human review.
"""
import asyncio
import os
import json
import logging # Added for more structured logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from supabase import create_client, Client as SupabaseClient
from tqdm.asyncio import tqdm as async_tqdm

# Assuming LLM model utilities will be imported from the project structure
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.custom_providers.production_gemini_chat_model import ProductionGeminiChatModel
    # AgentConfiguration might not be needed if all configs are passed or available globally
    # from src.retrieval_graph.configuration import AgentConfiguration 
    from scripts.db_processing.classification_models import ArticleTaggingOutput, CommentTaggingOutput # Added CommentTaggingOutput
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure this script is run from the project root or paths are correctly set.")
    sys.exit(1)

# --- Configuration ---
load_dotenv(os.path.join(project_root, '.env'))

logger = logging.getLogger(__name__) 
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'), 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

BATCH_SIZE = 50 # Reduced batch size for potentially longer comment processing
MAX_CONCURRENT_TASKS = 5 # Reduced concurrency slightly

KNOWLEDGE_SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL")
KNOWLEDGE_SUPABASE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY")

# FINALIZED_LAW_CATEGORIES removed as law classification is separate

# Using the comprehensive vocabulary for both articles and comments for now.
# This can be split or customized later if needed.
SHARED_TAG_VOCABULARY = [
    # ... (keeping the extensive list from the original file for brevity here) ...
    # --- 1. فروع القانون الموضوعي الرئيسية ---
    "القانون_الدستوري", "القانون_الدستوري/نظام_الحكم", "القانون_الدستوري/السلطات_العامة", "القانون_الدستوري/الحقوق_والحريات_الأساسية",
    "القانون_المدني", "القانون_المدني/الأشخاص_والأهلية", "القانون_المدني/الحقوق_العينية", "القانون_المدني/الحقوق_العينية/الملكية", 
    "القانون_المدني/الحقوق_العينية/الحقوق_المتفرعة_عن_الملكية", "القانون_المدني/الالتزامات_(مصادر_وآثار)", "القانون_المدني/العقود_(أحكام_عامة)",
    "القانون_المدني/العقود/عقد_البيع", "القانون_المدني/العقود/عقد_الإيجار", "القانون_المدني/العقود/عقد_الشركة_(مدنية)", "القانون_المدني/العقود/عقد_المقاولة",
    "القانون_المدني/العقود/عقد_الوكالة", "القانون_المدني/العقود/عقد_الكفالة", "القانون_المدني/العقود/عقد_الرهن", "القانون_المدني/العقود/عقد_الهبة",
    "القانون_المدني/المسؤولية_المدنية", "القانون_المدني/المسؤولية_المدنية/المسؤولية_العقدية", "القانون_المدني/المسؤولية_المدنية/المسؤولية_التقصيرية_(الفعل_الضار)",
    "القانون_المدني/إثبات_الحقوق_المدنية", "القانون_التجاري", "القانون_التجاري/الأعمال_التجارية_والتاجر", "القانون_التجاري/الشركات_التجارية",
    "القانون_التجاري/الشركات_التجارية/شركة_التضامن", "القانون_التجاري/الشركات_التجارية/شركة_التوصية_البسيطة", "القانون_التجاري/الشركات_التجارية/شركة_المحاصة",
    "القانون_التجاري/الشركات_التجارية/الشركة_ذات_المسؤولية_المحدودة", "القانون_التجاري/الشركات_التجارية/شركة_المساهمة", 
    "القانون_التجاري/الأوراق_التجارية_(شيك،_كمبيالة،_سند_لأمر)", "القانون_التجاري/العقود_التجارية_(نقل،_وكالة_بالعمولة،_سمسرة)",
    "القانون_التجاري/الإفلاس_والصلح_الواقي", "القانون_التجاري/الملكية_الفكرية_التجارية", "القانون_التجاري/الملكية_الفكرية_التجارية/العلامات_التجارية",
    "القانون_التجاري/الملكية_الفكرية_التجارية/الأسماء_التجارية", "القانون_التجاري/الملكية_الفكرية_التجارية/براءات_الاختراع_التجارية",
    "القانون_الجنائي_(قانون_العقوبات)", "القانون_الجنائي/الأحكام_العامة_للجريمة_والعقوبة", "القانون_الجنائي/الجريمة_(أركان_وأنواع)",
    "القانون_الجنائي/الجريمة/جرائم_الحدود", "القانون_الجنائي/الجريمة/جرائم_القصاص_والدية", "القانون_الجنائي/الجريمة/جرائم_التعزير",
    "القانون_الجنائي/الجريمة/الجرائم_الواقعة_على_الأشخاص", "القانون_الجنائي/الجريمة/الجرائم_الواقعة_على_الأموال", 
    "القانون_الجنائي/الجريمة/الجرائم_المخلة_بالثقة_العامة", "القانون_الجنائي/الجريمة/الجرائم_الواقعة_على_أمن_الدولة", "القانون_الجنائي/الجريمة/الجرائم_الإلكترونية",
    "القانون_الجنائي/العقوبة_(أنواع_وتطبيق)", "قانون_الأحوال_الشخصية", "قانون_الأحوال_الشخصية/الخطبة_والزواج_(أركان_وشروط)",
    "قانون_الأحوال_الشخصية/آثار_الزواج_(حقوق_وواجبات_الزوجين،_النفقة)", "قانون_الأحوال_الشخصية/الفرقة_بين_الزوجين_(طلاق،_خلع،_فسخ)",
    "قانون_الأحوال_الشخصية/النسب_والحضانة", "قانون_الأحوال_الشخصية/الولاية_على_النفس_والمال_(قاصر،_محجور_عليه)", "قانون_الأحوال_الشخصية/الوصية",
    "قانون_الأحوال_الشخصية/الميراث_(أسباب_وموانع،_أنصبة)", "قانون_الأحوال_الشخصية/الوقف", "القانون_الإداري", 
    "القانون_الإداري/التنظيم_الإداري_(مركزي_ولامركزي)", "القانون_الإداري/النشاط_الإداري_(قرار_إداري،_عقد_إداري،_مرفق_عام)",
    "القانون_الإداري/الوظيفة_العامة", "القانون_الإداري/الأموال_العامة", "القانون_الإداري/الرقابة_على_أعمال_الإدارة_(القضاء_الإداري)",
    "قانون_العمل", "قانون_العمل/عقد_العمل_الفردي", "قانون_العمل/حقوق_وواجبات_العامل_وصاحب_العمل", "قانون_العمل/الأجور_وساعات_العمل_والإجازات",
    "قانون_العمل/انتهاء_عقد_العمل_وتسوية_المنازعات_العمالية", "قانون_العمل/النقابات_العمالية_والتفاوض_الجماعي", 
    "قانون_العمل/الصحة_والسلامة_المهنية_وإصابات_العمل", "القانون_الدولي_العام", "القانون_الدولي_العام/مصادر_القانون_الدولي_(معاهدات،_عرف_دولي)",
    "القانون_الدولي_العام/أشخاص_القانون_الدولي_(دول،_منظمات_دولية)", "القانون_الدولي_العام/المسؤولية_الدولية", 
    "القانون_الدولي_العام/فض_المنازعات_الدولية", "القانون_الدولي_العام/قانون_البحار", "القانون_الدولي_الخاص", "القانون_الدولي_الخاص/الجنسية",
    "القانون_الدولي_الخاص/مركز_الأجانب", "القانون_الدولي_الخاص/تنازع_القوانين", "القانون_الدولي_الخاص/تنازع_الاختصاص_القضائي_الدولي",
    "القانون_الدولي_الخاص/تنفيذ_الأحكام_الأجنبية", "قانون_الملكية_الفكرية_(عام)", "قانون_الملكية_الفكرية/حق_المؤلف_والحقوق_المجاورة",
    "قانون_الملكية_الفكرية/براءات_الاختراع_والنماذج_الصناعية", "قانون_البيئة", "قانون_الاستثمار", "قانون_حماية_المستهلك",
    # --- 2. القانون الإجرائي ---
    "القانون_الإجرائي/الإجراءات_المدنية_والتجارية_(قانون_المرافعات)", 
    "القانون_الإجرائي/الإجراءات_المدنية_والتجارية/الدعوى_القضائية_(شروط_وإجراءات_رفعها)", "القانون_الإجرائي/الإجراءات_المدنية_والتجارية/الخصومة_وسيرها",
    "القانون_الإجرائي/الإجراءات_المدنية_والتجارية/الأحكام_القضائية_المدنية_(إصدارها_وطرق_الطعن_فيها)", 
    "القانون_الإجرائي/الإجراءات_المدنية_والتجارية/التنفيذ_الجبري", "القانون_الإجرائي/الإجراءات_الجنائية",
    "القانون_الإجرائي/الإجراءات_الجنائية/مرحلة_جمع_الاستدلالات_والتحقيق_الابتدائي", "القانون_الإجرائي/الإجراءات_الجنائية/النيابة_العامة_ودورها",
    "القانون_الإجرائي/الإجراءات_الجنائية/المحاكمة_الجنائية", "القانون_الإجرائي/الإجراءات_الجنائية/الأحكام_الجنائية_وطرق_الطعن_فيها",
    "القانون_الإجرائي/الإجراءات_الجنائية/تنفيذ_العقوبات", "القانون_الإجرائي/قواعد_الإثبات_(أحكام_عامة)", "القانون_الإجرائي/قواعد_الإثبات/عبء_الإثبات",
    "القانون_الإجرائي/قواعد_الإثبات/وسائل_الإثبات", "القانون_الإجرائي/قواعد_الإثبات/وسائل_الإثبات/الإثبات_بالكتابة_(أدلة_كتابية)",
    "القانون_الإجرائي/قواعد_الإثبات/وسائل_الإثبات/الإثبات_بالشهادة_(شهود)", "القانون_الإجرائي/قواعد_الإثبات/وسائل_الإثبات/القرائن_القانونية_والقضائية",
    "القانون_الإجرائي/قواعد_الإثبات/وسائل_الإثبات/الإقرار", "القانون_الإجرائي/قواعد_الإثبات/وسائل_الإثبات/اليمين_(حاسمة،_متممة)",
    "القانون_الإجرائي/قواعد_الإثبات/وسائل_الإثبات/المعاينة_والخبرة", "القانون_الإجرائي/قواعد_الإثبات/اليقين_والشك_(في_المواد_الجنائية_خاصة)",
    "القانون_الإجرائي/الطعون_في_الأحكام", "القانون_الإجرائي/الطعون_في_الأحكام/الاستئناف", 
    "القانون_الإجرائي/الطعون_في_الأحكام/النقض_(أو_التمييز_حسب_المحكمة_العليا)", "القانون_الإجرائي/الطعون_في_الأحكام/إعادة_النظر",
    "القانون_الإجرائي/الطعون_في_الأحكام/آجال_الطعن", "القانون_الإجرائي/المواعيد_الإجرائية_القانونية", "القانون_الإجرائي/الاختصاص_القضائي",
    "القانون_الإجرائي/الاختصاص_القضائي/الاختصاص_النوعي", "القانون_الإجرائي/الاختصاص_القضائي/الاختصاص_القيمي", 
    "القانون_الإجرائي/الاختصاص_القضائي/الاختصاص_المكاني", "القانون_الإجرائي/الاختصاص_القضائي/الاختصاص_الولائي",
    "القانون_الإجرائي/الصلح_والتحكيم_(كوسائل_بديلة)", "القانون_الإجرائي/تكاليف_ورسوم_التقاضي",
    # --- 3. الشريعة الإسلامية ومصادرها وأصول الفقه ---
    "الشريعة_الإسلامية_(كمصدر_رئيسي_للتشريع)", "الشريعة_الإسلامية/مقاصد_الشريعة_العامة", 
    "الشريعة_الإسلامية/مقاصد_الشريعة/درء_المفاسد_وجلب_المصالح", "الشريعة_الإسلامية/مقاصد_الشريعة/التيسير_ورفع_الحرج",
    "الشريعة_الإسلامية/الأحكام_الشرعية_الخمسة_(واجب،_مندوب،_مباح،_مكروه،_محرم)", "الشريعة_الإسلامية/الضرورات_تبيح_المحظورات_(مبدأ_الضرورة)",
    "الشريعة_الإسلامية/الربا_وتحريمه", "مصادر_التشريع_الإسلامي/الكتاب_(القرآن_الكريم)", "مصادر_التشريع_الإسلامي/السنة_النبوية",
    "مصادر_التشريع_الإسلامي/الإجماع", "مصادر_التشريع_الإسلامي/القياس", "أصول_الفقه_(كمنهجية_استنباط)",
    "أصول_الفقه/قواعد_التفسير_والتأويل_للنصوص", "أصول_الفقه/الاستحسان", "أصول_الفقه/المصالح_المرسلة", "أصول_الفقه/العرف",
    "أصول_الفقه/سد_الذرائع", "أصول_الفقه/الاستصحاب", "الفقه_الإسلامي_(كمنتج_اجتهادي)",
    # --- 4. مفاهيم قانونية عامة ومبادئ ---
    "المصلحة_العامة", "النظام_العام_والآداب", "مبدأ_المشروعية_(سيادة_القانون)", "مبدأ_المساواة_أمام_القانون",
    "حسن_النية_في_التعاملات", "التعسف_في_استعمال_الحق",
    # --- 5. الجهات الفاعلة والهيئات والوثائق ---
    "الجهات_القضائية_والقانونية/المحاكم_(بشكل_عام)", "الجهات_القضائية_والقانونية/المحكمة_الابتدائية", 
    "الجهات_القضائية_والقانونية/محكمة_الاستئناف", "الجهات_القضائية_والقانونية/المحكمة_العليا_(النقض)", "الجهات_القضائية_والقانونية/النيابة_العامة",
    "الجهات_القضائية_والقانونية/القضاة_(تعيينهم_واستقلالهم)", "الجهات_القضائية_والقانونية/المحامون_والمساعدة_القضائية",
    "الجهات_القضائية_والقانونية/هيئات_التحكيم", "الجهات_القضائية_والقانونية/كتاب_العدل_والتوثيق",
    "طبيعة_النص_القانوني/دستور", "طبيعة_النص_القانوني/قانون", "طبيعة_النص_القانوني/قرار_جمهوري_بقانون",
    "طبيعة_النص_القانوني/قرار_جمهوري_(تنفيذي)", "طبيعة_النص_القانوني/لائحة_تنفيذية_أو_تنظيمية", 
    "طبيعة_النص_القانوني/معاهدة_دولية_مصادق_عليها", "طبيعة_النص_القانوني/تعديل_تشريعي", "طبيعة_النص_القانوني/إلغاء_تشريعي_(صريح_أو_ضمني)"
]

LLM_PROVIDER_FOR_CLASSIFICATION = "google" 
LLM_MODEL_NAME_FOR_CLASSIFICATION = "gemini-2.5-flash-preview" 

# --- LLM Call Functions ---
async def llm_article_tagging(
    llm: ProductionGeminiChatModel,
    article_text: str, article_number: str, law_name: str,
    law_categories_of_parent_law: List[str], tag_vocabulary_list: List[str],
    model_name_for_call: str
) -> ArticleTaggingOutput:
    article_identifier = f"{law_name} - Art. {article_number}"
    logger.info(f"\\n[LLM CALL] Tagging Article: {article_identifier} using {model_name_for_call}")
    user_prompt_content = (
        f"Context: Law \"{law_name}\" (Categories: {json.dumps(law_categories_of_parent_law, ensure_ascii=False)}).\n"
        f"Article No: {article_number}\nText (first 1000 chars): \"{article_text[:1000]}\""
    )
    system_prompt = (
        "You are an expert legal AI specializing in Yemeni Law. Your primary task is to suggest highly relevant topical tags for a given law article, based on its text and the context of its parent law (name and classification categories).\n"
        "You MUST select **at least 3 tags, and up to a maximum of 7 tags,** for the article. Each tag you select MUST be an EXACT, VERBATIM string from the 'Official Tag Vocabulary' provided below. These tags are hierarchical, and the '/' character is part of the tag string itself and indicates this hierarchy.\n\n"
        "--- Official Tag Vocabulary ---\n"
        f"{json.dumps(tag_vocabulary_list, ensure_ascii=False, indent=2)}\n"
        "--- End of Official Tag Vocabulary ---\n\n"
        "Analyze the article text and its parent law context carefully. Choose the most specific and relevant tags from the vocabulary that accurately describe the core topics and legal concepts discussed in the article.\n"
        "Respond by calling the 'ArticleTaggingOutput' tool with your selected tags, a confidence score (0.0-1.0), and a brief reasoning **in Arabic** (قدم درجة ثقة وتعليلاً موجزاً باللغة العربية) justifying your choices."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}]
    tool_schema = ArticleTaggingOutput.model_json_schema()
    tools_payload = [{"type": "function", "function": {"name": ArticleTaggingOutput.__name__, "description": "Schema for article tags.", "parameters": tool_schema}}]
    tool_choice_payload = {"type": "function", "function": {"name": ArticleTaggingOutput.__name__}}

    logger.debug(f"  [LLM PRE-CALL] Article ID: {article_identifier}, Model: {model_name_for_call}")
    prompt_char_count = sum(len(msg.get("content", "")) for msg in messages if msg.get("content"))
    logger.debug(f"    Estimated Prompt Characters (sum of content fields): {prompt_char_count}")

    try:
        response = await llm.chat_completion_with_tools(
            messages=messages, tools=tools_payload, tool_choice=tool_choice_payload,
            model_override=model_name_for_call,
            user_metadata={"user_id": "script_article_tagging", "operation": "article_tagging"}
        )
        
        logger.debug(f"  [LLM POST-CALL] Article ID: {article_identifier} - LLM call completed.")
        if hasattr(response, 'usage') and response.usage is not None:
            logger.debug(f"    Response Usage (from response.usage): {response.usage}")
        else:
            logger.debug(f"    Response Usage: Attribute 'usage' not found or is None on response object.")
            
        if response.choices and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == ArticleTaggingOutput.__name__:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    if 'article_identifier' not in arguments: arguments['article_identifier'] = article_identifier
                    return ArticleTaggingOutput(**arguments)
                except (json.JSONDecodeError, ValidationError) as e:
                    return ArticleTaggingOutput(article_identifier=article_identifier, error_message=f"Parse/Validation Error: {e}. Args: {tool_call.function.arguments}")
        return ArticleTaggingOutput(article_identifier=article_identifier, error_message=f"LLM did not call tool. Resp: {response}")
    except Exception as e:
        logger.error(f"  [LLM CALL EXCEPTION] Article ID: {article_identifier} - Error during LLM call or processing response: {e}", exc_info=True)
        return ArticleTaggingOutput(article_identifier=article_identifier, error_message=f"LLM Call/Processing Exception: {e}")

async def llm_comment_tagging( # NEW FUNCTION
    llm: ProductionGeminiChatModel,
    comment_title: Optional[str], 
    comment_content: str, 
    comment_id_for_log: str, # For logging purposes
    tag_vocabulary_list: List[str],
    model_name_for_call: str
) -> CommentTaggingOutput:
    comment_identifier = comment_title if comment_title else comment_content[:75] + "..." # Create a useful identifier
    logger.info(f"\\n[LLM CALL] Tagging Comment: {comment_identifier} (ID: {comment_id_for_log}) using {model_name_for_call}")
    
    user_prompt_content = f"Comment Title: \"{comment_title if comment_title else 'N/A'}\"\n"
    user_prompt_content += f"Comment Content (first 1000 chars): \"{comment_content[:1000]}\""
    
    system_prompt = (
        "You are an expert legal AI specializing in Yemeni Law. Your task is to suggest highly relevant topical tags for a given legal comment or analysis, based on its title and content.\n"
        "You MUST select **at least 3 tags, and up to a maximum of 7 tags,** for the comment. Each tag you select MUST be an EXACT, VERBATIM string from the 'Official Tag Vocabulary' provided below. These tags are hierarchical, and the '/' character is part of the tag string itself and indicates this hierarchy.\n\n" # MODIFIED TAG COUNT
        "--- Official Tag Vocabulary ---\n"
        f"{json.dumps(tag_vocabulary_list, ensure_ascii=False, indent=2)}\n"
        "--- End of Official Tag Vocabulary ---\n\n"
        "Analyze the comment's title and content carefully. Choose the most specific and relevant tags from the vocabulary that accurately describe the core topics, legal concepts, or areas of law discussed in the comment.\n"
        "Respond by calling the 'CommentTaggingOutput' tool with your selected tags, a confidence score (0.0-1.0), and a brief reasoning **in Arabic** (قدم درجة ثقة وتعليلاً موجزاً باللغة العربية) justifying your choices."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}]
    tool_schema = CommentTaggingOutput.model_json_schema()
    tools_payload = [{"type": "function", "function": {"name": CommentTaggingOutput.__name__, "description": "Schema for comment tags.", "parameters": tool_schema}}]
    tool_choice_payload = {"type": "function", "function": {"name": CommentTaggingOutput.__name__}}

    logger.debug(f"  [LLM PRE-CALL] Comment ID: {comment_id_for_log}, Model: {model_name_for_call}")
    prompt_char_count = sum(len(msg.get("content", "")) for msg in messages if msg.get("content"))
    logger.debug(f"    Estimated Prompt Characters (sum of content fields): {prompt_char_count}")

    try:
        response = await llm.chat_completion_with_tools(
            messages=messages, tools=tools_payload, tool_choice=tool_choice_payload,
            model_override=model_name_for_call,
            user_metadata={"user_id": "script_comment_tagging", "operation": "comment_tagging"}
        )
        
        logger.debug(f"  [LLM POST-CALL] Comment ID: {comment_id_for_log} - LLM call completed.")
        if hasattr(response, 'usage') and response.usage is not None:
            logger.debug(f"    Response Usage (from response.usage): {response.usage}")
        else:
            logger.debug(f"    Response Usage: Attribute 'usage' not found or is None on response object.")
            
        if response.choices and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == CommentTaggingOutput.__name__:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    if 'comment_identifier' not in arguments: arguments['comment_identifier'] = comment_identifier
                    return CommentTaggingOutput(**arguments)
                except (json.JSONDecodeError, ValidationError) as e:
                    return CommentTaggingOutput(comment_identifier=comment_identifier, error_message=f"Parse/Validation Error: {e}. Args: {tool_call.function.arguments}")
        return CommentTaggingOutput(comment_identifier=comment_identifier, error_message=f"LLM did not call tool. Resp: {response}")
    except Exception as e:
        logger.error(f"  [LLM CALL EXCEPTION] Comment ID: {comment_id_for_log} - Error during LLM call or processing response: {e}", exc_info=True)
        return CommentTaggingOutput(comment_identifier=comment_identifier, error_message=f"LLM Call/Processing Exception: {e}")

# --- Database Interaction Functions ---
async def fetch_articles_for_tagging(db_client: SupabaseClient, batch_size: int, offset: int) -> List[Dict[str, Any]]:
    logger.info(f"\\n[DB] Fetching batch of up to {batch_size} articles for tagging (offset: {offset}, where suggested_tags IS NULL OR tags_need_review IS TRUE)...")
    try:
        response = await asyncio.to_thread(
            db_client.table("law_articles")
            .select("article_id, article_number, article_text, law_id, law(law_name, suggested_classification, suggested_sharia_influence, classification_llm_model)") 
            .or_("suggested_tags.is.null,tags_need_review.is.true") 
            .order("created_at", desc=False)
            .limit(batch_size)
            .offset(offset)
            .execute
        )
        if response.data:
            logger.info(f"  [DB] Fetched {len(response.data)} articles for this batch.")
            transformed = []
            for item in response.data:
                law_info = item.pop('law', {})
                item['law_name'] = law_info.get('law_name', "Unknown Law")
                item['parent_law_classification'] = law_info.get('suggested_classification', []) 
                item['parent_sharia_influence'] = law_info.get('suggested_sharia_influence') 
                item['parent_law_model'] = law_info.get('classification_llm_model') 
                transformed.append(item)
            return transformed
        else: logger.info("  [DB] No articles found needing tagging suggestions.")
        return response.data or []
    except Exception as e:
        logger.error(f"  [DB ERROR] Error fetching articles: {e}", exc_info=True); return []

async def fetch_comments_for_tagging(db_client: SupabaseClient, batch_size: int, offset: int) -> List[Dict[str, Any]]: # NEW FUNCTION
    logger.info(f"\\n[DB] Fetching batch of up to {batch_size} comments for tagging (offset: {offset}, where suggested_tags IS NULL OR tags_need_review IS TRUE)...")
    try:
        # Assuming 'comments' table has 'comment_id' as PK, 'title', 'content'
        # and the new 'suggested_tags', 'tags_need_review' columns.
        # The PK is actually 'article_id' but we'll use it as comment_id for this script's logic.
        response = await asyncio.to_thread(
            db_client.table("comments") 
            .select("article_id, title, content") # Using 'article_id' as the comment's unique ID
            .or_("suggested_tags.is.null,tags_need_review.is.true") 
            .order("created_at", desc=False) # Assuming comments have created_at
            .limit(batch_size)
            .offset(offset)
            .execute
        )
        if response.data:
            logger.info(f"  [DB] Fetched {len(response.data)} comments for this batch.")
            # No transformation needed if fields are directly used
            return response.data
        else: 
            logger.info("  [DB] No comments found needing tagging suggestions.")
        return response.data or []
    except Exception as e:
        logger.error(f"  [DB ERROR] Error fetching comments: {e}", exc_info=True); return []


async def update_article_suggestions_in_db(
    db_client: SupabaseClient, 
    article_id: str, 
    suggestions: ArticleTaggingOutput, 
    llm_model_name: str,
    parent_classification: Optional[List[str]], 
    parent_sharia: Optional[bool]
):
    logger.info(f"  [DB UPDATE] Writing suggestions for article_id {article_id}...")
    try:
        update_data = {
            "suggested_tags": suggestions.suggested_tags,
            "tagging_confidence": suggestions.confidence_score,
            "tagging_reasoning": suggestions.reasoning,
            "tagging_llm_model": llm_model_name,
            "tagging_processed_at": datetime.now(timezone.utc).isoformat(),
            "parent_law_classification": parent_classification, 
            "parent_law_sharia_influence": parent_sharia, 
            "tags_need_review": True 
        }
        auto_approve_tags = (
            not suggestions.error_message and
            suggestions.confidence_score is not None and suggestions.confidence_score >= 0.80 and
            suggestions.suggested_tags 
        )
        if auto_approve_tags:
            update_data["tags_need_review"] = False
            logger.info(f"    Article tagging for article_id {article_id} meets auto-approval criteria.")
        else:
            logger.info(f"    Article tagging for article_id {article_id} requires manual review.")
        if suggestions.error_message:
             update_data["tagging_reasoning"] = f"LLM Error: {suggestions.error_message}\\n\\nPrevious Reasoning: {suggestions.reasoning or ''}".strip()
        await asyncio.to_thread(
            db_client.table("law_articles").update(update_data).eq("article_id", article_id).execute
        )
        logger.info(f"    Successfully updated suggestions for article_id {article_id}.")
    except Exception as e:
        logger.error(f"    [DB UPDATE ERROR] Failed to update article_id {article_id}: {e}", exc_info=True)

async def update_comment_suggestions_in_db( # NEW FUNCTION
    db_client: SupabaseClient, 
    comment_db_id: str, # This is the 'article_id' from comments table, acting as comment_id
    suggestions: CommentTaggingOutput, 
    llm_model_name: str
):
    logger.info(f"  [DB UPDATE] Writing suggestions for comment_id {comment_db_id}...")
    try:
        update_data = {
            "suggested_tags": suggestions.suggested_tags,
            "tagging_confidence": suggestions.confidence_score,
            "tagging_reasoning": suggestions.reasoning,
            "tagging_llm_model": llm_model_name,
            "tagging_processed_at": datetime.now(timezone.utc).isoformat(),
            "tags_need_review": True 
        }
        auto_approve_tags = (
            not suggestions.error_message and
            suggestions.confidence_score is not None and suggestions.confidence_score >= 0.80 and
            suggestions.suggested_tags
        )
        if auto_approve_tags:
            update_data["tags_need_review"] = False
            logger.info(f"    Comment tagging for comment_id {comment_db_id} meets auto-approval criteria.")
        else:
            logger.info(f"    Comment tagging for comment_id {comment_db_id} requires manual review.")
        if suggestions.error_message:
             update_data["tagging_reasoning"] = f"LLM Error: {suggestions.error_message}\\n\\nPrevious Reasoning: {suggestions.reasoning or ''}".strip()
        
        # Using 'article_id' as the PK for comments table as per schema investigation
        await asyncio.to_thread(
            db_client.table("comments").update(update_data).eq("article_id", comment_db_id).execute
        )
        logger.info(f"    Successfully updated suggestions for comment_id {comment_db_id}.")
    except Exception as e:
        logger.error(f"    [DB UPDATE ERROR] Failed to update comment_id {comment_db_id}: {e}", exc_info=True)


async def process_article_concurrently(
    article_data: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    llm_classifier_instance: ProductionGeminiChatModel,
    supabase_client_instance: SupabaseClient,
    tag_vocab: List[str],
    llm_model_name_for_call: str,
    db_model_id_str: str
):
    async with semaphore: 
        article_id = article_data.get("article_id")
        article_identifier_for_log = f"{article_data.get('law_name', 'Unknown Law')} - Art. {article_data.get('article_number', 'N/A')}"
        logger.info(f"  [CONCURRENT_TASK_START] Processing article: {article_identifier_for_log}")
        suggestion = None
        try:
            suggestion = await llm_article_tagging(
                llm_classifier_instance, 
                article_data.get("article_text", ""), 
                str(article_data.get("article_number", "N/A")),
                article_data.get("law_name", "Unknown Law"),
                article_data.get("parent_law_classification", []), 
                tag_vocab, 
                llm_model_name_for_call
            )
        except Exception as e_llm:
            logger.error(f"  [CONCURRENT_LLM_ERROR] LLM call failed for {article_identifier_for_log}: {e_llm}", exc_info=True)
            suggestion = ArticleTaggingOutput(
                article_identifier=article_identifier_for_log,
                error_message=f"LLM Exception in concurrent task: {str(e_llm)}"
            )
        if article_id and suggestion: 
            try:
                await update_article_suggestions_in_db(
                    supabase_client_instance, article_id, suggestion, db_model_id_str,
                    article_data.get("parent_law_classification", []), 
                    article_data.get("parent_sharia_influence") 
                )
            except Exception as e_db:
                logger.error(f"  [CONCURRENT_DB_ERROR] DB update failed for {article_identifier_for_log} (ID: {article_id}): {e_db}", exc_info=True)
        logger.info(f"  [CONCURRENT_TASK_END] Finished processing article: {article_identifier_for_log}")
        return article_id 

async def process_comment_concurrently( # NEW FUNCTION
    comment_data: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    llm_tagger_instance: ProductionGeminiChatModel,
    supabase_client_instance: SupabaseClient,
    tag_vocab: List[str],
    llm_model_name_for_call: str,
    db_model_id_str: str
):
    async with semaphore:
        # Using 'article_id' as the comment's unique ID from the DB table
        comment_db_id = comment_data.get("article_id") 
        comment_title = comment_data.get("title")
        comment_content = comment_data.get("content", "")
        comment_identifier_for_log = f"Comment (ID: {comment_db_id}, Title: {comment_title if comment_title else 'N/A'})"
        
        logger.info(f"  [CONCURRENT_TASK_START] Processing comment: {comment_identifier_for_log}")
        suggestion = None
        try:
            suggestion = await llm_comment_tagging(
                llm_tagger_instance,
                comment_title,
                comment_content,
                comment_db_id, # Pass the actual DB ID for logging inside llm_comment_tagging
                tag_vocab,
                llm_model_name_for_call
            )
        except Exception as e_llm:
            logger.error(f"  [CONCURRENT_LLM_ERROR] LLM call failed for {comment_identifier_for_log}: {e_llm}", exc_info=True)
            suggestion = CommentTaggingOutput(
                comment_identifier=comment_identifier_for_log, # Use the broader identifier for the output object
                error_message=f"LLM Exception in concurrent task: {str(e_llm)}"
            )
        
        if comment_db_id and suggestion:
            try:
                await update_comment_suggestions_in_db(
                    supabase_client_instance,
                    comment_db_id,
                    suggestion,
                    db_model_id_str
                )
            except Exception as e_db:
                logger.error(f"  [CONCURRENT_DB_ERROR] DB update failed for {comment_identifier_for_log}: {e_db}", exc_info=True)
        
        logger.info(f"  [CONCURRENT_TASK_END] Finished processing comment: {comment_identifier_for_log}")
        return comment_db_id

# --- Main PoC Logic ---
async def main_poc():
    if not (KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY):
        logger.error("Supabase URL or Key not found. Exiting."); return
    try:
        supabase_client: SupabaseClient = create_client(KNOWLEDGE_SUPABASE_URL, KNOWLEDGE_SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}. Exiting.", exc_info=True); return

    llm_processor = ProductionGeminiChatModel() # Single instance for both articles and comments
    model_id_for_db = LLM_MODEL_NAME_FOR_CLASSIFICATION 
    logger.info(f"LLM for tagging initialized (Intended Model: {model_id_for_db}).")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    # --- Tag Articles --- (Temporarily Commented Out to Focus on Comments)
    # logger.info("\\n\\n--- Starting Article Tagging ---")
    # article_offset = 0
    # total_articles_processed_successfully = 0
    
    # while True:
    #     logger.info(f"Fetching article DB batch with offset {article_offset} (Batch Size: {BATCH_SIZE})...")
    #     articles_to_process_from_db_batch = await fetch_articles_for_tagging(supabase_client, BATCH_SIZE, article_offset)
        
    #     if not articles_to_process_from_db_batch:
    #         logger.info("No more articles to tag.")
    #         break

    #     logger.info(f"DB Batch Fetched. Processing {len(articles_to_process_from_db_batch)} articles for tagging (Concurrency: {MAX_CONCURRENT_TASKS})...")
        
    #     article_tasks = []
    #     for article_data_item in articles_to_process_from_db_batch:
    #         article_tasks.append(process_article_concurrently(
    #             article_data=article_data_item,
    #             semaphore=semaphore,
    #             llm_classifier_instance=llm_processor, # Use the single llm_processor instance
    #             supabase_client_instance=supabase_client,
    #             tag_vocab=SHARED_TAG_VOCABULARY, # Use shared vocabulary
    #             llm_model_name_for_call=LLM_MODEL_NAME_FOR_CLASSIFICATION,
    #             db_model_id_str=model_id_for_db
    #         ))
        
    #     processed_articles_in_batch = 0
    #     for task_future in async_tqdm(asyncio.as_completed(article_tasks), total=len(article_tasks), desc=f"Tagging Articles (Offset {article_offset})"):
    #         try:
    #             completed_article_id = await task_future
    #             if completed_article_id: 
    #                 processed_articles_in_batch += 1
    #         except Exception as e_concurrent_task:
    #             logger.error(f"  [MAIN_LOOP_ERROR] A concurrent article processing task raised an unhandled exception: {e_concurrent_task}", exc_info=True)
        
    #     logger.info(f"Finished processing article DB batch (Offset {article_offset}). Articles attempted: {len(articles_to_process_from_db_batch)}, Successfully initiated DB update for: {processed_articles_in_batch}.")
    #     total_articles_processed_successfully += processed_articles_in_batch
    #     article_offset += BATCH_SIZE
    # logger.info(f"--- Article Tagging Finished. Total articles for which DB update was initiated: {total_articles_processed_successfully} ---")

    # --- Tag Comments --- NEW SECTION ---
    logger.info("\\n\\n--- Starting Comment Tagging ---")
    comment_offset = 0
    total_comments_processed_successfully = 0
    
    while True:
        logger.info(f"Fetching comment DB batch with offset {comment_offset} (Batch Size: {BATCH_SIZE})...")
        comments_to_process_from_db_batch = await fetch_comments_for_tagging(supabase_client, BATCH_SIZE, comment_offset)
        
        if not comments_to_process_from_db_batch:
            logger.info("No more comments to tag.")
            break

        logger.info(f"DB Batch Fetched. Processing {len(comments_to_process_from_db_batch)} comments for tagging (Concurrency: {MAX_CONCURRENT_TASKS})...")
        
        comment_tasks = []
        for comment_data_item in comments_to_process_from_db_batch:
            comment_tasks.append(process_comment_concurrently(
                comment_data=comment_data_item,
                semaphore=semaphore,
                llm_tagger_instance=llm_processor, # Use the single llm_processor instance
                supabase_client_instance=supabase_client,
                tag_vocab=SHARED_TAG_VOCABULARY, # Use shared vocabulary
                llm_model_name_for_call=LLM_MODEL_NAME_FOR_CLASSIFICATION,
                db_model_id_str=model_id_for_db
            ))
            
        processed_comments_in_batch = 0
        for task_future in async_tqdm(asyncio.as_completed(comment_tasks), total=len(comment_tasks), desc=f"Tagging Comments (Offset {comment_offset})"):
            try:
                completed_comment_id = await task_future
                if completed_comment_id:
                    processed_comments_in_batch += 1
            except Exception as e_concurrent_task:
                logger.error(f"  [MAIN_LOOP_ERROR] A concurrent comment processing task raised an unhandled exception: {e_concurrent_task}", exc_info=True)

        logger.info(f"Finished processing comment DB batch (Offset {comment_offset}). Comments attempted: {len(comments_to_process_from_db_batch)}, Successfully initiated DB update for: {processed_comments_in_batch}.")
        total_comments_processed_successfully += processed_comments_in_batch
        comment_offset += BATCH_SIZE
    logger.info(f"--- Comment Tagging Finished. Total comments for which DB update was initiated: {total_comments_processed_successfully} ---")


if __name__ == "__main__":
    asyncio.run(main_poc())
