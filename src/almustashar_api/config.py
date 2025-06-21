import os
from dotenv import load_dotenv

# Construct the path to the .env file, assuming it's in the project root
# (one level up from src/, then one level up from almustashar_api/)
# This might need adjustment based on where uvicorn is run from.
# For now, let's assume .env is at the project root relative to this config file.
# Project Root -> src -> almustashar_api -> config.py
# So, .env is ../../.env from config.py
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO:     .env file loaded successfully from: {dotenv_path}")
else:
    # Fallback if .env is next to the main.py or where the app is run from
    dotenv_path_alt = os.path.join(os.path.dirname(__file__), ".env") # if .env is in almustashar_api
    if os.path.exists(dotenv_path_alt):
        load_dotenv(dotenv_path=dotenv_path_alt)
        print(f"INFO:     .env file loaded successfully from: {dotenv_path_alt}")
    else:
        print("WARNING:  .env file not found at primary or alternative locations. Environment variables must be set externally.")

# API Keys and Settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # Assuming you might use Pinecone directly or via agent
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # For embeddings or other OpenAI models

# Supabase settings (for auth or direct client use if needed by API, not just agent)
USERS_SUPABASE_URL = os.getenv("USERS_SUPABASE_URL")
USERS_SUPABASE_KEY = os.getenv("USERS_SUPABASE_KEY") # Typically the anon key for client-side, or service_role for server
KNOWLEDGE_SUPABASE_URL = os.getenv("KNOWLEDGE_SUPABASE_URL")
KNOWLEDGE_SUPABASE_KEY = os.getenv("KNOWLEDGE_SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # For admin/service operations

# CORS settings
CORS_ALLOWED_ORIGINS_STR = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,http://192.168.8.155:8000") # Default for local dev, added phone IP
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in CORS_ALLOWED_ORIGINS_STR.split(',')]

# Application settings
API_TITLE = "Almustashar Agent API"
API_VERSION = "0.1.0"

# Logging configuration (can be expanded)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Placeholder for other configurations
# Example: DEFAULT_PIPELINE_NAME = "default_almustashar_pipeline"

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment.")
# Add more checks for critical env vars if necessary

print(f"INFO:     Config loaded. CORS Origins: {CORS_ALLOWED_ORIGINS}")
