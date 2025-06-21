import os

class ProductionConfig:
    """Production configuration with fallback and load balancing"""
    
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    
    # Config IDs (set these in your environment)
    FALLBACK_CONFIG_ID = os.getenv("FALLBACK_CONFIG_ID") # e.g., "pc-yemeni-409276"
    LOADBALANCE_CONFIG_ID = os.getenv("LOADBALANCE_CONFIG_ID") # e.g., "pc-yemeni-f0ac29"
    PREMIUM_CONFIG_ID = os.getenv("PREMIUM_CONFIG_ID") # e.g., "pc-yemeni-d8455b"
    
    # Virtual Keys (can also be loaded from env if they change per environment)
    GEMINI_FLASH_VIRTUAL_KEY = os.getenv("GOOGLE_VIRTUAL_KEY_ID", "google-virtual-10b6cf") # Default from your .env
    GEMINI_PRO_VIRTUAL_KEY = os.getenv("GEMINI_PRO_VIRTUAL_KEY", "gemini-pro-e6d800") # Default from your .env
    # OPENAI_EMBEDDING_VIRTUAL_KEY = os.getenv("OPENAI_VIRTUAL_KEY") # For embeddings, if needed here

    # Strategy selection mapping (can be more dynamic)
    STRATEGY_MAPPING = {
        "simple_query": "loadbalance",
        "complex_analysis": "fallback", 
        "urgent_legal": "premium", # This will use PREMIUM_CONFIG_ID or dynamic premium in ProductionGeminiChatModel
        "batch_processing": "loadbalance",
        "default_strategy": "fallback" 
    }
    
    # Priority mapping
    PRIORITY_MAPPING = {
        "background": "low",
        "user_query": "normal",
        "urgent": "high",
        "critical": "critical",
        "default_priority": "normal"
    }

class StagingConfig(ProductionConfig): # Inherits from Production, can override
    """Staging configuration - can be simpler or use different Config IDs"""
    FALLBACK_CONFIG_ID = os.getenv("STAGING_FALLBACK_CONFIG_ID", ProductionConfig.FALLBACK_CONFIG_ID)
    LOADBALANCE_CONFIG_ID = os.getenv("STAGING_LOADBALANCE_CONFIG_ID", ProductionConfig.LOADBALANCE_CONFIG_ID)
    PREMIUM_CONFIG_ID = os.getenv("STAGING_PREMIUM_CONFIG_ID", ProductionConfig.PREMIUM_CONFIG_ID)
    # Override strategies if needed for staging
    # STRATEGY_MAPPING = { ... } 

class DevelopmentConfig(ProductionConfig): # Inherits, can override
    """Development configuration - perhaps single provider or specific debug configs"""
    # Example: Dev might always use a specific virtual key directly or a simpler fallback
    FALLBACK_CONFIG_ID = os.getenv("DEV_FALLBACK_CONFIG_ID", ProductionConfig.FALLBACK_CONFIG_ID) 
    # Or, for very simple dev, might not use Portkey configs:
    # FALLBACK_CONFIG_ID = None 
    # LOADBALANCE_CONFIG_ID = None
    # PREMIUM_CONFIG_ID = None
    # STRATEGY_MAPPING = {"default_strategy": "direct_gemini_flash"} # Custom strategy for dev

def get_app_config():
    """Returns the appropriate config based on APP_ENV environment variable."""
    app_env = os.getenv("APP_ENV", "development").lower()
    if app_env == "production":
        return ProductionConfig()
    elif app_env == "staging":
        return StagingConfig()
    else: # Default to development
        return DevelopmentConfig()

# Example of how you might get the current configuration
# current_app_config = get_app_config()
