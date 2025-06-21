from typing import Dict, Any, Tuple
# Assuming your config classes are in src.config.app_config
# Adjust the import path if your project structure is different
# or if get_app_config is the intended way to get the config object.
from src.config.app_config import ProductionConfig, StagingConfig, DevelopmentConfig, get_app_config
import logging

logger = logging.getLogger(__name__)

class StrategySelector:
    """Intelligently select strategy and priority based on query characteristics."""
    
    def __init__(self, app_env: str = None):
        """
        Initializes the StrategySelector.
        Args:
            app_env: Optional string to force a specific environment ('production', 'staging', 'development').
                     If None, it will be determined by the APP_ENV environment variable.
        """
        if app_env:
            current_env = app_env.lower()
            if current_env == "production":
                self.config = ProductionConfig()
            elif current_env == "staging":
                self.config = StagingConfig()
            else: # Default to development if app_env is specified but not recognized
                self.config = DevelopmentConfig()
        else: # Determine from APP_ENV
            self.config = get_app_config() 
        
        logger.info(f"StrategySelector initialized with config for environment: {self.config.__class__.__name__}")

    def select_strategy(self, query_analysis: Dict[str, Any]) -> Tuple[str, str]:
        """
        Select strategy and priority based on query analysis.
        
        Args:
            query_analysis: A dictionary containing characteristics of the query.
                            Expected keys might include 'complexity', 'urgency', 'query_type'.
        
        Returns:
            tuple: (strategy_name, priority_level)
        """
        
        complexity = query_analysis.get("complexity", "medium") # Default if not provided
        urgency = query_analysis.get("urgency", "normal")
        query_type = query_analysis.get("query_type", "general") # e.g., 'simple_definition', 'complex_legal_analysis'

        strategy = self.config.STRATEGY_MAPPING.get("default_strategy", "fallback") 
        priority = self.config.PRIORITY_MAPPING.get("default_priority", "normal")

        # More specific strategy selection based on query_type or other analysis
        if query_type == "simple_definition" or query_type == "basic_lookup":
            strategy = self.config.STRATEGY_MAPPING.get("simple_query", strategy)
        elif query_type in ["complex_legal_analysis", "multi_law_comparison"] or complexity == "high":
            strategy = self.config.STRATEGY_MAPPING.get("complex_analysis", strategy)
        
        # Urgency can override priority
        if urgency == "critical" or query_type == "court_deadline": # Example of urgent query_type
            strategy = self.config.STRATEGY_MAPPING.get("urgent_legal", strategy) # Use premium for urgent
            priority = self.config.PRIORITY_MAPPING.get("critical", priority)
        elif urgency == "high":
            priority = self.config.PRIORITY_MAPPING.get("urgent", priority) # 'urgent' maps to 'high' priority

        logger.debug(f"Selected strategy: '{strategy}', priority: '{priority}' for query_analysis: {query_analysis}")
        return strategy, priority
    
    def get_timeout_config(self, strategy: str) -> Dict[str, int]:
        """Get timeout and retry configuration based on strategy."""
        # These are example timeouts; adjust based on expected performance and cost.
        # These could also be part of the environment-specific config classes.
        timeout_configs = {
            "premium": {"timeout": 60, "retry_attempts": 3}, # Longer timeout for premium/complex
            "fallback": {"timeout": 45, "retry_attempts": 2}, 
            "loadbalance": {"timeout": 30, "retry_attempts": 1},
            "direct_gemini_flash": {"timeout": 30, "retry_attempts": 1}, # Example for a dev strategy
            "default_strategy": {"timeout": 45, "retry_attempts": 2} # Fallback timeout
        }
        selected_config = timeout_configs.get(strategy, timeout_configs["default_strategy"])
        logger.debug(f"Timeout config for strategy '{strategy}': {selected_config}")
        return selected_config

# Example Usage (for testing the selector independently):
if __name__ == '__main__':
    # Test with different environments by setting APP_ENV or passing app_env to constructor
    # os.environ["APP_ENV"] = "production" 
    selector = StrategySelector() # Defaults to APP_ENV or "development"
    
    test_queries_analysis = [
        {"complexity": "low", "urgency": "normal", "query_type": "simple_definition"},
        {"complexity": "high", "urgency": "normal", "query_type": "complex_legal_analysis"},
        {"complexity": "medium", "urgency": "critical", "query_type": "court_deadline"},
        {"query_type": "general"}, # Test defaults
    ]

    for analysis in test_queries_analysis:
        strat, prio = selector.select_strategy(analysis)
        timeout_conf = selector.get_timeout_config(strat)
        print(f"Analysis: {analysis} -> Strategy: {strat}, Priority: {prio}, Timeout: {timeout_conf}")
