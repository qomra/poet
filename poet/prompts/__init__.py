# poet/prompts/__init__.py

from .prompt_manager import PromptManager

# Create a global prompt manager instance
# This will be the single source of truth for all prompt templates
_global_prompt_manager = None

def get_global_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _global_prompt_manager
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager()
    return _global_prompt_manager

def set_global_prompt_manager_language(language: str):
    """Set the language for the global prompt manager."""
    global _global_prompt_manager
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager()
    _global_prompt_manager.set_default_language(language)

def initialize_global_prompt_manager(prompts_dir: str = None, default_language: str = "arabic"):
    """Initialize the global prompt manager with specific settings."""
    global _global_prompt_manager
    _global_prompt_manager = PromptManager(prompts_dir, default_language)
    return _global_prompt_manager

# Convenience imports
__all__ = [
    'PromptManager',
    'get_global_prompt_manager',
    'set_global_prompt_manager_language',
    'initialize_global_prompt_manager'
] 