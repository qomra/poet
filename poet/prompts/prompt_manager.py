# poet/prompts/prompt_manager.py

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class PromptCategory(Enum):
    """Categories of prompts for different tasks"""
    CONSTRAINT_INFERENCE = "constraint_inference"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    REFINEMENT = "refinement"
    PLANNING = "planning"
    SEARCH = "search"
    ANALYSIS = "analysis"
    COMPILER = "compiler"

@dataclass
class PromptTemplate:
    """Represents a single prompt template"""
    name: str
    description: str
    template: str
    category: PromptCategory
    parameters: List[str]
    metadata: Dict[str, Any]
    language: str = "arabic"  # Default to Arabic
    
    def format(self, **kwargs) -> str:
        """Format the template with provided parameters"""
        # Validate required parameters
        missing_params = set(self.parameters) - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        return self.template.format(**kwargs)

class PromptManager:
    """
    Manages prompt templates for different tasks in the poetry generation pipeline.
    
    Handles loading, caching, and formatting of prompt templates from YAML files.
    Provides a centralized way to manage all LLM interactions across the system.
    Supports multiple languages (Arabic and English).
    """
    
    def __init__(self, prompts_dir: Optional[str] = None, default_language: str = "arabic"):
        self.prompts_dir = Path(prompts_dir or self._get_default_prompts_dir())
        self.default_language = default_language
        self._templates: Dict[str, PromptTemplate] = {}
        self._language_templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self._load_all_templates()
    
    def _get_default_prompts_dir(self) -> str:
        """Get default prompts directory path"""
        current_file = Path(__file__).parent
        return current_file / "templates"
    
    def _load_all_templates(self):
        """Load all prompt templates from the templates directory"""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        # Walk through all subdirectories and load YAML files
        for yaml_file in self.prompts_dir.rglob("*.yaml"):
            try:
                self._load_template_file(yaml_file)
            except Exception as e:
                print(f"Warning: Failed to load template {yaml_file}: {e}")
    
    def _load_template_file(self, yaml_file: Path):
        """Load a single template file"""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Determine category from directory structure
        category_name = yaml_file.parent.name
        try:
            category = PromptCategory(category_name)
        except ValueError:
            # If parent directory doesn't match enum, try grandparent
            category_name = yaml_file.parent.parent.name
            category = PromptCategory(category_name)
        
        # Extract parameters from template
        parameters = self._extract_parameters(data.get('template', ''))
        
        # Determine language from filename or metadata
        language = self._determine_language(yaml_file, data)
        
        template = PromptTemplate(
            name=data['name'],
            description=data['description'],
            template=data['template'],
            category=category,
            parameters=parameters,
            metadata=data.get('metadata', {}),
            language=language
        )
        
        # Store in main templates dict
        self._templates[template.name] = template
        
        # Store in language-specific dict
        if language not in self._language_templates:
            self._language_templates[language] = {}
        self._language_templates[language][template.name] = template
    
    def _determine_language(self, yaml_file: Path, data: Dict[str, Any]) -> str:
        """Determine the language of a template from filename or metadata"""
        # Check metadata first
        if 'metadata' in data and 'language' in data['metadata']:
            return data['metadata']['language']
        
        # Check filename for language indicators
        filename = yaml_file.stem
        if filename.endswith('_en'):
            return 'english'
        elif filename.endswith('_ar'):
            return 'arabic'
        
        # Default to Arabic for backward compatibility
        return 'arabic'
    
    def _extract_parameters(self, template_str: str) -> List[str]:
        """Extract parameter names from template string"""
        import re
        # Find all {parameter_name} patterns
        parameters = re.findall(r'\{(\w+)\}', template_str)
        return list(set(parameters))  # Remove duplicates
    
    def _get_language_specific_template_name(self, base_name: str, language: str) -> str:
        """Get the language-specific template name for a base template name"""
        if language == "english":
            return f"{base_name}_en"
        elif language == "arabic":
            return base_name
        return base_name
    
    def get_template(self, name: str, language: Optional[str] = None) -> PromptTemplate:
        """Get a template by name, optionally specifying language preference"""
        target_language = language or self.default_language
        
        # First, try to get language-specific template
        if target_language in self._language_templates and name in self._language_templates[target_language]:
            return self._language_templates[target_language][name]
        
        # If not found, try to get the language-specific version of the template name
        lang_specific_name = self._get_language_specific_template_name(name, target_language)
        if target_language in self._language_templates and lang_specific_name in self._language_templates[target_language]:
            return self._language_templates[target_language][lang_specific_name]
        
        # If still not found, try to find any template with that name
        # but prioritize the default language if available
        available_templates = []
        for lang, templates in self._language_templates.items():
            if name in templates:
                available_templates.append((lang, templates[name]))
            # Also check for language-specific versions
            lang_specific_name = self._get_language_specific_template_name(name, lang)
            if lang_specific_name in templates:
                available_templates.append((lang, templates[lang_specific_name]))
        
        if available_templates:
            # Sort by priority: default language first, then others
            available_templates.sort(key=lambda x: (0 if x[0] == self.default_language else 1, x[0]))
            return available_templates[0][1]
        
        # Fallback to main templates dict
        if name in self._templates:
            return self._templates[name]
        
        raise KeyError(f"Template '{name}' not found for language '{target_language}'")
    
    def get_templates_by_category(self, category: PromptCategory, language: Optional[str] = None) -> List[PromptTemplate]:
        """Get all templates in a specific category, optionally filtered by language"""
        templates = [t for t in self._templates.values() if t.category == category]
        
        if language:
            templates = [t for t in templates if t.language == language]
        
        return templates
    
    def get_templates_by_language(self, language: str) -> List[PromptTemplate]:
        """Get all templates for a specific language"""
        if language in self._language_templates:
            return list(self._language_templates[language].values())
        return []
    
    def list_templates(self, language: Optional[str] = None) -> List[str]:
        """List all available template names, optionally filtered by language"""
        if language:
            return list(self._language_templates.get(language, {}).keys())
        return list(self._templates.keys())
    
    def format_prompt(self, template_name: str, language: Optional[str] = None, **kwargs) -> str:
        """Format a prompt template with parameters, optionally specifying language"""
        template = self.get_template(template_name, language)
        return template.format(**kwargs)
    
    def reload_templates(self):
        """Reload all templates from disk"""
        self._templates.clear()
        self._language_templates.clear()
        self._load_all_templates()
    
    def add_template(self, template: PromptTemplate):
        """Add a template programmatically"""
        self._templates[template.name] = template
        
        # Add to language-specific dict
        if template.language not in self._language_templates:
            self._language_templates[template.language] = {}
        self._language_templates[template.language][template.name] = template
    
    def validate_template(self, template_name: str, language: Optional[str] = None, **kwargs) -> bool:
        """Validate that all required parameters are provided for a template"""
        try:
            template = self.get_template(template_name, language)
            missing_params = set(template.parameters) - set(kwargs.keys())
            return len(missing_params) == 0
        except KeyError:
            return False
    
    def get_template_info(self, template_name: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a template"""
        template = self.get_template(template_name, language)
        return {
            'name': template.name,
            'description': template.description,
            'category': template.category.value,
            'parameters': template.parameters,
            'metadata': template.metadata,
            'language': template.language
        }
    
    def set_default_language(self, language: str):
        """Set the default language for template retrieval"""
        self.default_language = language
    
    def get_available_languages(self) -> List[str]:
        """Get list of available languages"""
        return list(self._language_templates.keys())

# Convenience functions for common prompt operations

def load_prompt_manager(prompts_dir: Optional[str] = None, default_language: str = "arabic") -> PromptManager:
    """Factory function to create and load a prompt manager"""
    return PromptManager(prompts_dir, default_language)

def format_constraint_inference_prompt(user_prompt: str, constraint_type: str, language: Optional[str] = None) -> str:
    """Quick function to format constraint inference prompts"""
    pm = PromptManager()
    template_name = f"{constraint_type}_inference"
    return pm.format_prompt(template_name, language=language, user_prompt=user_prompt)

def format_generation_prompt(context: Dict[str, Any], generation_type: str, language: Optional[str] = None) -> str:
    """Quick function to format generation prompts"""
    pm = PromptManager()
    template_name = f"{generation_type}_generation"
    return pm.format_prompt(template_name, language=language, **context)


