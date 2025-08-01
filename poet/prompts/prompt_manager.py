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

@dataclass
class PromptTemplate:
    """Represents a single prompt template"""
    name: str
    description: str
    template: str
    category: PromptCategory
    parameters: List[str]
    metadata: Dict[str, Any]
    
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
    """
    
    def __init__(self, prompts_dir: Optional[str] = None):
        self.prompts_dir = Path(prompts_dir or self._get_default_prompts_dir())
        self._templates: Dict[str, PromptTemplate] = {}
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
        
        template = PromptTemplate(
            name=data['name'],
            description=data['description'],
            template=data['template'],
            category=category,
            parameters=parameters,
            metadata=data.get('metadata', {})
        )
        
        self._templates[template.name] = template
    
    def _extract_parameters(self, template_str: str) -> List[str]:
        """Extract parameter names from template string"""
        import re
        # Find all {parameter_name} patterns
        parameters = re.findall(r'\{(\w+)\}', template_str)
        return list(set(parameters))  # Remove duplicates
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a template by name"""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        return self._templates[name]
    
    def get_templates_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        """Get all templates in a specific category"""
        return [t for t in self._templates.values() if t.category == category]
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self._templates.keys())
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with parameters"""
        template = self.get_template(template_name)
        return template.format(**kwargs)
    
    def reload_templates(self):
        """Reload all templates from disk"""
        self._templates.clear()
        self._load_all_templates()
    
    def add_template(self, template: PromptTemplate):
        """Add a template programmatically"""
        self._templates[template.name] = template
    
    def validate_template(self, template_name: str, **kwargs) -> bool:
        """Validate that all required parameters are provided for a template"""
        try:
            template = self.get_template(template_name)
            missing_params = set(template.parameters) - set(kwargs.keys())
            return len(missing_params) == 0
        except KeyError:
            return False
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        template = self.get_template(template_name)
        return {
            'name': template.name,
            'description': template.description,
            'category': template.category.value,
            'parameters': template.parameters,
            'metadata': template.metadata
        }

# Convenience functions for common prompt operations

def load_prompt_manager(prompts_dir: Optional[str] = None) -> PromptManager:
    """Factory function to create and load a prompt manager"""
    return PromptManager(prompts_dir)

def format_constraint_inference_prompt(user_prompt: str, constraint_type: str) -> str:
    """Quick function to format constraint inference prompts"""
    pm = PromptManager()
    template_name = f"{constraint_type}_inference"
    return pm.format_prompt(template_name, user_prompt=user_prompt)

def format_generation_prompt(context: Dict[str, Any], generation_type: str) -> str:
    """Quick function to format generation prompts"""
    pm = PromptManager()
    template_name = f"{generation_type}_generation"
    return pm.format_prompt(template_name, **context)


