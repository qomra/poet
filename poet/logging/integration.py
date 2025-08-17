from typing import Optional, Dict, Any
from pathlib import Path
from poet.logging.harmony_capture import get_capture, capture_method
from poet.compiler.harmony import HarmonyCompiler
from poet.llm.base_llm import BaseLLM

class HarmonyIntegration:
    """
    Integration helper for adding Harmony capture to existing pipeline
    """
    
    @staticmethod
    def instrument_component(component, component_name: str, call_type: str):
        """
        Dynamically instrument a component's methods with capture decorators
        
        Usage:
            parser = ConstraintParser(llm)
            HarmonyIntegration.instrument_component(parser, "ConstraintParser", "parse")
        """
        # Get all methods that don't start with underscore
        for attr_name in dir(component):
            if not attr_name.startswith('_'):
                attr = getattr(component, attr_name)
                if callable(attr):
                    # Wrap the method with capture decorator
                    wrapped = capture_method(component_name, call_type)(attr)
                    setattr(component, attr_name, wrapped)
    
    @staticmethod
    def start_captured_execution(user_prompt: str, constraints: Dict[str, Any] = None):
        """Start a new captured execution"""
        capture = get_capture()
        
        # Convert Constraints object to dict if needed
        if constraints is not None and hasattr(constraints, 'to_dict'):
            constraints = constraints.to_dict()
        
        return capture.start_execution(user_prompt, constraints)
    
    @staticmethod
    def complete_and_reason(llm: BaseLLM, final_poem: Any = None, 
                           quality_assessment: Any = None,
                           output_dir: Path = None) -> Dict[str, Any]:
        """
        Complete execution and generate Harmony reasoning
        
        Returns:
            Dictionary containing both structured data and conversation string
        """
        capture = get_capture()
        capture.complete_execution(final_poem, quality_assessment)
        
        # Get the execution data
        execution = capture.get_execution()
        if not execution:
            return {}
        
        # Save raw execution data
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            execution_file = output_dir / f"{execution.execution_id}_raw.json"
            capture.export_execution(execution_file)
        
        # Generate structured Harmony data
        reasoner = HarmonyCompiler(llm)
        try:
            # Serialize the execution object to ensure all Constraints objects are converted to dictionaries
            execution_dict = execution.to_dict()
            structured_data = reasoner.generate_structured_harmony(execution_dict)
            
            # Save structured data
            if output_dir:
                structured_file = output_dir / f"{execution.execution_id}_structured.json"
                import json
                with open(structured_file, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, ensure_ascii=False, indent=2)
                print(f"Structured harmony data saved to: {structured_file}")
            
            # Convert to conversation format and save as text
            conversation = reasoner.create_harmony_conversation(structured_data)
            if output_dir:
                harmony_file = output_dir / f"{execution.execution_id}_harmony.txt"
                reasoner.save_harmony_reasoning(str(conversation), harmony_file)
            
            # Return both structured data and conversation string
            return {
                'structured_data': structured_data,
                'conversation_string': str(conversation),
                'execution_id': execution.execution_id
            }
            
        except Exception as e:
            print(f"Warning: Failed to generate harmony data: {e}")
            return {}

