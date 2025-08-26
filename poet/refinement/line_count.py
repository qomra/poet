# poet/refinement/line_count.py

import logging
from typing import Dict, Any, Optional
from poet.core.node import Node
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment


class LineCountRefiner(Node):
    """
    Refines poem line count to match constraints.
    
    Supports iteration context for refinement pipelines.
    """
    
    def __init__(self, llm, prompt_manager=None, iteration: int = None, **kwargs):
        super().__init__(**kwargs)
        
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.iteration = iteration
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine poem line count.
        
        Args:
            input_data: Input data containing poem, constraints, and evaluation
            context: Pipeline context
            
        Returns:
            Output data with refined poem
        """
        # Validate inputs
        poem = input_data.get('poem')
        constraints = input_data.get('constraints')
        evaluation = input_data.get('evaluation')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Check if line count refinement is needed
        if not self.should_refine(poem, constraints):
            self.logger.info("ℹ️ Line count refinement not needed")
            return {
                'poem': poem,
                'refined': False,
                'refiner_used': 'line_count_refiner'
            }
        
        # Perform line count refinement
        refined_poem = self._refine_line_count(poem, constraints)
        
        # Store harmony data
        output_data = {
            'poem': refined_poem,
            'refined': True,
            'refiner_used': 'line_count_refiner',
            'iteration': self.iteration
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def should_refine(self, poem: LLMPoem, constraints: Constraints) -> bool:
        """Check if line count refinement is needed."""
        current_lines = len(poem.verses)
        target_lines = constraints.line_count or 4  # Default to 4 lines
        
        return current_lines != target_lines
    
    def _refine_line_count(self, poem: LLMPoem, constraints: Constraints) -> LLMPoem:
        """Refine the poem's line count."""
        self.logger.info("📏 Refining poem line count")
        
        target_lines = constraints.line_count or 4
        current_lines = len(poem.verses)
        
        # Create refinement prompt
        refinement_prompt = self.prompt_manager.format_prompt(
            'line_count_refinement',
            poem_text=poem.get_text(),
            current_lines=current_lines,
            target_lines=target_lines,
            theme=constraints.theme,
            meter=constraints.meter,
            qafiya=constraints.qafiya,
            iteration=self.iteration or 1
        )
        
        # Generate refined poem
        response = self.llm.generate(refinement_prompt)
        
        # Parse response and create new poem
        refined_verses = self._parse_refinement_response(response)
        
        # Create refined poem
        refined_poem = LLMPoem(
            verses=refined_verses,
            llm_provider=self.llm.__class__.__name__,
            model_name=getattr(self.llm, 'model_name', 'unknown'),
            constraints={
                'theme': poem.constraints.get('theme') if poem.constraints else None,
                'meter': poem.constraints.get('meter') if poem.constraints else None,
                'qafiya': poem.constraints.get('qafiya') if poem.constraints else None
            }
        )
        
        self.logger.info("✅ Line count refinement completed")
        return refined_poem
    
    def _parse_refinement_response(self, response: str) -> list:
        """Parse the refinement response into verses."""
        # Simple parsing - split by lines and group into verses
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Group lines into verses (assuming 2 lines per verse)
        verses = []
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                verses.append([lines[i], lines[i + 1]])
            else:
                verses.append([lines[i]])
        
        return verses
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """Generate natural reasoning for this refiner node."""
        iteration_text = f" (Iteration {self.iteration})" if self.iteration else ""
        refined = output_data.get('refined', False)
        
        if refined:
            reasoning = f"I refined the poem's line count{iteration_text}."
            reasoning += " I adjusted the number of lines to match the specified constraints."
        else:
            reasoning = f"I checked the poem's line count{iteration_text}."
            reasoning += " The line count was already correct, so no refinement was needed."
        
        return reasoning
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        poem = self.harmony_data['input'].get('poem')
        if poem:
            return f"Refined line count for poem with {len(poem.verses)} verses"
        return "Refined line count"
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        refined = self.harmony_data['output'].get('refined', False)
        return f"Line count refinement: {'Applied' if refined else 'Not needed'}"
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refiner_used', 'iteration']