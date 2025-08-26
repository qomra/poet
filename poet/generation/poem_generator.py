# poet/generation/poem_generator.py

import logging
from typing import Dict, Any, Optional
from poet.core.node import Node
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints


class SimplePoemGenerator(Node):
    """
    Generates Arabic poems based on constraints.
    
    Supports harmony generation for poem creation.
    """
    
    def __init__(self, llm, prompt_manager=None, **kwargs):
        super().__init__(**kwargs)
        
        self.llm = llm
        self.prompt_manager = prompt_manager
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a poem based on constraints.
        
        Args:
            input_data: Input data containing constraints
            context: Pipeline context
            
        Returns:
            Output data with generated poem
        """
        # Validate inputs
        constraints = input_data.get('constraints')
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Generate poem
        poem = self._generate_poem(constraints)
        
        # Store harmony data
        output_data = {
            'poem': poem,
            'generated': True
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def _generate_poem(self, constraints: Constraints) -> LLMPoem:
        """Generate a poem based on constraints."""
        self.logger.info("🎭 Generating poem based on constraints")
        
        # Create generation prompt
        # In Arabic poetry: 1 bait = 2 lines (صدر + عجز)
        # So line_count represents number of baits, not total lines
        bait_count = constraints.line_count or 2
        total_lines = bait_count * 2  # Each bait = 2 lines
        
        self.logger.info(f"🎭 Generating {bait_count} baits ({total_lines} total lines)")
        
        generation_prompt = self.prompt_manager.format_prompt(
            'poem_generation',
            theme=constraints.theme,
            meter=constraints.meter,
            qafiya=constraints.qafiya,
            line_count=bait_count,  # This represents number of baits
            tone=constraints.tone,
            imagery=constraints.imagery,
            keywords=constraints.keywords
        )
        # Generate poem
        response = self.llm.generate(generation_prompt)
        
        # Parse response and create poem
        verses = self._parse_generation_response(response)
        
        # Create poem object
        poem = LLMPoem(
            verses=verses,
            llm_provider=self.llm.__class__.__name__,
            model_name=getattr(self.llm, 'model_name', 'unknown'),
            constraints={
                'theme': constraints.theme,
                'meter': constraints.meter,
                'qafiya': constraints.qafiya
            }
        )
        
        self.logger.info("✅ Poem generated successfully")
        return poem
    
    def _parse_generation_response(self, response: str) -> list:
        """Parse the generation response into verses."""
        # Simple parsing - split by lines and filter empty lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Return lines as individual verses
        return lines
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """Generate natural reasoning for this generator node."""
        constraints = input_data.get('constraints')
        poem = output_data.get('poem')
        
        reasoning = f"I generated a poem based on the constraints."
        
        if constraints:
            reasoning += f" The poem follows the theme '{constraints.theme}' with meter '{constraints.meter}' and qafiya '{constraints.qafiya}'."
        
        if poem:
            reasoning += f" The generated poem has {len(poem.verses)} verses."
        
        reasoning += " I focused on creating verses that match the specified constraints while maintaining poetic quality."
        
        return reasoning
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        constraints = self.harmony_data['input'].get('constraints')
        if constraints:
            return f"Generated poem for theme: {constraints.theme}, meter: {constraints.meter}, qafiya: {constraints.qafiya}"
        return "Generated poem"
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        poem = self.harmony_data['output'].get('poem')
        if poem:
            if hasattr(poem, 'verses'):
                verses = poem.verses
                if isinstance(verses, list) and len(verses) > 0:
                    # Show all verses, not just the first one
                    verses_text = "\n".join([f"Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                    return f"Generated: {len(verses)} verses\n{verses_text}"
            elif isinstance(poem, dict) and 'verses' in poem:
                verses = poem['verses']
                if isinstance(verses, list) and len(verses) > 0:
                    # Show all verses, not just the first one
                    verses_text = "\n".join([f"Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                    return f"Generated: {len(verses)} verses\n{verses_text}"
        return "Poem verses created"
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'generated']
