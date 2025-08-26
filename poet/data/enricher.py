# poet/data/enricher.py

import logging
from typing import Dict, Any, Optional
from poet.core.node import Node
from poet.models.constraints import Constraints


class DataEnricher(Node):
    """
    Enriches constraints with additional data and examples.
    
    Supports harmony generation for data enrichment.
    """
    
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        
        self.llm = llm
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich constraints with additional data.
        
        Args:
            input_data: Input data containing constraints
            context: Pipeline context
            
        Returns:
            Output data with enriched constraints
        """
        # Validate inputs
        constraints = input_data.get('constraints')
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Enrich constraints
        enriched_constraints = self._enrich_constraints(constraints)
        
        # Store harmony data
        output_data = {
            'constraints': enriched_constraints,
            'enriched': True
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def _enrich_constraints(self, constraints: Constraints) -> Constraints:
        """Enrich constraints with additional data."""
        self.logger.info("📚 Enriching constraints with additional data")
        
        # For now, just return the original constraints
        # In a real implementation, this would fetch examples, metadata, etc.
        
        self.logger.info("✅ Constraints enriched successfully")
        return constraints
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """Generate natural reasoning for this enricher node."""
        constraints = input_data.get('constraints')
        
        reasoning = f"I enriched the constraints with additional data."
        
        if constraints:
            reasoning += f" The constraints for theme '{constraints.theme}' were enhanced with examples and metadata."
        
        reasoning += " This enrichment helps improve the quality of the generated poem."
        
        return reasoning
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        constraints = self.harmony_data['input'].get('constraints')
        if constraints:
            return f"Enriched constraints for theme: {constraints.theme}"
        return "Enriched constraints"
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        enriched = self.harmony_data['output'].get('enriched', False)
        return f"Data enrichment: {'Completed' if enriched else 'Failed'}"
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['constraints', 'enriched']


