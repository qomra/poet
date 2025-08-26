# poet/search/best_of_n_node.py

import json
import logging
from typing import Dict, Any, List, Optional
from poet.core.node import Node
from poet.prompts import get_global_prompt_manager
from poet.llm.base_llm import BaseLLM


class BestOfNNode(Node):
    """
    Wrapper node that applies Best-Of-N search strategy to any underlying node.
    
    This node takes an existing node (generator, evaluator, refiner) and runs it
    multiple times with different parameters, then selects the best result using
    LLM-based selection.
    """
    
    def __init__(self, underlying_node: Node, n_candidates: int = 5, 
                 selection_prompt: str = "generation_selection",
                 selection_metric: str = "overall_score",
                 temperature_range: Optional[List[float]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.underlying_node = underlying_node
        self.n_candidates = n_candidates
        self.selection_prompt = selection_prompt
        self.selection_metric = selection_metric
        self.temperature_range = temperature_range or [0.5, 0.7, 0.9, 1.1, 1.3]
        self.prompt_manager = get_global_prompt_manager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Best-Of-N search by running the underlying node multiple times.
        
        Args:
            input_data: Input data for the underlying node
            context: Pipeline context including LLM
            
        Returns:
            Best result from all candidates
        """
        try:
            self.logger.info(f"🎯 Starting Best-Of-N search with {self.n_candidates} candidates")
            
            # Get LLM from context
            llm = context.get('llm')
            if not llm:
                raise ValueError("LLM not provided in context")
            
            # Generate candidates
            candidates = self._generate_candidates(input_data, context)
            
            if not candidates:
                self.logger.warning("🚫 No candidates generated, returning original input")
                return input_data
            
            # Select best candidate
            best_result = self._select_best_candidate(candidates, input_data, context, llm)
            
            self.logger.info(f"🏆 Best-Of-N search completed, selected candidate {best_result.get('selected_index', 0)}")
            return best_result
            
        except Exception as e:
            self.logger.error(f"💥 Best-Of-N search failed: {e}")
            # Return original input if search fails
            return input_data
    
    def _generate_candidates(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple candidates by running the underlying node with different parameters."""
        candidates = []
        
        for i in range(self.n_candidates):
            try:
                # Create modified context for diversity
                search_context = self._modify_context(context, i)
                
                # Run the underlying node
                result = self.underlying_node.run(input_data, search_context)
                
                # Add metadata to track which candidate this is
                result['_candidate_index'] = i
                result['_candidate_temperature'] = search_context.get('temperature', 1.0)
                
                candidates.append(result)
                self.logger.debug(f"✨ Generated candidate {i+1}/{self.n_candidates}")
                
            except Exception as e:
                self.logger.warning(f"💔 Failed to generate candidate {i}: {e}")
                continue
        
        return candidates
    
    def _modify_context(self, context: Dict[str, Any], candidate_index: int) -> Dict[str, Any]:
        """Modify context to create diversity between candidates."""
        modified_context = context.copy()
        
        # Modify temperature for diversity
        if candidate_index < len(self.temperature_range):
            modified_context['temperature'] = self.temperature_range[candidate_index]
        else:
            # Cycle through temperature range if more candidates than temperatures
            modified_context['temperature'] = self.temperature_range[candidate_index % len(self.temperature_range)]
        
        return modified_context
    
    def _select_best_candidate(self, candidates: List[Dict[str, Any]], 
                             input_data: Dict[str, Any], 
                             context: Dict[str, Any],
                             llm: BaseLLM) -> Dict[str, Any]:
        """Select the best candidate using LLM-based selection."""
        try:
            # Prepare candidates data for selection prompt
            candidates_data = self._format_candidates_for_selection(candidates)
            
            # Get constraints for selection prompt
            constraints = input_data.get('constraints')
            
            # Handle constraints object vs dictionary
            if hasattr(constraints, 'meter'):
                # Constraints is a dataclass object
                meter = constraints.meter or 'غير محدد'
                qafiya = constraints.qafiya or 'غير محدد'
                line_count = constraints.line_count or 4
                theme = constraints.theme or 'غير محدد'
                tone = constraints.tone or 'غير محدد'
            else:
                # Constraints is a dictionary or None
                constraints = constraints or {}
                meter = constraints.get('meter', 'غير محدد')
                qafiya = constraints.get('qafiya', 'غير محدد')
                line_count = constraints.get('line_count', 4)
                theme = constraints.get('theme', 'غير محدد')
                tone = constraints.get('tone', 'غير محدد')
            
            # Format selection prompt based on prompt type
            if self.selection_prompt in ['prosody_refiner_selection', 'qafiya_refiner_selection']:
                # Refiner-specific prompts need different parameters
                if self.selection_prompt == 'qafiya_refiner_selection':
                    # Get qafiya-specific parameters from constraints
                    qafiya_type = getattr(constraints, 'qafiya_type', 'غير محدد') if hasattr(constraints, 'qafiya_type') else 'غير محدد'
                    qafiya_harakah = getattr(constraints, 'qafiya_harakah', 'غير محدد') if hasattr(constraints, 'qafiya_harakah') else 'غير محدد'
                    
                    # Get original poem text
                    original_poem = ""
                    if 'poem' in input_data and hasattr(input_data['poem'], 'verses'):
                        original_poem = "\n".join(input_data['poem'].verses)
                    
                    formatted_prompt = self.prompt_manager.format_prompt(
                        self.selection_prompt,
                        original_poem=original_poem,
                        qafiya=qafiya,
                        qafiya_type=qafiya_type,
                        qafiya_harakah=qafiya_harakah,
                        n_candidates=len(candidates),
                        candidates=candidates_data
                    )
                else:  # prosody_refiner_selection
                    # Get original poem text
                    original_poem = ""
                    if 'poem' in input_data and hasattr(input_data['poem'], 'verses'):
                        original_poem = "\n".join(input_data['poem'].verses)
                    
                    formatted_prompt = self.prompt_manager.format_prompt(
                        self.selection_prompt,
                        original_poem=original_poem,
                        meter=meter,
                        n_candidates=len(candidates),
                        candidates=candidates_data
                    )
            else:
                # Generic selection prompt
                formatted_prompt = self.prompt_manager.format_prompt(
                    self.selection_prompt,
                    meter=meter,
                    qafiya=qafiya,
                    line_count=line_count,
                    theme=theme,
                    tone=tone,
                    candidates_data=candidates_data,
                    selection_metric=self.selection_metric
                )
            
            # Get LLM selection
            response = llm.generate(formatted_prompt)
            
            # Parse selection result
            selection_result = self._parse_selection_response(response)
            
            # Get the selected candidate
            selected_index = selection_result.get('selected_candidate', 0)
            if 0 <= selected_index < len(candidates):
                best_candidate = candidates[selected_index].copy()
                best_candidate['selection_metadata'] = selection_result
                best_candidate['selected_index'] = selected_index
                return best_candidate
            else:
                self.logger.warning(f"🔢 Invalid selected index {selected_index}, using first candidate")
                return candidates[0]
                
        except Exception as e:
            self.logger.error(f"🎲 Failed to select best candidate: {e}")
            # Return first candidate as fallback
            return candidates[0] if candidates else input_data
    
    def _format_candidates_for_selection(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidates data for the selection prompt."""
        formatted_candidates = []
        
        for i, candidate in enumerate(candidates):
            candidate_text = f"**المرشح {i}:**\n"
            
            # Extract poem verses if available
            if 'poem' in candidate and hasattr(candidate['poem'], 'verses'):
                verses = candidate['poem'].verses
                candidate_text += "\n".join(verses)
            elif 'verses' in candidate:
                candidate_text += "\n".join(candidate['verses'])
            
            # Add quality information if available
            if 'poem' in candidate and hasattr(candidate['poem'], 'quality') and candidate['poem'].quality:
                quality = candidate['poem'].quality
                candidate_text += f"\n\n**جودة التقييم:**\n"
                candidate_text += f"- النتيجة الإجمالية: {quality.overall_score:.3f}\n"
                if quality.prosody_issues:
                    candidate_text += f"- مشاكل العروض: {', '.join(quality.prosody_issues)}\n"
                if quality.qafiya_issues:
                    candidate_text += f"- مشاكل القافية: {', '.join(quality.qafiya_issues)}\n"
            
            formatted_candidates.append(candidate_text)
        
        return "\n\n---\n\n".join(formatted_candidates)
    
    def _parse_selection_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM selection response."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"🔧 Failed to parse selection response: {e}")
            return {"selected_candidate": 0}
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return self.underlying_node.validate_input(input_data)
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output data."""
        return self.underlying_node.validate_output(output_data)
    
    def get_required_inputs(self) -> list:
        """Get required input keys."""
        return self.underlying_node.get_required_inputs()
    
    def get_output_keys(self) -> list:
        """Get output keys."""
        return self.underlying_node.get_output_keys()
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """Generate natural reasoning for this Best-Of-N node."""
        node_type = getattr(self, 'node_type', 'unknown')
        n_candidates = getattr(self, 'n_candidates', 'unknown')
        selected_index = output_data.get('selected_index', 'unknown')
        
        reasoning = f"I performed Best-Of-N search for {node_type} with {n_candidates} candidates."
        reasoning += f" After evaluating all candidates, I selected candidate {selected_index} as the best option."
        
        if node_type == 'generation':
            reasoning += " This candidate showed the best adherence to the poetic constraints and theme."
        elif node_type == 'evaluation':
            reasoning += " This candidate received the highest quality score across all evaluation metrics."
        elif node_type == 'refiner_chain':
            reasoning += " This candidate showed the most improvement after refinement iterations."
        
        return reasoning
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        # Show what type of operation this Best-Of-N node is performing
        node_type = getattr(self, 'node_type', 'unknown')
        n_candidates = getattr(self, 'n_candidates', 'unknown')
        
        if node_type == 'generation':
            return f"Best-Of-N generation with {n_candidates} candidates"
        elif node_type == 'evaluation':
            return f"Best-Of-N evaluation with {n_candidates} candidates"
        elif node_type == 'refiner_chain':
            return f"Best-Of-N refinement with {n_candidates} candidates"
        else:
            return f"Best-Of-N search with {n_candidates} candidates"
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        # Get the selected candidate information
        selected_index = self.harmony_data['output'].get('selected_index', 'unknown')
        selection_metadata = self.harmony_data['output'].get('selection_metadata', {})
        
        # Get candidate details
        if 'poem' in self.harmony_data['output']:
            poem = self.harmony_data['output']['poem']
            if hasattr(poem, 'verses'):
                verses = poem.verses
                if isinstance(verses, list) and len(verses) > 0:
                    verses_text = "\n".join([f"Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                    return f"Best-Of-N selected candidate {selected_index}:\n{verses_text}"
        
        return f"Best-Of-N selected candidate {selected_index}"
