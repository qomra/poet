# poet/interface/dataset_interface.py

import json
import time
import signal
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base_interface import BaseInterface
from poet.core.agent import DynamicAgent as Agent
from poet.models.poem import LLMPoem


class DatasetInterface(BaseInterface):
    """Dataset processing interface for poetry generation experiments."""
    
    def __init__(self, agent: Agent, dataset_path: str, output_path: str, n_per_instance: Optional[int] = None):
        # Store the agent and paths
        self.agent = agent
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.n_per_instance = n_per_instance or 1  # Default to 1 generation per instance
        self.running = False
        self._setup_signal_handlers()
        
        # Setup logging using centralized configuration
        import logging
        from poet.utils.logging_config import configure_logging
        configure_logging(level=logging.INFO, suppress_http=True)
        self.logger = logging.getLogger(__name__)
        
        # Load dataset
        self.dataset = self._load_dataset()
        self.total_items = len(self.dataset)
        self.processed_count = 0
        
        # Initialize output structure
        self.output_data = []
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the dataset from JSON file."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded dataset with {len(data)} items from {self.dataset_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _save_output(self):
        """Save the current output to file."""
        try:
            # Create output directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.output_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved output to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")
    
    def _process_dataset_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single dataset item and return multiple results with flattened harmony."""
        results = []
        poem_id = item['poem_id']
        prompt_text = item['prompt']['text']
        
        self.logger.info(f"Processing item {poem_id}: {prompt_text[:100]}... (generating {self.n_per_instance} poems)")
        
        for generation_idx in range(self.n_per_instance):
            try:
                self.logger.info(f"  Generating poem {generation_idx + 1}/{self.n_per_instance} for item {poem_id}")
                
                # Run the pipeline
                result = self.agent.run_pipeline(prompt_text)
                
                # Debug: log the result structure
                self.logger.info(f"Pipeline result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
                # Extract harmony results if available
                harmony_results = ""
                if result.get('success'):
                    # Use the structured harmony data if available
                    if result.get('harmony_structured_data'):
                        try:
                            structured_data = result['harmony_structured_data']
                            
                            # Extract analysis channel messages from the structured data
                            messages = structured_data.get('messages', [])
                            analysis_messages = []
                            
                            for msg in messages:
                                if msg.get('channel') == 'analysis' and msg.get('role') == 'assistant':
                                    content = msg.get('content', '')
                                    if content:
                                        analysis_messages.append(content)
                            
                            if analysis_messages:
                                harmony_results = "\n\n".join(analysis_messages)
                            else:
                                # Fallback to conversation string if no analysis messages found
                                harmony_results = result.get('harmony_reasoning', "")
                                
                        except Exception as e:
                            # Fallback to conversation string if structured data parsing fails
                            harmony_results = result.get('harmony_reasoning', "")
                    else:
                        # Fallback to conversation string if no structured data
                        harmony_results = result.get('harmony_reasoning', "")
                
                # Create output item
                output_item = {
                    'poem_id': f"{poem_id}_gen_{generation_idx + 1}",
                    'original_poem_id': poem_id,
                    'generation_index': generation_idx + 1,
                    'reference': item['reference'],
                    'prompt': item['prompt'],
                    'ai': {
                        'text': "\n".join(result.get('poem').verses) if result.get('poem') else "",
                        'provider': 'poet_system',
                        'model': 'pipeline_generated',
                        'thoughtMap': harmony_results
                    }
                }
                
                results.append(output_item)
                self.logger.info(f"  Successfully generated poem {generation_idx + 1}/{self.n_per_instance} for item {poem_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate poem {generation_idx + 1} for item {poem_id}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Add error result
                error_item = {
                    'poem_id': f"{poem_id}_gen_{generation_idx + 1}_error",
                    'original_poem_id': poem_id,
                    'generation_index': generation_idx + 1,
                    'reference': item.get('reference', {}),
                    'prompt': item.get('prompt', {}),
                    'ai': item.get('ai', {}),
                    'error': str(e)
                }
                results.append(error_item)
        
        return results
    
    def run(self):
        """Run the dataset processing."""
        self.running = True
        total_generations = self.total_items * self.n_per_instance
        self.logger.info(f"Starting dataset processing for {self.total_items} items, generating {self.n_per_instance} poems per item (total: {total_generations} generations)")
        
        try:
            for i, item in enumerate(self.dataset):
                if not self.running:
                    self.logger.info("Processing interrupted by user")
                    break
                
                # Process the item (generates multiple results)
                results = self._process_dataset_item(item)
                self.output_data.extend(results)
                self.processed_count += 1
                
                # Save progress every 5 items
                if self.processed_count % 5 == 0:
                    self._save_output()
                    self.logger.info(f"Progress: {self.processed_count}/{self.total_items} items processed ({len(self.output_data)} total generations)")
                
                # Small delay to avoid overwhelming the API
                time.sleep(1)
            
            # Final save
            self._save_output()
            self.logger.info(f"Dataset processing completed. Processed {self.processed_count} items, generated {len(self.output_data)} total poems.")
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            self._save_output()
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            self._save_output()
            raise
    
    def stop(self):
        """Stop the processing."""
        self.running = False
        self.logger.info("Stopping dataset processing...")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, stopping...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        total_generations = self.total_items * self.n_per_instance
        return {
            'total_items': self.total_items,
            'processed_count': self.processed_count,
            'total_generations': total_generations,
            'generated_count': len(self.output_data),
            'progress_percentage': (self.processed_count / self.total_items * 100) if self.total_items > 0 else 0,
            'running': self.running
        }
    
    def is_completed(self) -> bool:
        """Check if the dataset processing has completed."""
        return self.processed_count >= self.total_items and not self.running 