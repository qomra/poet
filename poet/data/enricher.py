# poet/data/enricher.py

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from poet.core.node import Node
from poet.models.constraints import Constraints, ExampleData
from poet.models.search import CorpusExample, WebExample, DataExample
from poet.data.corpus_manager import CorpusManager
from poet.analysis.knowledge_retriever import CorpusKnowledgeRetriever, WebKnowledgeRetriever


class EnricherError(Exception):
    """Raised when data enrichment fails"""
    pass


class DataEnricher(Node):
    """
    Enriches user constraints with relevant examples from corpus and web search.
    
    This node takes user constraints and searches configured data sources to
    retrieve relevant poems, examples, and contextual information that can
    inform the poem generation process.
    """
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Will be initialized when sources are provided
        self._corpus_retriever = None
        self._web_retriever = None
        
    def _initialize_retrievers(self, sources: List[Dict[str, Any]], context: Dict[str, Any]):
        """Initialize retrieval components based on source configuration"""
        
        for source in sources:
            source_type = source.get('type')
            
            if source_type == 'corpus':
                # Initialize corpus retriever
                local_knowledge_path = source.get('local_knowledge_path')
                if local_knowledge_path:
                    try:
                        corpus_manager = CorpusManager(local_knowledge_path)
                        self._corpus_retriever = CorpusKnowledgeRetriever(corpus_manager)
                        self.logger.info(f"Initialized corpus retriever with path: {local_knowledge_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize corpus retriever: {e}")
                        
            elif source_type == 'web_search':
                # Initialize web search retriever
                if self.llm:
                    try:
                        search_provider_type = source.get('provider', 'duckduckgo')
                        search_provider_config = source.get('config', {})
                        self._web_retriever = WebKnowledgeRetriever(
                            self.llm, 
                            search_provider_type, 
                            search_provider_config
                        )
                        self.logger.info(f"Initialized web retriever with provider: {search_provider_type}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize web retriever: {e}")
                else:
                    self.logger.warning("Cannot initialize web retriever: LLM not available")
                    
    def enrich_constraints(self, constraints: Constraints, sources: List[Dict[str, Any]], 
                          context: Dict[str, Any]) -> Constraints:
        """
        Enrich constraints with relevant examples from configured sources.
        
        Args:
            constraints: User constraints to enrich
            sources: List of data source configurations
            context: Pipeline context
            
        Returns:
            Enhanced constraints with example_data populated
        """
        try:
            # Initialize retrievers based on source configuration
            self._initialize_retrievers(sources, context)
            
            # Create ExampleData structure
            example_data = ExampleData(
                corpus_examples=[],
                web_examples=[],
                retrieval_metadata={}
            )
            
            # Process each source
            for source in sources:
                source_type = source.get('type')
                search_criteria = source.get('search_criteria', [])
                top_k = source.get('top_k', 5)
                # create a constraints object from the search criteria
                search_constraints = Constraints()
                for criterion in search_criteria:
                    if criterion == 'meter':
                        search_constraints.meter = constraints.meter
                    elif criterion == 'qafiya':
                        search_constraints.qafiya = constraints.qafiya
                    elif criterion == 'theme':
                        search_constraints.theme = constraints.theme
                self.logger.info(f"🔍 Search constraints: {search_constraints}")
                if source_type == 'corpus' and self._corpus_retriever:
                    try:
                        # Extract relevant search criteria for corpus
                        if self._should_search_corpus(constraints, search_criteria):
                            corpus_result = self._corpus_retriever.search(
                                search_constraints, 
                                max_results=top_k,
                                strategy="exact_match"
                            )
                            
                            # Create CorpusExample objects
                            for record in corpus_result.corpus_results:
                                # Determine which criteria matched
                                matched_criteria = self._get_matched_criteria(constraints, record, search_criteria)
                                
                                corpus_example = CorpusExample(
                                    search_criteria=matched_criteria,
                                    title=record.title or "",
                                    verses=record.verses if isinstance(record.verses, str) else '\n'.join(record.verses) if record.verses else "",
                                    meter=record.meter or "",
                                    qafiya=record.qafiya or "",
                                    theme=record.theme or "",
                                    poet_name=record.poet_name or "",
                                    poet_era=record.poet_era or "",
                                    metadata={
                                        'source_type': 'corpus',
                                        'retrieval_strategy': corpus_result.retrieval_strategy
                                    }
                                )
                                example_data.corpus_examples.append(corpus_example)
                            
                            example_data.retrieval_metadata['corpus'] = {
                                'total_found': corpus_result.total_found,
                                'strategy': corpus_result.retrieval_strategy,
                                'search_criteria_used': search_criteria,
                                'examples_count': len(example_data.corpus_examples)
                            }
                            
                            self.logger.info(f"Retrieved {len(example_data.corpus_examples)} corpus examples")
                            
                    except Exception as e:
                        self.logger.error(f"Corpus enrichment failed: {e}")
                        
                elif source_type == 'web_search' and self._web_retriever:
                    try:
                        # Perform web search enrichment
                        web_result = self._web_retriever.search(
                            constraints,
                            max_queries_per_round=3,
                            max_rounds=1
                        )
                        
                        # Create WebExample objects
                        for result in web_result.search_results:
                            # Determine which criteria were used for web search
                            matched_criteria = self._get_web_search_criteria(constraints, search_criteria)
                            
                            web_example = WebExample(
                                search_criteria=matched_criteria,
                                title=result.title,
                                content=result.content,
                                url=result.url,
                                relevance_score=getattr(result, 'relevance_score', None),
                                metadata={
                                    'source_type': 'web_search',
                                    'search_metadata': web_result.metadata
                                }
                            )
                            example_data.web_examples.append(web_example)
                        
                        example_data.retrieval_metadata['web'] = {
                            'total_queries': web_result.metadata.get('total_queries', 0),
                            'total_results': web_result.metadata.get('total_results', 0),
                            'rounds_executed': web_result.metadata.get('rounds_executed', 0),
                            'search_criteria_used': search_criteria,
                            'examples_count': len(example_data.web_examples)
                        }
                        
                        self.logger.info(f"Retrieved {len(example_data.web_examples)} web examples")
                        
                    except Exception as e:
                        self.logger.error(f"Web search enrichment failed: {e}")
                        
            # Update constraints with enrichment data
            constraints.example_data = example_data
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"Failed to enrich constraints: {e}")
            raise EnricherError(f"Data enrichment failed: {e}")
            
    def _should_search_corpus(self, constraints: Constraints, search_criteria: List[str]) -> bool:
        """Determine if corpus search should be performed based on criteria"""
        if not search_criteria:
            return True  # Search if no specific criteria
            
        for criterion in search_criteria:
            if criterion == 'meter' and constraints.meter:
                return True
            elif criterion == 'qafiya' and constraints.qafiya:
                return True
            elif criterion == 'theme' and constraints.theme:
                return True
                
        return False
    
    def _get_matched_criteria(self, constraints: Constraints, record, search_criteria: List[str]) -> List[str]:
        """Determine which search criteria were matched for this record"""
        matched = []
        
        for criterion in search_criteria:
            if criterion == 'meter' and constraints.meter and record.meter:
                if constraints.meter.lower() in record.meter.lower():
                    matched.append('meter')
            elif criterion == 'qafiya' and constraints.qafiya and record.qafiya:
                if constraints.qafiya.lower() in record.qafiya.lower():
                    matched.append('qafiya')
            elif criterion == 'theme' and constraints.theme and record.theme:
                if constraints.theme.lower() in record.theme.lower():
                    matched.append('theme')
        
        return matched or search_criteria  # Return original criteria if no specific matches
    
    def _get_web_search_criteria(self, constraints: Constraints, search_criteria: List[str]) -> List[str]:
        """Determine search criteria used for web search"""
        active_criteria = []
        
        for criterion in search_criteria:
            if criterion == 'meter' and constraints.meter:
                active_criteria.append('meter')
            elif criterion == 'qafiya' and constraints.qafiya:
                active_criteria.append('qafiya')
            elif criterion == 'theme' and constraints.theme:
                active_criteria.append('theme')
        
        return active_criteria or search_criteria
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """Generate natural reasoning for this enricher node."""
        constraints = input_data.get('constraints')
        enrichment_performed = output_data.get('enrichment_performed', False)
        
        reasoning = f"I enriched the constraints with additional data."
        
        if constraints:
            reasoning += f" The constraints for theme '{constraints.theme}' were enhanced with examples and metadata."
        
        if enrichment_performed:
            reasoning += " I successfully retrieved relevant examples from corpus and web sources to improve the poem generation process."
        else:
            reasoning += " No enrichment was performed due to missing source configuration."
        
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
        
        enrichment_performed = self.harmony_data['output'].get('enrichment_performed', False)
        enriched_data_available = self.harmony_data['output'].get('enriched_data_available', False)
        
        if enrichment_performed and enriched_data_available:
            return "Data enrichment: Completed with examples retrieved"
        elif enrichment_performed:
            return "Data enrichment: Completed but no examples found"
        else:
            return "Data enrichment: Not performed"
        
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data enrichment node.
        
        Args:
            input_data: Input data containing constraints and user_prompt
            context: Pipeline context with LLM and other components
            
        Returns:
            Output data with enriched constraints
        """
        # Set up context
        self.llm = context.get('llm')
        
        # Extract required data
        constraints = input_data.get('constraints')
        if not constraints:
            raise ValueError("constraints not found in input_data")
            
        # Get sources configuration from node config or input data
        self.logger.info(f"🔍 Config keys: {list(self.config.keys())}")
        self.logger.info(f"🔍 Config content: {self.config}")
        
        # Check for sources in different possible locations
        sources = []
        if 'sources' in self.config:
            sources = self.config['sources']
        elif 'data_enrichment' in self.config and 'sources' in self.config['data_enrichment']:
            sources = self.config['data_enrichment']['sources']
        elif 'sources' in input_data:
            sources = input_data['sources']
            
        if not sources:
            self.logger.warning("No sources configured for data enrichment")
            self.logger.warning(f"🔍 Available config keys: {list(self.config.keys())}")
            self.logger.warning(f"🔍 Available input keys: {list(input_data.keys())}")
            output_data = {
                'constraints': constraints,
                'enrichment_performed': False
            }
            self._store_harmony_data(input_data, output_data)
            return output_data
            
        # Enrich constraints
        enriched_constraints = self.enrich_constraints(constraints, sources, context)
        
        output_data = {
            'constraints': enriched_constraints,
            'enrichment_performed': True,
            'enriched_data_available': enriched_constraints.example_data is not None
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['constraints', 'enrichment_performed', 'enriched_data_available']


