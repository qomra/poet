# poet/analysis/knowledge_retriever.py

import logging
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from poet.llm.base_llm import BaseLLM
from poet.models.constraints import UserConstraints
from poet.models.search import SearchQuery, QueryGenerationResult, EvaluatedResult, ResultEvaluationResult
from poet.data.search_provider import SearchProviderFactory, SearchResult
from poet.data.corpus_manager import CorpusManager, PoemRecord, SearchCriteria
from poet.prompts.prompt_manager import PromptManager


@dataclass
class RetrievalResult:
    """Results from knowledge retrieval"""
    total_found: int = 0
    search_criteria: SearchCriteria = None
    retrieval_strategy: str = None
    metadata: Dict[str, Any] = None

@dataclass
class WebRetrievalResult(RetrievalResult):
    """Results from web retrieval"""
    web_results: List[SearchResult] = field(default_factory=list)

@dataclass
class CorpusRetrievalResult(RetrievalResult):
    """Results from corpus retrieval"""
    corpus_results: List[PoemRecord] = field(default_factory=list)

class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from the corpus based on user constraints.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def search(self, constraints: UserConstraints, 
                         max_results: int = 5,
                         strategy: str = "best_match") -> RetrievalResult:
        """
        Retrieve relevant examples based on user constraints.
        """
        pass

class WebKnowledgeRetriever(KnowledgeRetriever):
    """
    Retrieves relevant knowledge from the web based on user constraints.
    """
    def __init__(self, llm_provider: BaseLLM, search_provider_type: str, search_provider_config: Dict[str, Any]):
        super().__init__()
        self.llm = llm_provider
        self.search_provider = SearchProviderFactory.create_provider(search_provider_type, search_provider_config)
        self.prompt_manager = PromptManager()
        self.logger = logging.getLogger(__name__)
    
    def search(self, constraints: UserConstraints, 
                         max_queries_per_round: int = 5,
                         max_rounds: int = 1) -> WebRetrievalResult:
        """
        Retrieve relevant knowledge from the web using multi-round search enrichment.
        
        Args:
            constraints: User requirements for the poem
            max_queries_per_round: Maximum number of queries to generate per round
            max_rounds: Maximum number of search rounds
            
        Returns:
            WebRetrievalResult with search results and metadata
        """
        self.logger.info(f"Starting web search enrichment with {max_rounds} rounds")
        
        all_results = []
        search_metadata = {
            "rounds_executed": 0,
            "total_queries": 0,
            "total_results": 0,
            "evaluation_summary": []
        }
        
        for round_num in range(max_rounds):
            self.logger.info(f"Executing search round {round_num + 1}")
            
            # Generate search queries
            queries = self._generate_search_queries(constraints, max_queries_per_round)
            if not queries:
                self.logger.warning(f"No queries generated for round {round_num + 1}")
                break
            
            # Execute searches
            round_results = []
            for query in queries:
                try:
                    search_response = self.search_provider.search(query.query, max_results=3)
                    round_results.extend(search_response.results)
                    search_metadata["total_queries"] += 1
                except Exception as e:
                    self.logger.error(f"Search failed for query '{query.query}': {e}")
            
            if not round_results:
                self.logger.warning(f"No results found in round {round_num + 1}")
                break
            
            # Evaluate results
            evaluation = self._evaluate_search_results(constraints, round_results)
            
            # Add high-quality results to final results
            for eval_result in evaluation.evaluated_results:
                if eval_result.is_worth_following and eval_result.relevance_score >= 7:
                    if eval_result.result_index < len(round_results):
                        all_results.append(round_results[eval_result.result_index])
            
            search_metadata["rounds_executed"] += 1
            search_metadata["total_results"] += len(round_results)
            search_metadata["evaluation_summary"].append({
                "round": round_num + 1,
                "queries_executed": len(queries),
                "results_found": len(round_results),
                "high_quality_results": len([r for r in evaluation.evaluated_results if r.is_worth_following]),
                "overall_assessment": evaluation.overall_assessment,
                "gaps_identified": evaluation.gaps_identified
            })
            
            # Check if we need follow-up
            if not evaluation.followup_needed:
                self.logger.info("No follow-up needed, ending search")
                break
        
        return WebRetrievalResult(
            web_results=all_results,
            total_found=len(all_results),
            search_criteria=None,  # Not applicable for web search
            retrieval_strategy=f"multi_round_web_search_{search_metadata['rounds_executed']}_rounds",
            metadata=search_metadata
        )
    
    def _generate_search_queries(self, constraints: UserConstraints, max_queries: int) -> List[SearchQuery]:
        """Generate search queries using LLM"""
        try:
            # Format the query generation prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'query_generator',
                original_prompt=constraints.original_prompt or "",
                meter=constraints.meter or "",
                qafiya=constraints.qafiya or "",
                line_count=constraints.line_count or "",
                theme=constraints.theme or "",
                tone=constraints.tone or "",
                imagery=", ".join(constraints.imagery) if constraints.imagery else "",
                keywords=", ".join(constraints.keywords) if constraints.keywords else "",
                sections=", ".join(constraints.sections) if constraints.sections else "",
                register=constraints.register or "",
                era=constraints.era or "",
                poet_style=constraints.poet_style or ""
            )
            
            # Get LLM response
            response = self.llm.generate(formatted_prompt)
            
            # Parse JSON response
            data = self._parse_llm_response(response)
            
            # Create SearchQuery objects
            queries = []
            for query_data in data.get("queries", [])[:max_queries]:
                queries.append(SearchQuery(
                    query=query_data["query"],
                    purpose=query_data["purpose"]
                ))
            
            self.logger.info(f"Generated {len(queries)} search queries")
            return queries
            
        except Exception as e:
            self.logger.error(f"Failed to generate search queries: {e}")
            return []
    
    def _evaluate_search_results(self, constraints: UserConstraints, search_results: List[SearchResult]) -> ResultEvaluationResult:
        """Evaluate search results using LLM"""
        try:
            # Format search results for evaluation
            results_text = self._format_search_results_for_evaluation(search_results)
            
            # Format the evaluation prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'result_evaluator',
                original_prompt=constraints.original_prompt or "",
                meter=constraints.meter or "",
                qafiya=constraints.qafiya or "",
                line_count=constraints.line_count or "",
                theme=constraints.theme or "",
                tone=constraints.tone or "",
                imagery=", ".join(constraints.imagery) if constraints.imagery else "",
                keywords=", ".join(constraints.keywords) if constraints.keywords else "",
                sections=", ".join(constraints.sections) if constraints.sections else "",
                register=constraints.register or "",
                era=constraints.era or "",
                poet_style=constraints.poet_style or "",
                search_results=results_text
            )
            
            # Get LLM response
            response = self.llm.generate(formatted_prompt)
            
            # Parse JSON response
            data = self._parse_llm_response(response)
            
            # Create EvaluatedResult objects
            evaluated_results = []
            for eval_data in data.get("evaluated_results", []):
                evaluated_results.append(EvaluatedResult(
                    result_index=eval_data["result_index"],
                    relevance_score=eval_data["relevance_score"],
                    quality_score=eval_data["quality_score"],
                    usefulness_score=eval_data["usefulness_score"],
                    is_worth_following=eval_data["is_worth_following"],
                    key_insights=eval_data["key_insights"],
                    recommendation=eval_data["recommendation"]
                ))
            
            return ResultEvaluationResult(
                evaluated_results=evaluated_results,
                overall_assessment=data.get("overall_assessment", ""),
                gaps_identified=data.get("gaps_identified", []),
                followup_needed=data.get("followup_needed", False)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate search results: {e}")
            # Return default evaluation
            return ResultEvaluationResult(
                evaluated_results=[],
                overall_assessment="Evaluation failed",
                gaps_identified=[],
                followup_needed=False
            )
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
    
    def _format_search_results_for_evaluation(self, search_results: List[SearchResult]) -> str:
        """Format search results as text for evaluation prompt"""
        formatted = []
        for i, result in enumerate(search_results):
            formatted.append(f"نتيجة {i + 1}:\n")
            formatted.append(f"العنوان: {result.title}\n")
            formatted.append(f"الرابط: {result.url}\n")
            formatted.append(f"الملخص: {result.snippet}\n")
            formatted.append("---\n")
        
        return "\n".join(formatted)

class CorpusKnowledgeRetriever(KnowledgeRetriever):
    """
    Retrieves relevant examples and knowledge from the corpus based on user constraints.
    
    Bridges the gap between user requirements (UserConstraints) and corpus search
    (CorpusManager) for the analysis layer in the poetry generation pipeline.
    """
    
    def __init__(self, corpus_manager: CorpusManager):
        super().__init__()
        self.corpus_manager = corpus_manager
    
    def search(self, constraints: UserConstraints, 
                         max_results: int = 5,
                         strategy: str = "best_match") -> CorpusRetrievalResult:
        """
        Retrieve relevant examples based on user constraints.
        
        Args:
            constraints: User requirements for the poem
            max_results: Maximum number of examples to return
            strategy: Retrieval strategy ("best_match", "diverse", "exact_match")
            
        Returns:
            RetrievalResult with examples and metadata
        """
        self.logger.info(f"Retrieving examples with strategy: {strategy}")
        
        if strategy == "exact_match":
            return self._exact_match_retrieval(constraints, max_results)
        elif strategy == "diverse":
            return self._diverse_retrieval(constraints, max_results)
        else:  # best_match (default)
            return self._best_match_retrieval(constraints, max_results)
    
    def _best_match_retrieval(self, constraints: UserConstraints, max_examples: int) -> CorpusRetrievalResult:
        """Best match strategy - use OR mode for broader matching"""

        # Use OR mode for broader matching
        search_criteria = self._constraints_to_search_criteria(constraints, mode="OR")
        examples = self.corpus_manager.search(search_criteria, limit=max_examples)

        return CorpusRetrievalResult(
            corpus_results=examples,
            total_found=len(examples),
            search_criteria=search_criteria,
            retrieval_strategy="best_match_or",
            metadata={"match_type": "or_mode"}
        )
    
    def _exact_match_retrieval(self, constraints: UserConstraints, max_examples: int) -> CorpusRetrievalResult:
        """Exact match strategy - only return poems matching all criteria"""
        
        search_criteria = self._constraints_to_search_criteria(constraints, mode="AND")
        examples = self.corpus_manager.search(search_criteria, limit=max_examples)
        
        return CorpusRetrievalResult(
            corpus_results=examples,
            total_found=len(examples),
            search_criteria=search_criteria,
            retrieval_strategy="exact_match",
            metadata={"strict_matching": True}
        )
    
    def _diverse_retrieval(self, constraints: UserConstraints, max_examples: int) -> CorpusRetrievalResult:
        """Diverse strategy - get varied examples covering different aspects"""
        
        all_examples = []
        metadata = {}
        
        # Get examples for each constraint type
        if constraints.meter:
            meter_examples = self.corpus_manager.find_by_meter(constraints.meter, limit=2)
            all_examples.extend(meter_examples)
            metadata["meter_examples"] = len(meter_examples)
        
        if constraints.theme:
            theme_examples = self.corpus_manager.find_by_theme(constraints.theme, limit=2)
            all_examples.extend([ex for ex in theme_examples if ex not in all_examples])
            metadata["theme_examples"] = len(theme_examples)
        
        if constraints.qafiya:
            qafiya_examples = self.corpus_manager.find_by_qafiya(constraints.qafiya, limit=2)
            all_examples.extend([ex for ex in qafiya_examples if ex not in all_examples])
            metadata["qafiya_examples"] = len(qafiya_examples)
        
        if constraints.poet_style:
            poet_examples = self.corpus_manager.find_by_poet(constraints.poet_style, limit=2)
            all_examples.extend([ex for ex in poet_examples if ex not in all_examples])
            metadata["poet_examples"] = len(poet_examples)
        
        # If still need more, add random samples
        if len(all_examples) < max_examples:
            remaining = max_examples - len(all_examples)
            random_examples = self.corpus_manager.sample_random(remaining * 2)
            all_examples.extend([ex for ex in random_examples if ex not in all_examples])
        
        final_examples = all_examples[:max_examples]
        
        search_criteria = self._constraints_to_search_criteria(constraints, mode="OR")
        
        return CorpusRetrievalResult(
            corpus_results=final_examples,
            total_found=len(all_examples),
            search_criteria=search_criteria,
            retrieval_strategy="diverse",
            metadata=metadata
        )
    
    def _constraints_to_search_criteria(self, constraints: UserConstraints, mode: str = "AND") -> SearchCriteria:
        """Convert UserConstraints to SearchCriteria"""
        
        return SearchCriteria(
            meter=constraints.meter,
            theme=constraints.theme,
            qafiya=constraints.qafiya,
            poet_name=constraints.poet_style,
            poet_era=constraints.era,
            language_type=constraints.register,
            min_verses=constraints.line_count - 1 if constraints.line_count else None,
            max_verses=constraints.line_count + 1 if constraints.line_count else None,
            keywords=constraints.keywords if constraints.keywords else None,
            search_mode=mode
        )
    
    def validate_constraints_feasibility(self, constraints: UserConstraints) -> Dict[str, Any]:
        """
        Check if constraints are feasible given the available corpus.
        
        Args:
            constraints: User constraints to validate
            
        Returns:
            Dictionary with feasibility information
        """
        validation = {
            "feasible": True,
            "issues": [],
            "suggestions": [],
            "available_options": {}
        }
        
        # Check meter availability
        if constraints.meter:
            if not self.corpus_manager.validate_meter_exists(constraints.meter):
                validation["feasible"] = False
                validation["issues"].append(f"Meter '{constraints.meter}' not found in corpus")
                
                # Suggest similar meters
                variations = self.corpus_manager.get_meter_variations(constraints.meter.split()[-1])
                if variations:
                    validation["suggestions"].append(f"Similar meters available: {variations}")
                    validation["available_options"]["meters"] = variations
        
        # Check theme availability
        if constraints.theme:
            if not self.corpus_manager.validate_theme_exists(constraints.theme):
                validation["feasible"] = False
                validation["issues"].append(f"Theme '{constraints.theme}' not found in corpus")
                
                # Suggest similar themes
                variations = self.corpus_manager.get_theme_variations(constraints.theme)
                if variations:
                    validation["suggestions"].append(f"Similar themes available: {variations}")
                    validation["available_options"]["themes"] = variations
        
        # Check qafiya availability
        if constraints.qafiya:
            if not self.corpus_manager.validate_qafiya_exists(constraints.qafiya):
                validation["feasible"] = False
                validation["issues"].append(f"Qafiya '{constraints.qafiya}' not found in corpus")
                
                # Suggest available qafiyas
                available_qafiyas = self.corpus_manager.get_available_qafiya()
                if available_qafiyas:
                    validation["suggestions"].append(f"Available qafiyas: {available_qafiyas[:5]}")
                    validation["available_options"]["qafiyas"] = available_qafiyas
        
        # Check if combination yields any results
        if validation["feasible"]:
            search_criteria = self._constraints_to_search_criteria(constraints, mode="AND")
            results = self.corpus_manager.search(search_criteria, limit=1)
            
            if not results:
                validation["feasible"] = False
                validation["issues"].append("No poems match the combination of constraints")
                validation["suggestions"].append("Consider relaxing some constraints or using OR search mode")
        
        return validation
    
    def get_constraint_statistics(self, constraints: UserConstraints) -> Dict[str, Any]:
        """
        Get statistics about how well constraints match the corpus.
        
        Args:
            constraints: User constraints to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Individual constraint statistics
        if constraints.meter:
            meter_results = self.corpus_manager.find_by_meter(constraints.meter)
            stats["meter_matches"] = len(meter_results)
        
        if constraints.theme:
            theme_results = self.corpus_manager.find_by_theme(constraints.theme)
            stats["theme_matches"] = len(theme_results)
        
        if constraints.qafiya:
            qafiya_results = self.corpus_manager.find_by_qafiya(constraints.qafiya)
            stats["qafiya_matches"] = len(qafiya_results)
        
        if constraints.poet_style:
            poet_results = self.corpus_manager.find_by_poet(constraints.poet_style)
            stats["poet_matches"] = len(poet_results)
        
        # Combined statistics
        and_criteria = self._constraints_to_search_criteria(constraints, mode="AND")
        and_results = self.corpus_manager.search(and_criteria)
        stats["exact_matches"] = len(and_results)
        
        or_criteria = self._constraints_to_search_criteria(constraints, mode="OR")
        or_results = self.corpus_manager.search(or_criteria)
        stats["partial_matches"] = len(or_results)
        
        # Coverage statistics
        total_poems = self.corpus_manager.get_total_poems()
        stats["coverage"] = {
            "exact_percentage": (len(and_results) / total_poems) * 100 if total_poems > 0 else 0,
            "partial_percentage": (len(or_results) / total_poems) * 100 if total_poems > 0 else 0
        }
        
        return stats
    
    def suggest_alternatives(self, constraints: UserConstraints) -> Dict[str, List[str]]:
        """
        Suggest alternative constraints that might yield better results.
        
        Args:
            constraints: Original constraints
            
        Returns:
            Dictionary with alternative suggestions
        """
        suggestions = {
            "meters": [],
            "themes": [],
            "qafiyas": [],
            "poets": []
        }
        
        # Get available options from corpus
        available_meters = self.corpus_manager.get_available_meters()
        available_themes = self.corpus_manager.get_available_themes()
        available_qafiyas = self.corpus_manager.get_available_qafiya()
        available_poets = self.corpus_manager.get_available_poets()
        
        # Suggest popular options (top 5 by frequency)
        if available_meters:
            suggestions["meters"] = available_meters[:5]
        
        if available_themes:
            suggestions["themes"] = available_themes[:5]
        
        if available_qafiyas:
            suggestions["qafiyas"] = available_qafiyas[:5]
        
        if available_poets:
            suggestions["poets"] = available_poets[:5]
        
        return suggestions
