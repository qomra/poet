# poet/analysis/knowledge_retriever.py

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from poet.models.constraints import UserConstraints
from poet.data.corpus_manager import CorpusManager, PoemRecord, SearchCriteria


@dataclass
class RetrievalResult:
    """Results from knowledge retrieval"""
    examples: List[PoemRecord]
    total_found: int
    search_criteria: SearchCriteria
    retrieval_strategy: str
    metadata: Dict[str, Any]

class KnowledgeRetriever:
    """
    Retrieves relevant examples and knowledge from the corpus based on user constraints.
    
    Bridges the gap between user requirements (UserConstraints) and corpus search
    (CorpusManager) for the analysis layer in the poetry generation pipeline.
    """
    
    def __init__(self, corpus_manager: CorpusManager):
        self.corpus_manager = corpus_manager
        self.logger = logging.getLogger(__name__)
    
    def retrieve_examples(self, constraints: UserConstraints, 
                         max_examples: int = 5,
                         strategy: str = "best_match") -> RetrievalResult:
        """
        Retrieve relevant examples based on user constraints.
        
        Args:
            constraints: User requirements for the poem
            max_examples: Maximum number of examples to return
            strategy: Retrieval strategy ("best_match", "diverse", "exact_match")
            
        Returns:
            RetrievalResult with examples and metadata
        """
        self.logger.info(f"Retrieving examples with strategy: {strategy}")
        
        if strategy == "exact_match":
            return self._exact_match_retrieval(constraints, max_examples)
        elif strategy == "diverse":
            return self._diverse_retrieval(constraints, max_examples)
        else:  # best_match (default)
            return self._best_match_retrieval(constraints, max_examples)
    
    def _best_match_retrieval(self, constraints: UserConstraints, max_examples: int) -> RetrievalResult:
        """Best match strategy - use OR mode for broader matching"""

        # Use OR mode for broader matching
        search_criteria = self._constraints_to_search_criteria(constraints, mode="OR")
        examples = self.corpus_manager.search(search_criteria, limit=max_examples)

        return RetrievalResult(
            examples=examples,
            total_found=len(examples),
            search_criteria=search_criteria,
            retrieval_strategy="best_match_or",
            metadata={"match_type": "or_mode"}
        )
    
    def _exact_match_retrieval(self, constraints: UserConstraints, max_examples: int) -> RetrievalResult:
        """Exact match strategy - only return poems matching all criteria"""
        
        search_criteria = self._constraints_to_search_criteria(constraints, mode="AND")
        examples = self.corpus_manager.search(search_criteria, limit=max_examples)
        
        return RetrievalResult(
            examples=examples,
            total_found=len(examples),
            search_criteria=search_criteria,
            retrieval_strategy="exact_match",
            metadata={"strict_matching": True}
        )
    
    def _diverse_retrieval(self, constraints: UserConstraints, max_examples: int) -> RetrievalResult:
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
        
        return RetrievalResult(
            examples=final_examples,
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
