# poet/models/search.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SearchQuery:
    """Represents a single search query with its purpose"""
    query: str
    purpose: str


@dataclass
class QueryGenerationResult:
    """Result from query generation prompt"""
    queries: List[SearchQuery]


@dataclass
class EvaluatedResult:
    """Represents evaluation of a single search result"""
    result_index: int
    relevance_score: int
    quality_score: int
    usefulness_score: int
    is_worth_following: bool
    key_insights: List[str]
    recommendation: str


@dataclass
class ResultEvaluationResult:
    """Result from result evaluation prompt"""
    evaluated_results: List[EvaluatedResult]
    overall_assessment: str
    gaps_identified: List[str]
    followup_needed: bool 