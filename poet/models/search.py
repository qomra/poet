# poet/models/search.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


@dataclass
class SearchQuery:
    """Represents a search query with context"""
    query: str
    purpose: str


@dataclass  
class QueryGenerationResult:
    """Result of query generation process"""
    queries: List[SearchQuery]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatedResult:
    """Represents an evaluated search result"""
    result_index: int
    is_worth_following: bool
    relevance_score: int
    analysis: str
    justification: str
    recommendation: str


@dataclass
class ResultEvaluationResult:
    """Result of search result evaluation"""
    evaluated_results: List[EvaluatedResult]
    summary_analysis: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataExample(ABC):
    """
    Base class for examples retrieved from different data sources.
    
    Contains the actual example content along with metadata about
    the search criteria used to find this example.
    """
    search_criteria: List[str]  # The criteria that matched this example (e.g., ['meter', 'qafiya'])
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def get_formatted_content(self) -> str:
        """Return formatted content suitable for inclusion in prompts"""
        pass
    
    @abstractmethod
    def get_source_description(self) -> str:
        """Return description of the data source"""
        pass


@dataclass
class CorpusExample(DataExample):
    """
    Example retrieved from the local poetry corpus.
    
    Contains poem data with metadata about the poet and source.
    """
    title: str = field(default="")
    verses: str = field(default="")
    meter: str = field(default="")
    qafiya: str = field(default="")
    theme: str = field(default="")
    poet_name: str = field(default="")
    poet_era: str = field(default="")
    
    def get_formatted_content(self) -> str:
        """Format the corpus example for prompt inclusion - language neutral"""
        formatted = f"{self.title} ({self.poet_name}, {self.poet_era})\n"
        formatted += f"{self.meter} | {self.qafiya}"
        if self.theme:
            formatted += f" | {self.theme}"
        formatted += f"\n{self.verses}\n"
        return formatted
    
    def get_source_description(self) -> str:
        """Return description of corpus source"""
        return f"corpus:{','.join(self.search_criteria)}"


@dataclass  
class WebExample(DataExample):
    """
    Example retrieved from web search.
    
    Contains web content with URL and relevance information.
    """
    title: str = field(default="")
    content: str = field(default="")
    url: str = field(default="")
    relevance_score: Optional[float] = field(default=None)
    
    def get_formatted_content(self) -> str:
        """Format the web example for prompt inclusion - language neutral"""
        formatted = f"{self.title}\n{self.content}\n"
        if self.relevance_score:
            formatted += f"Score: {self.relevance_score:.1f}\n"
        return formatted
        
    def get_source_description(self) -> str:
        """Return description of web source"""
        return f"web:{','.join(self.search_criteria)}" 