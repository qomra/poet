# poet/data/corpus_manager.py

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

try:
    from datasets import load_from_disk, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    Dataset = None

@dataclass
class PoemRecord:
    """Represents a single poem record from the corpus"""
    title: str
    meter: str
    verses: str
    qafiya: str
    theme: str
    url: str
    poet_name: str
    poet_description: str
    poet_url: str
    poet_era: str
    poet_location: str
    description: str
    language_type: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoemRecord':
        """Create PoemRecord from dictionary"""
        return cls(
            title=data.get('poem title', ''),
            meter=data.get('poem meter', ''),
            verses=data.get('poem verses', ''),
            qafiya=data.get('poem qafiya', ''),
            theme=data.get('poem theme', ''),
            url=data.get('poem url', ''),
            poet_name=data.get('poet name', ''),
            poet_description=data.get('poet description', ''),
            poet_url=data.get('poet url', ''),
            poet_era=data.get('poet era', ''),
            poet_location=data.get('poet location', ''),
            description=data.get('poem description', ''),
            language_type=data.get('poem language type', '')
        )
    
    def matches_meter(self, meter: str) -> bool:
        """Check if poem matches the specified meter"""
        if not meter or not self.meter:
            return False
        return meter.strip().lower() in self.meter.strip().lower()
    
    def matches_theme(self, theme: str) -> bool:
        """Check if poem matches the specified theme"""
        if not theme or not self.theme:
            return False
        return theme.strip().lower() in self.theme.strip().lower()
    
    def matches_qafiya(self, qafiya: str) -> bool:
        """Check if poem matches the specified qafiya"""
        if not qafiya or not self.qafiya:
            return False
        return qafiya.strip().lower() in self.qafiya.strip().lower()
    
    def matches_poet(self, poet_name: str) -> bool:
        """Check if poem is by the specified poet"""
        if not poet_name or not self.poet_name:
            return False
        return poet_name.strip().lower() in self.poet_name.strip().lower()
    
    def get_verse_count(self) -> int:
        """Get number of verses in the poem"""
        if not self.verses:
            return 0
        
        # Handle both string (with newlines) and list formats
        if isinstance(self.verses, str):
            verses_list = self.verses.split('\n')
        else:
            verses_list = self.verses
            
        return len([v for v in verses_list if v and str(v).strip()])

@dataclass
class SearchCriteria:
    """Search criteria for corpus queries"""
    meter: Optional[str] = None
    theme: Optional[str] = None
    qafiya: Optional[str] = None
    poet_name: Optional[str] = None
    poet_era: Optional[str] = None
    poet_location: Optional[str] = None
    language_type: Optional[str] = None
    min_verses: Optional[int] = None
    max_verses: Optional[int] = None
    keywords: Optional[List[str]] = None
    search_mode: str = "AND"  # "AND" or "OR"
    
class CorpusManager:
    """
    Manages access to the ashaar poetry corpus using HuggingFace datasets.
    
    Provides efficient search, filtering, and retrieval of poems
    based on various criteria like meter, theme, poet, etc.
    """
    
    def __init__(self, local_knowledge_path: str):
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        self.local_knowledge_path = Path(local_knowledge_path)
        self.dataset_path = self.local_knowledge_path / "ashaar"
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self._dataset: Optional[Dataset] = None
        self._poems: List[PoemRecord] = []
        self._loaded = False
        
        # Indices for fast lookup
        self._meter_index: Dict[str, List[int]] = defaultdict(list)
        self._theme_index: Dict[str, List[int]] = defaultdict(list)
        self._poet_index: Dict[str, List[int]] = defaultdict(list)
        self._era_index: Dict[str, List[int]] = defaultdict(list)
        self._qafiya_index: Dict[str, List[int]] = defaultdict(list)

        # Statistics
        self._stats: Dict[str, Any] = {}
        
        self.logger.info(f"Initialized CorpusManager with local knowledge path: {local_knowledge_path}")
    
    def load_corpus(self, force_reload: bool = False) -> None:
        """
        Load the corpus using HuggingFace datasets.
        
        Args:
            force_reload: Force reload even if already loaded
        """
        if self._loaded and not force_reload:
            return
        
        self.logger.info(f"Loading ashaar dataset from {self.dataset_path}...")
        
        try:
            # Load dataset using HuggingFace datasets
            self._dataset = load_from_disk(str(self.dataset_path))
            
            # Convert to PoemRecord objects
            self._poems = []
            for item in self._dataset:
                try:
                    poem = PoemRecord.from_dict(item)
                    self._poems.append(poem)
                except Exception as e:
                    self.logger.warning(f"Error processing poem record: {e}")
            
            self._build_indices()
            self._compute_statistics()
            self._loaded = True
            
            self.logger.info(f"Loaded {len(self._poems)} poems from ashaar dataset")
            
        except Exception as e:
            self.logger.error(f"Failed to load ashaar dataset: {e}")
            raise FileNotFoundError(f"Could not load ashaar dataset from {self.dataset_path}: {e}")
    
    def _build_indices(self) -> None:
        """Build search indices for fast lookup"""
        self.logger.info("Building search indices...")
        
        self._meter_index.clear()
        self._theme_index.clear()
        self._poet_index.clear()
        self._era_index.clear()
        self._qafiya_index.clear()
        for i, poem in enumerate(self._poems):
            # Meter index
            if poem.meter:
                meter_key = poem.meter.strip().lower()
                self._meter_index[meter_key].append(i)
            
            # Theme index
            if poem.theme:
                theme_key = poem.theme.strip().lower()
                self._theme_index[theme_key].append(i)
            
            # Poet index
            if poem.poet_name:
                poet_key = poem.poet_name.strip().lower()
                self._poet_index[poet_key].append(i)
            
            # Era index
            if poem.poet_era:
                era_key = poem.poet_era.strip().lower()
                self._era_index[era_key].append(i)

            # Qafiya index
            if poem.qafiya:
                qafiya_key = poem.qafiya.strip().lower()
                self._qafiya_index[qafiya_key].append(i)

    def _compute_statistics(self) -> None:
        """Compute corpus statistics"""
        if not self._poems:
            return
        
        meters = set()
        themes = set()
        poets = set()
        eras = set()
        qafiya = set()
        verse_counts = []
        
        for poem in self._poems:
            if poem.meter:
                meters.add(poem.meter.strip())
            if poem.theme:
                themes.add(poem.theme.strip())
            if poem.poet_name:
                poets.add(poem.poet_name.strip())
            if poem.poet_era:
                eras.add(poem.poet_era.strip())
            if poem.qafiya:
                qafiya.add(poem.qafiya.strip())
            verse_counts.append(poem.get_verse_count())
        
        self._stats = {
            'total_poems': len(self._poems),
            'unique_meters': len(meters),
            'unique_themes': len(themes),
            'unique_poets': len(poets),
            'unique_eras': len(eras),
            'avg_verses': sum(verse_counts) / len(verse_counts) if verse_counts else 0,
            'min_verses': min(verse_counts) if verse_counts else 0,
            'max_verses': max(verse_counts) if verse_counts else 0,
            'meters': sorted(list(meters)),
            'themes': sorted(list(themes)),
            'poets': sorted(list(poets)),
            'eras': sorted(list(eras)),
            'qafiya': sorted(list(qafiya))
        }
    
    def search(self, criteria: SearchCriteria, limit: Optional[int] = None) -> List[PoemRecord]:
        """
        Search poems based on criteria.
        
        Args:
            criteria: Search criteria
            limit: Maximum number of results
            
        Returns:
            List of matching poems
        """
        self.load_corpus()
        
        # Apply search logic based on mode
        if criteria.search_mode.upper() == "OR":
            candidates = set()  # Start empty for OR
            
            # Apply filters using OR logic
            if criteria.meter:
                meter_matches = set()
                meter_key = criteria.meter.strip().lower()
                for indexed_meter, poem_indices in self._meter_index.items():
                    if meter_key in indexed_meter:
                        meter_matches.update(poem_indices)
                candidates.update(meter_matches)
            
            if criteria.theme:
                theme_matches = set()
                theme_key = criteria.theme.strip().lower()
                for indexed_theme, poem_indices in self._theme_index.items():
                    if theme_key in indexed_theme:
                        theme_matches.update(poem_indices)
                candidates.update(theme_matches)
            
            if criteria.qafiya:
                qafiya_matches = set()
                qafiya_key = criteria.qafiya.strip().lower()
                for indexed_qafiya, poem_indices in self._qafiya_index.items():
                    if qafiya_key in indexed_qafiya:
                        qafiya_matches.update(poem_indices)
                candidates.update(qafiya_matches)
            
            if criteria.poet_name:
                poet_matches = set()
                poet_key = criteria.poet_name.strip().lower()
                for indexed_poet, poem_indices in self._poet_index.items():
                    if poet_key in indexed_poet:
                        poet_matches.update(poem_indices)
                candidates.update(poet_matches)
            
            if criteria.poet_era:
                era_matches = set()
                era_key = criteria.poet_era.strip().lower()
                for indexed_era, poem_indices in self._era_index.items():
                    if era_key in indexed_era:
                        era_matches.update(poem_indices)
                candidates.update(era_matches)
            
            # If no indexed criteria provided, return all poems
            if not candidates and not any([criteria.meter, criteria.theme, criteria.qafiya, criteria.poet_name, criteria.poet_era]):
                candidates = set(range(len(self._poems)))
                
        else:  # AND mode (default)
            candidates = set(range(len(self._poems)))
            
            # Apply filters using AND logic
            if criteria.meter:
                meter_matches = set()
                meter_key = criteria.meter.strip().lower()
                for indexed_meter, poem_indices in self._meter_index.items():
                    if meter_key in indexed_meter:
                        meter_matches.update(poem_indices)
                candidates &= meter_matches
            
            if criteria.theme:
                theme_matches = set()
                theme_key = criteria.theme.strip().lower()
                for indexed_theme, poem_indices in self._theme_index.items():
                    if theme_key in indexed_theme:
                        theme_matches.update(poem_indices)
                candidates &= theme_matches
            
            if criteria.qafiya:
                qafiya_matches = set()
                qafiya_key = criteria.qafiya.strip().lower()
                for indexed_qafiya, poem_indices in self._qafiya_index.items():
                    if qafiya_key in indexed_qafiya:
                        qafiya_matches.update(poem_indices)
                candidates &= qafiya_matches
            
            if criteria.poet_name:
                poet_matches = set()
                poet_key = criteria.poet_name.strip().lower()
                for indexed_poet, poem_indices in self._poet_index.items():
                    if poet_key in indexed_poet:
                        poet_matches.update(poem_indices)
                candidates &= poet_matches
            
            if criteria.poet_era:
                era_matches = set()
                era_key = criteria.poet_era.strip().lower()
                for indexed_era, poem_indices in self._era_index.items():
                    if era_key in indexed_era:
                        era_matches.update(poem_indices)
                candidates &= era_matches
        
        # Apply additional filters
        results = []
        for i in candidates:
            poem = self._poems[i]
            
            # Verse count filters
            verse_count = poem.get_verse_count()
            if criteria.min_verses and verse_count < criteria.min_verses:
                continue
            if criteria.max_verses and verse_count > criteria.max_verses:
                continue
            
            # Language type filter
            if criteria.language_type and poem.language_type and criteria.language_type.lower() not in poem.language_type.lower():
                continue
            
            # Location filter
            if criteria.poet_location and criteria.poet_location.lower() not in poem.poet_location.lower():
                continue
            
            # Keywords filter
            if criteria.keywords:
                text_content = f"{poem.title} {poem.verses} {poem.description}".lower()
                if criteria.search_mode.upper() == "OR":
                    # OR mode: at least one keyword must match
                    if not any(keyword.lower() in text_content for keyword in criteria.keywords):
                        continue
                else:
                    # AND mode: all keywords must match
                    if not all(keyword.lower() in text_content for keyword in criteria.keywords):
                        continue
            
            results.append(poem)
            
            if limit and len(results) >= limit:
                break
        
        self.logger.info(f"Found {len(results)} poems matching criteria")
        return results
    
    def find_by_meter(self, meter: str, limit: Optional[int] = None) -> List[PoemRecord]:
        """Find poems by meter"""
        criteria = SearchCriteria(meter=meter)
        return self.search(criteria, limit)
    
    def find_by_theme(self, theme: str, limit: Optional[int] = None) -> List[PoemRecord]:
        """Find poems by theme"""
        criteria = SearchCriteria(theme=theme)
        return self.search(criteria, limit)
    
    def find_by_poet(self, poet_name: str, limit: Optional[int] = None) -> List[PoemRecord]:
        """Find poems by poet"""
        criteria = SearchCriteria(poet_name=poet_name)
        return self.search(criteria, limit)
    
    def find_by_qafiya(self, qafiya: str, limit: Optional[int] = None) -> List[PoemRecord]:
        """Find poems by qafiya"""
        criteria = SearchCriteria(qafiya=qafiya)
        return self.search(criteria, limit)
    
    def get_examples_for_constraints(self, meter: Optional[str] = None, 
                                   theme: Optional[str] = None,
                                   qafiya: Optional[str] = None,
                                   verse_count: Optional[int] = None,
                                   limit: int = 5) -> List[PoemRecord]:
        """
        Get example poems matching specific constraints.
        
        Args:
            meter: Desired meter
            theme: Desired theme
            verse_count: Desired number of verses
            limit: Maximum examples to return
            
        Returns:
            List of example poems
        """
        criteria = SearchCriteria(
            meter=meter,
            theme=theme,
            qafiya=qafiya,
            min_verses=verse_count - 1 if verse_count else None,
            max_verses=verse_count + 1 if verse_count else None
        )
        
        return self.search(criteria, limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        self.load_corpus()
        return self._stats.copy()
    
    def get_available_meters(self) -> List[str]:
        """Get list of available meters"""
        self.load_corpus()
        return self._stats.get('meters', [])
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes"""
        self.load_corpus()
        return self._stats.get('themes', [])
    
    def get_available_poets(self) -> List[str]:
        """Get list of available poets"""
        self.load_corpus()
        return self._stats.get('poets', [])
    
    def get_available_qafiya(self) -> List[str]:
        """Get list of available qafiya"""
        self.load_corpus()
        return self._stats.get('qafiya', [])
    
    def sample_random(self, count: int = 10) -> List[PoemRecord]:
        """Get random sample of poems"""
        import random
        
        self.load_corpus()
        
        if count >= len(self._poems):
            return self._poems.copy()
        
        return random.sample(self._poems, count)
    
    def validate_meter_exists(self, meter: str) -> bool:
        """Check if meter exists in corpus"""
        self.load_corpus()
        meter_key = meter.strip().lower()
        return any(meter_key in indexed_meter for indexed_meter in self._meter_index.keys())
    
    def validate_theme_exists(self, theme: str) -> bool:
        """Check if theme exists in corpus"""
        self.load_corpus()
        theme_key = theme.strip().lower()
        return any(theme_key in indexed_theme for indexed_theme in self._theme_index.keys())
    
    def validate_qafiya_exists(self, qafiya: str) -> bool:
        """Check if qafiya exists in corpus"""
        self.load_corpus()
        qafiya_key = qafiya.strip().lower()
        return any(qafiya_key in indexed_qafiya for indexed_qafiya in self._qafiya_index.keys())
    
    def get_meter_variations(self, meter: str) -> List[str]:
        """Get all meter variations containing the given meter"""
        self.load_corpus()
        meter_key = meter.strip().lower()
        variations = []
        
        for indexed_meter in self._meter_index.keys():
            if meter_key in indexed_meter:
                # Find the original meter name from poems
                for poem_idx in self._meter_index[indexed_meter][:1]:  # Just get one example
                    variations.append(self._poems[poem_idx].meter.strip())
                    break
        
        return sorted(list(set(variations)))
    
    def get_theme_variations(self, theme: str) -> List[str]:
        """Get all theme variations containing the given theme"""
        self.load_corpus()
        theme_key = theme.strip().lower()
        variations = []
        
        for indexed_theme in self._theme_index.keys():
            if theme_key in indexed_theme:
                # Find the original theme name from poems
                for poem_idx in self._theme_index[indexed_theme][:1]:  # Just get one example
                    variations.append(self._poems[poem_idx].theme.strip())
                    break
        
        return sorted(list(set(variations)))
    
    def is_loaded(self) -> bool:
        """Check if corpus is loaded"""
        return self._loaded
    
    def get_total_poems(self) -> int:
        """Get total number of poems"""
        self.load_corpus()
        return len(self._poems)
    
    def get_raw_dataset(self) -> Optional[Dataset]:
        """Get the raw HuggingFace dataset"""
        self.load_corpus()
        return self._dataset
