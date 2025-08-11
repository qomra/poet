# poet/interface/base_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseInterface(ABC):
    """Base interface class for running poet experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get("interface", {}).get("log_level", "INFO")
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @abstractmethod
    def run(self) -> None:
        """
        Run the interface.
        
        This method should run indefinitely until completion or interruption.
        """
        pass
    
    @abstractmethod
    def is_completed(self) -> bool:
        """
        Check if the interface has completed its task.
        
        Returns:
            True if completed, False otherwise
        """
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources before shutdown."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
