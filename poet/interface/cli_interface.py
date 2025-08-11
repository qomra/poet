# poet/interface/cli_interface.py

import sys
import time
import signal
import threading
from typing import Dict, Any, Optional
from .base_interface import BaseInterface

class LoggingInterceptor:
    """Intercepts logging calls and prints them to console."""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level.upper()
        self.original_handlers = {}
        self._setup_interception()
    
    def _setup_interception(self):
        """Setup logging interception."""
        import logging
        
        # Store original handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            self.original_handlers[id(handler)] = handler
            root_logger.removeHandler(handler)
        
        # Add our custom handler
        custom_handler = logging.StreamHandler(sys.stdout)
        custom_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(custom_handler)
        root_logger.setLevel(logging.DEBUG)
    
    def restore_logging(self):
        """Restore original logging configuration."""
        import logging
        
        root_logger = logging.getLogger()
        
        # Remove our custom handler
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Restore original handlers
        for handler in self.original_handlers.values():
            root_logger.addHandler(handler)

class CLIInterface(BaseInterface):
    """CLI interface for running poet experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.running = False
        self.completed = False
        self.log_interceptor = None
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _setup_logging_interception(self):
        """Setup logging interception for CLI output."""
        log_level = self.config.get("interface", {}).get("log_level", "INFO")
        self.log_interceptor = LoggingInterceptor(log_level)
    
    def _restore_logging(self):
        """Restore original logging configuration."""
        if self.log_interceptor:
            self.log_interceptor.restore_logging()
    
    def run(self) -> None:
        """Run the CLI interface."""
        try:
            self._setup_logging_interception()
            self.running = True
            self.completed = False
            
            print(f"🚀 Starting {self.config.get('interface', {}).get('name', 'Poet Experiment')}")
            print("Press Ctrl+C to stop")
            print("-" * 50)
            
            # Main loop
            while self.running and not self.completed:
                try:
                    # Check for completion
                    if self.is_completed():
                        self.completed = True
                        break
                    
                    # Small delay to prevent busy waiting
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    print("\n⏹️  Interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    break
            
            if self.completed:
                print("✅ Experiment completed successfully!")
            else:
                print("⏹️  Experiment stopped")
                
        finally:
            self._restore_logging()
            self.cleanup()
    
    def is_completed(self) -> bool:
        """Check if the interface has completed."""
        return self.completed
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.running = False
        self._restore_logging()
    
    def mark_completed(self):
        """Mark the interface as completed."""
        self.completed = True
