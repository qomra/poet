# poet/interface/cli_interface.py

import sys
import time
import signal
from typing import Dict, Any, Optional
from .base_interface import BaseInterface

class CLIInterface(BaseInterface):
    """Interactive CLI interface for poetry generation."""
    
    def __init__(self, agent):
        # Store the agent instead of config
        self.agent = agent
        self.running = False
        self._setup_signal_handlers()
        
        # Setup basic logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self) -> None:
        """Run the interactive CLI interface."""
        try:
            self.running = True
            
            print("🚀 Welcome to Poet - Arabic Poetry Generation System")
            print("=" * 60)
            print("Commands:")
            print("  - Type your poetry prompt and press Enter")
            print("  - Type 'quit' or 'exit' to close")
            print("  - Type 'help' for more information")
            print("  - Type 'status' to see pipeline status")
            print("-" * 60)
            
            # Main input loop
            while self.running:
                try:
                    # Get user input
                    user_input = input("\n🎭 Enter your poetry prompt: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() in ['help', 'h']:
                        self._show_help()
                        continue
                    elif user_input.lower() in ['status', 's']:
                        self._show_status()
                        continue
                    elif user_input.lower() in ['clear', 'cls']:
                        import os
                        os.system('clear' if os.name == 'posix' else 'cls')
                        continue
                    
                    # Process poetry generation
                    print(f"\n🔄 Processing: {user_input}")
                    print("-" * 40)
                    
                    result = self.agent.run_pipeline(user_input)
                    
                    if result['success']:
                        print("\n✅ Poetry generation completed successfully!")
                        if 'poem' in result:
                            print("\n📜 Generated Poem:")
                            print("=" * 40)
                            poem = result['poem']
                            print(str(poem))
                            print("=" * 40)
                        
                        if 'evaluation' in result:
                            print("\n📊 Evaluation:")
                            print("-" * 20)
                            eval_data = result['evaluation']
                            if isinstance(eval_data, dict):
                                for key, value in eval_data.items():
                                    if key != 'lines':  # Skip line details
                                        print(f"  {key}: {value}")
                            else:
                                print(f"  {eval_data}")
                    else:
                        print(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}")
                    
                    print("\n" + "=" * 60)
                    
                except KeyboardInterrupt:
                    print("\n⏹️  Interrupted by user")
                    break
                except EOFError:
                    print("\n👋 End of input, goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                
        finally:
            self.cleanup()
    
    def _show_help(self):
        """Show help information."""
        print("\n📖 Help - Available Commands:")
        print("  - Type any text to generate poetry")
        print("  - quit/exit/q: Close the application")
        print("  - help/h: Show this help message")
        print("  - status/s: Show pipeline status")
        print("  - clear/cls: Clear the screen")
        print("\n💡 Tips:")
        print("  - Be specific about your poetry requirements")
        print("  - Include details about meter, rhyme, theme, etc.")
        print("  - Examples:")
        print("    * 'اكتب قصيدة عن الحب في بحر الطويل'")
        print("    * 'قصيدة في مدح العلم على وزن الرمل'")
        print("    * 'شعر عن الوطن مع قافية في الهمزة'")
    
    def _show_status(self):
        """Show pipeline status."""
        try:
            pipeline_info = self.agent.get_pipeline_info()
            print(f"\n🔧 Pipeline Status:")
            print(f"  - Total nodes: {pipeline_info['node_count']}")
            print("  - Active nodes:")
            for node in pipeline_info['nodes']:
                print(f"    • {node['name']}")
        except Exception as e:
            print(f"❌ Could not get pipeline status: {e}")
    
    def is_completed(self) -> bool:
        """Check if the interface has completed."""
        return not self.running
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.running = False
        print("\n🧹 Cleaning up...")
    
    def mark_completed(self):
        """Mark the interface as completed."""
        self.running = False
