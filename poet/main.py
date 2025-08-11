#!/usr/bin/env python3
"""
Main entry point for the Poet application using the dynamic pipeline system.
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.dynamic_agent import DynamicAgent
from llm.llm_factory import LLMFactory
from prompts.prompt_manager import PromptManager
from interface.cli_interface import CLIInterface


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Poet - Arabic Poetry Generation')
    parser.add_argument('--config', '-c', 
                       default='config/simple_generation_refinement.yaml',
                       help='Path to configuration file')
    parser.add_argument('--prompt', '-p',
                       help='Direct poetry prompt (optional)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        print(f"Configuration loaded from: {args.config}")
        
        # Create LLM
        llm_factory = LLMFactory()
        llm = llm_factory.create_llm(config['llm'])
        print(f"LLM initialized: {llm.__class__.__name__}")
        
        # Create prompt manager
        prompt_manager = PromptManager()
        print("Prompt manager initialized")
        
        # Create dynamic agent
        agent = DynamicAgent(config, llm, prompt_manager)
        print("Dynamic agent initialized")
        
        # Show pipeline info
        pipeline_info = agent.get_pipeline_info()
        print(f"\nPipeline configured with {pipeline_info['node_count']} nodes:")
        for node in pipeline_info['nodes']:
            print(f"  - {node['name']}")
        
        # Handle different modes
        if args.prompt:
            # Direct prompt mode
            print(f"\nGenerating poetry for prompt: {args.prompt}")
            result = agent.run_pipeline(args.prompt)
            
            if result['success']:
                print("\nPoetry generation completed successfully!")
                if 'poem' in result:
                    print(f"Generated poem: {result['poem']}")
                if 'evaluation' in result:
                    print(f"Evaluation: {result['evaluation']}")
            else:
                print(f"Generation failed: {result.get('error', 'Unknown error')}")
                
        elif args.interactive:
            # Interactive mode
            print("\nStarting interactive mode...")
            interface = CLIInterface(agent)
            interface.run()
            
        else:
            # Default: show help
            print("\nNo mode specified. Use --prompt for direct generation or --interactive for CLI mode.")
            print("Example: python -m poet.main --prompt 'اكتب قصيدة عن الحب'")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
