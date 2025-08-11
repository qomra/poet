#!/usr/bin/env python3
"""
Poet - Arabic Poetry Generation System

Main entry point for running poetry generation experiments.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add the poet package to the path
sys.path.insert(0, str(Path(__file__).parent))

from poet.core.agent import Agent
from poet.interface.cli_interface import CLIInterface
from poet.llm.llm_factory import get_real_llm_from_env
from poet.llm.base_llm import LLMConfig
from poet.llm.groq_adapter import GroqAdapter
from poet.llm.openai_adapter import OpenAIAdapter
from poet.llm.anthropic_adapter import AnthropicAdapter

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def create_llm(config: Dict[str, Any]) -> Any:
    """Create LLM instance based on configuration."""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "groq")
    
    # Get API key from config or environment
    api_key = llm_config.get("api_key")
    if not api_key:
        # Try to get from environment
        if provider == "groq":
            api_key = get_real_llm_from_env()
        elif provider == "openai":
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print(f"Error: No API key found for {provider}")
        sys.exit(1)
    
    # Create LLM config
    llm_config_obj = LLMConfig(
        model_name=llm_config.get("model"),
        api_key=api_key,
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens"),
        timeout=llm_config.get("timeout", 320)
    )
    
    # Create appropriate adapter
    if provider == "groq":
        return GroqAdapter(llm_config_obj)
    elif provider == "openai":
        return OpenAIAdapter(llm_config_obj)
    elif provider == "anthropic":
        return AnthropicAdapter(llm_config_obj)
    else:
        print(f"Error: Unknown LLM provider: {provider}")
        sys.exit(1)

def create_interface(config: Dict[str, Any]) -> CLIInterface:
    """Create interface based on configuration."""
    interface_config = config.get("interface", {})
    interface_type = interface_config.get("type", "cli")
    
    if interface_type == "cli":
        return CLIInterface(config)
    else:
        print(f"Error: Unknown interface type: {interface_type}")
        sys.exit(1)

def run_experiment(config: Dict[str, Any], user_prompt: str = None):
    """Run the poetry generation experiment."""
    print("🎭 Initializing Poet - Arabic Poetry Generation System")
    print("=" * 60)
    
    # Create LLM
    print("🤖 Creating LLM instance...")
    llm = create_llm(config)
    print(f"✅ LLM created: {llm.__class__.__name__}")
    
    # Create agent
    print("🧠 Creating agent...")
    agent = Agent(config, llm)
    print(f"✅ Agent created: {config.get('agent', {}).get('name', 'Unknown')}")
    
    # Create interface
    print("🖥️  Creating interface...")
    interface = create_interface(config)
    interface_type = config.get("interface", {}).get("type", "cli")
    print(f"✅ Interface created: {interface_type}")
    
    # If user prompt provided, run pipeline directly
    if user_prompt:
        print(f"\n📝 Running pipeline with prompt: {user_prompt}")
        result = agent.run_pipeline(user_prompt)
        
        if result.get("success"):
            print("\n🎉 Pipeline completed successfully!")
            print(f"📊 Final evaluation: {result.get('evaluation', 'N/A')}")
        else:
            print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        
        return
    
    # Run interface
    print("\n🚀 Starting interface...")
    try:
        interface.run()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running interface: {e}")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Poet - Arabic Poetry Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python poet.py -c config/simple_generation_refinement.yaml
  
  # Run with config and user prompt
  python poet.py -c config/simple_generation_refinement.yaml -p "Write a poem about nature"
  
  # Run with default config
  python poet.py
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/simple_generation_refinement.yaml",
        help="Path to configuration file (default: config/simple_generation_refinement.yaml)"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="User prompt for poetry generation (optional)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Poet 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    print(f"📁 Loading configuration from: {config_path}")
    config = load_config(args.config)
    
    # Run experiment
    run_experiment(config, args.prompt)

if __name__ == "__main__":
    main()
