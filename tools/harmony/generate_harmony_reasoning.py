#!/usr/bin/env python3
"""
Generate harmony reasoning using real LLMs.
This script loads captured fixture data and generates harmony reasoning.
"""

import asyncio
import json
import os
import sys
import argparse
from pathlib import Path

# Add the poet package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from poet.llm.llm_factory import get_real_llm_from_env
from poet.compiler.harmony import HarmonyCompiler
from poet.logging.harmony_capture import PipelineExecution, CapturedCall


def load_fixture_data():
    """Load harmony fixture data"""
    fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "harmony_test.json"
    
    if not fixture_path.exists():
        print("Error: Harmony fixture not found. Run capture_fixture.py first.")
        return None
    
    with open(fixture_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def reconstruct_execution(fixture_data):
    """Reconstruct PipelineExecution from fixture data"""
    execution = PipelineExecution(
        execution_id=fixture_data["execution_id"],
        started_at=fixture_data["started_at"],
        user_prompt=fixture_data["user_prompt"],
        initial_constraints=fixture_data["initial_constraints"]
    )
    
    # Reconstruct calls
    for call_data in fixture_data["calls"]:
        call = CapturedCall(
            call_id=call_data["call_id"],
            timestamp=call_data["timestamp"],
            component_name=call_data["component_name"],
            method_name=call_data["method_name"],
            call_type=call_data["call_type"],
            inputs=call_data["inputs"],
            outputs=call_data["outputs"],
            error=call_data.get("error"),
            llm_provider=call_data.get("llm_provider"),
            model_name=call_data.get("model_name"),
            prompt=call_data.get("prompt"),
            response=call_data.get("response"),
            tokens_used=call_data.get("tokens_used"),
            duration_ms=call_data.get("duration_ms"),
            success=call_data.get("success", True)
        )
        execution.add_call(call)
    
    # Set final outputs
    execution.final_poem = fixture_data.get("final_poem")
    execution.quality_assessment = fixture_data.get("quality_assessment")
    
    return execution


def main():
    """Main function to generate harmony reasoning"""
    parser = argparse.ArgumentParser(description="Generate harmony reasoning using real LLMs")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                       help="LLM provider to use (default: anthropic)")
    parser.add_argument("--max-tokens", type=int, default=4000,
                       help="Maximum tokens for response (default: 4000)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    
    args = parser.parse_args()
    
    print("Harmony Reasoning Generator")
    print("=" * 50)
    print(f"Provider: {args.provider}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    
    # Set environment variables
    os.environ["TEST_REAL_LLMS"] = "1"
    os.environ["REAL_LLM_PROVIDER"] = args.provider
    
    # Load fixture data
    fixture_data = load_fixture_data()
    if not fixture_data:
        return
    
    # Get real LLM
    llm = get_real_llm_from_env()
    if not llm:
        print("Error: No real LLM available. Check API keys and configuration.")
        print("Available providers: anthropic, openai")
        return
    
    print(f"Using LLM: {llm.__class__.__name__}")
    
    # Reconstruct execution
    execution = reconstruct_execution(fixture_data)
    print(f"Execution ID: {execution.execution_id}")
    print(f"User prompt: {execution.user_prompt}")
    print(f"Number of calls: {len(execution.calls)}")
    
    # Create compiler and generate reasoning
    compiler = HarmonyCompiler(llm)
    
    print("\nGenerating structured harmony data...")
    try:
        # Generate structured data
        structured_data = compiler.generate_structured_harmony(execution)
        
        print("\nGenerated Structured Harmony Data:")
        print("=" * 80)
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        print("=" * 80)
        
        # Convert to conversation format
        conversation = compiler.create_harmony_conversation(structured_data)
        reasoning = str(conversation)
        
        # Save structured data to file
        output_file = Path(__file__).parent / "temp" / f"{execution.execution_id}_harmony_{args.provider}.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reasoning)
        
        print(f"\nHarmony reasoning saved to: {output_file}")
        print(f"Reasoning length: {len(reasoning)} characters")
        
        # Also save the structured JSON data
        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        print(f"Structured data saved to: {json_file}")
        
        # Save metadata
        metadata = {
            "execution_id": execution.execution_id,
            "llm_provider": args.provider,
            "model": llm.__class__.__name__,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "reasoning_length": len(reasoning),
            "timestamp": execution.started_at
        }
        
        metadata_file = output_file.with_suffix('.metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadata saved to: {metadata_file}")
        
    except Exception as e:
        print(f"Error generating harmony reasoning: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 