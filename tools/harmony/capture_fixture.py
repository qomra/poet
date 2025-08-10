#!/usr/bin/env python3
"""
Script to capture harmony integration output for testing.
This script either loads from existing fixtures or runs the full workflow to generate execution data.

Usage:
    python capture_fixture.py
    
Output files:
    - {execution_id}_raw.json: Raw execution data
    - {execution_id}_harmony.txt: Harmony conversation format
    - {execution_id}_structured.json: Structured JSON for openai_harmony
    - harmony_structured_test.json: Structured data in fixtures directory
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the poet package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from poet.llm.anthropic_adapter import AnthropicAdapter
from poet.prompts.prompt_manager import PromptManager
from poet.logging.integration import HarmonyIntegration
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.analysis.qafiya_selector import QafiyaSelector
from poet.analysis.bahr_selector import BahrSelector
from poet.evaluation.poem import PoemEvaluator
from poet.refinement.refiner_chain import RefinerChain
from poet.evaluation.poem import EvaluationType
from poet.refinement import LineCountRefiner, ProsodyRefiner, QafiyaRefiner, TashkeelRefiner
from poet.logging.capture_middleware import capture_component
from poet.compiler.harmony import HarmonyCompiler


async def load_or_generate_execution():
    """Load execution from fixture if available, otherwise generate it"""
    
    fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "harmony_test.json"
    
    if fixture_path.exists():
        print("=== Loading execution from existing fixture ===")
        with open(fixture_path, 'r', encoding='utf-8') as f:
            fixture_data = json.load(f)
        
        # Create execution object from fixture data
        from poet.logging.harmony_capture import PipelineExecution, CapturedCall
        from datetime import datetime
        
        # Convert fixture data back to PipelineExecution object
        execution = PipelineExecution(
            execution_id=fixture_data.get("execution_id", "test-execution"),
            started_at=datetime.fromisoformat(fixture_data.get("started_at", "2025-01-01T00:00:00Z")),
            user_prompt=fixture_data.get("user_prompt", ""),
            initial_constraints=fixture_data.get("initial_constraints", {}),
            final_poem=fixture_data.get("final_poem"),
            quality_assessment=fixture_data.get("quality_assessment"),
            total_llm_calls=fixture_data.get("total_llm_calls", 0),
            total_tokens=fixture_data.get("total_tokens", 0)
        )
        
        # Set additional fields if they exist in the fixture
        if fixture_data.get("completed_at"):
            execution.completed_at = datetime.fromisoformat(fixture_data["completed_at"])
        if fixture_data.get("total_duration_ms"):
            execution.total_duration_ms = fixture_data["total_duration_ms"]
        
        # Convert calls back to CapturedCall objects
        for call_data in fixture_data.get("calls", []):
            call = CapturedCall(
                call_id=call_data.get("call_id", ""),
                timestamp=datetime.fromisoformat(call_data.get("timestamp", "2025-01-01T00:00:00Z")),
                component_name=call_data.get("component_name", call_data.get("component", "")),  # Handle both old and new field names
                method_name=call_data.get("method_name", call_data.get("method", "")),  # Handle both old and new field names
                call_type=call_data.get("call_type", ""),  # Use call_type if available
                inputs=call_data.get("inputs", call_data.get("input", {})),  # Handle both old and new field names
                outputs=call_data.get("outputs", call_data.get("output", {})),  # Handle both old and new field names
                error=call_data.get("error"),
                llm_provider=call_data.get("llm_provider"),
                model_name=call_data.get("model_name"),
                prompt=call_data.get("prompt"),
                response=call_data.get("response"),
                tokens_used=call_data.get("tokens_used", 0),
            )
            execution.calls.append(call)
        
        print(f"Loaded execution ID: {execution.execution_id}")
        return execution, True  # Indicate that it was loaded from fixture
    
    print("=== Generating new execution (fixture not found) ===")
    return await generate_execution()


async def generate_execution():
    """Generate execution by running the full workflow"""
    
    # Initialize components
    from poet.llm.base_llm import LLMConfig
    
    # Read API key from fixtures
    llms_fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "llms.json"
    with open(llms_fixture_path, 'r', encoding='utf-8') as f:
        llms_config = json.load(f)

    # Read test data
    data_fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "test_data.json"
    with open(data_fixture_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    example = test_data[0]
    user_prompt = example["prompt"]["text"]
    expected_constraints = example["agent"]["constraints"]
    
    anthropic_config = llms_config["anthropic"]
    llm_config = LLMConfig(
        model_name=anthropic_config["model"],
        temperature=0.7,
        api_key=anthropic_config["api_key"],
        max_tokens=anthropic_config["max_tokens"]
    )
    llm = AnthropicAdapter(llm_config)
    prompt_manager = PromptManager()
    
    # Create test data
    constraint_data = {
        "meter": expected_constraints["meter"],
        "qafiya": expected_constraints["qafiya"],
        "theme": expected_constraints["theme"],
        "tone": expected_constraints["tone"]
    }
    
    print(f"User prompt: {user_prompt}")
    
    # Start captured execution
    HarmonyIntegration.start_captured_execution(user_prompt, constraint_data)
    
    # Create captured components
    qafiya_selector_orig = QafiyaSelector(llm, prompt_manager)
    qafiya_selector = capture_component(qafiya_selector_orig, "QafiyaSelector")
    bahar_selector_orig = BahrSelector(llm, prompt_manager)
    bahar_selector = capture_component(bahar_selector_orig, "BahrSelector")
    poem_evaluator_orig = PoemEvaluator(llm)
    poem_evaluator = capture_component(poem_evaluator_orig, "PoemEvaluator")
    
    refiners = [
        LineCountRefiner(llm, prompt_manager),
        ProsodyRefiner(llm, prompt_manager),
        QafiyaRefiner(llm, prompt_manager),
        TashkeelRefiner(llm, prompt_manager)
    ]
    refiner_chain_orig = RefinerChain(refiners, llm, max_iterations=1)
    refiner_chain = capture_component(refiner_chain_orig, "RefinerChain")
    
    # Run the workflow
    constraints = Constraints(
        meter=constraint_data["meter"],
        qafiya=constraint_data["qafiya"],
        theme=constraint_data["theme"],
        tone=constraint_data["tone"]
    )
    
    print("Step 1: Enriching constraints...")
    enriched_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
    enriched_constraints = bahar_selector.select_bahr(enriched_constraints, user_prompt)
    
    # Use hardcoded initial poem for test predictability
    initial_poem = LLMPoem(
        verses=[
            "تَبْكي السَّحابُ عَلَى فِراقِكَ يا شَفَقْ",
            "وَالْقَلْبُ يَشْكُو مَا أَصابَهُ مِنْ حُرَقْ",
            "فِي لَيْلَةٍ حَزِنَتْ عَلَيْكَ دُجَى السُّهَا",
            "وَجْدِي عَلَيْكَ يَزِيدُ مَعْ نَفْحِ العَبَقْ"
        ],
        llm_provider='test', 
        model_name='hardcoded', 
        constraints=enriched_constraints.to_dict()
    )
    
    print("Step 2: Refining poem...")
    refined_poem, _ = await refiner_chain.refine(
        initial_poem,
        enriched_constraints,
        target_quality=1.0
    )
    
    print("Step 3: Final evaluation...")
    final_evaluation = poem_evaluator.evaluate_poem(
        refined_poem,
        enriched_constraints,
        [EvaluationType.PROSODY, EvaluationType.QAFIYA]
    )
    
    # Get the execution data
    from poet.logging.harmony_capture import get_capture
    capture = get_capture()
    execution = capture.get_execution()
    # put final_poem to execution
    execution.final_poem = refined_poem.to_dict()
    execution.quality_assessment = final_evaluation.to_dict()
    
    if execution:
        print(f"Generated execution ID: {execution.execution_id}")
        print(f"Total calls captured: {len(execution.calls)}")
        print(f"Total LLM calls: {execution.total_llm_calls}")
        print(f"Total tokens: {execution.total_tokens}")
    
    return execution, False  # Return tuple with flag indicating it was generated


async def generate_harmony_data(execution, llm):
    """Generate structured harmony data from execution"""
    
    print("=== Generating structured harmony data ===")
    
    try:
        compiler = HarmonyCompiler(llm)
        # Generate structured harmony data
        print("🔧 Generating structured harmony data...")
        structured_data = compiler.generate_structured_harmony(execution)
        
        if structured_data is None:
            print("✗ Failed to generate structured harmony data")
            return None, None
        
        # Debug: Print the actual structure of structured_data
        print(f"📊 Structured data keys: {list(structured_data.keys())}")
        print(f"📊 Structured data type: {type(structured_data)}")
        print(f"📊 Structured data content: {structured_data}")
        
        # Print structured data summary
        print("📋 Generated structured harmony data:")
        if 'system_message' in structured_data:
            print(f"  - System message: {structured_data['system_message']['model_identity']}")
        else:
            print("  - System message: Not found in structured data")
        
        if 'messages' in structured_data:
            print(f"  - Messages count: {len(structured_data['messages'])}")
        else:
            print("  - Messages: Not found in structured data")
        
        # Convert to conversation format
        try:
            conversation = compiler.create_harmony_conversation(structured_data)
            if isinstance(conversation, dict) and 'error' in conversation:
                print(f"  - Conversation creation failed: {conversation['error']}")
                messages = "Failed to create conversation"
            else:
                print(f"  - Conversation created successfully: {type(conversation)}")
                messages = str(conversation) if conversation else "Failed to create conversation"
        except Exception as e:
            print(f"⚠️ Warning: Could not create harmony conversation: {e}")
            messages = "Failed to create conversation"
        
        return structured_data, messages
        
    except Exception as e:
        print(f"✗ Error generating harmony data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_outputs(execution, structured_data, conversation_str,was_loaded):
    """Save all outputs to files"""
    
    # Save structured harmony data
    if structured_data:
        # Save to fixtures directory
        fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "harmony_structured_test.json"
        with open(fixture_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        print(f"Structured fixture saved to: {fixture_path}")
    
    # Save harmony conversation
    if conversation_str:
        # Save to fixtures directory
        fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "harmony_test_output.txt"
        with open(fixture_path, 'w', encoding='utf-8') as f:
            f.write(conversation_str)
        print(f"Harmony output saved to: {fixture_path}")
    
    # Save execution fixture
    if not was_loaded:  
        fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "harmony_test.json"
        fixture_path.parent.mkdir(exist_ok=True)
        
        with open(fixture_path, 'w', encoding='utf-8') as f:
            json.dump(execution.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"Execution fixture saved to: {fixture_path}")


async def main():
    """Main function to orchestrate the process"""
    
    # Load or generate execution
    result = await load_or_generate_execution()
    if isinstance(result, tuple):
        execution, was_loaded = result
    else:
        # Handle backward compatibility
        execution = result
        was_loaded = False
    
    if not execution:
        print("Failed to load or generate execution!")
        return
    
    # Generate harmony data
    from poet.llm.base_llm import LLMConfig
    from poet.llm.anthropic_adapter import AnthropicAdapter
    
    # Read API key for harmony generation
    llms_fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "llms.json"
    with open(llms_fixture_path, 'r', encoding='utf-8') as f:
        llms_config = json.load(f)
    
    anthropic_config = llms_config["anthropic"]
    llm_config = LLMConfig(
        model_name=anthropic_config["model"],
        temperature=0.7,
        api_key=anthropic_config["api_key"],
        max_tokens=anthropic_config["max_tokens"]
    )
    llm = AnthropicAdapter(llm_config)
    
    structured_data, conversation_str = await generate_harmony_data(execution, llm)
    
    # Only save outputs if execution was generated (not loaded from fixture)

    save_outputs(execution, structured_data, conversation_str,was_loaded)
    
    print("\n=== Process completed successfully ===")


if __name__ == "__main__":
    asyncio.run(main()) 