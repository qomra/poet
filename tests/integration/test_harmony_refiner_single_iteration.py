import os
import pytest
import json
from pathlib import Path

# Import your project components
from poet.evaluation.poem import PoemEvaluator, EvaluationType
from poet.generation.poem_generator import SimplePoemGenerator
from poet.refinement import RefinerChain, ProsodyRefiner, QafiyaRefiner
from poet.analysis.qafiya_selector import QafiyaSelector
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.prompts.prompt_manager import PromptManager
from poet.llm.llm_factory import get_real_llm_from_env

# Import the Harmony capture components
from poet.logging.capture_middleware import capture_component
from poet.logging.integration import HarmonyIntegration


@pytest.mark.integration
@pytest.mark.real_data
class TestHarmonyCaptureWorkflow:
    """
    Integration test for the full poetry generation workflow with Harmony capture enabled.
    This test will:
    1. Load from fixtures/harmony_test.json if available, or run the workflow to generate data
    2. Start a capture session.
    3. Wrap all agent components (parser, generator, evaluator, refiners) to automatically capture their calls.
    4. Run the end-to-end poetry generation process.
    5. Complete the capture, which generates a raw execution log (JSON) and a final,
       LLM-generated Harmony reasoning trace.
    6. Assert that the outputs are generated correctly.
    """

    @pytest.fixture(scope="class")
    def prompt_manager(self):
        """Create a PromptManager instance."""
        return PromptManager()

    @pytest.fixture(scope="class")
    def real_llm(self):
        """Get the real LLM from environment variables."""
        if os.environ.get("TEST_REAL_LLMS") != "true":
            pytest.skip("Skipping real LLM tests as TEST_REAL_LLMS is not set to 'true'")
        return get_real_llm_from_env()

    @pytest.fixture(scope="class")
    def test_data(self):
        """Load test data from fixtures."""
        test_file = Path(__file__).parent.parent / "fixtures" / "test_data.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_or_generate_execution(self, real_llm, prompt_manager, test_data):
        """
        Load execution from fixture if available, otherwise generate it by running the workflow.
        Returns tuple of (execution, was_loaded_from_fixture)
        """
        fixture_path = Path(__file__).parent.parent / "fixtures" / "harmony_test.json"
        
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
                    component_name=call_data.get("component_name", call_data.get("component", "")),
                    method_name=call_data.get("method_name", call_data.get("method", "")),
                    call_type=call_data.get("call_type", ""),
                    inputs=call_data.get("inputs", call_data.get("input", {})),
                    outputs=call_data.get("outputs", call_data.get("output", {})),
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
        return self._generate_execution(real_llm, prompt_manager, test_data), False

    def _generate_execution(self, real_llm, prompt_manager, test_data):
        """Generate execution by running the full workflow"""
        
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        constraints_data = example["agent"]["constraints"]
        
        print(f"User prompt: {user_prompt}")
        
        # Start captured execution
        HarmonyIntegration.start_captured_execution(user_prompt, constraints_data)
        
        # Create captured components
        qafiya_selector_orig = QafiyaSelector(real_llm, prompt_manager)
        qafiya_selector = capture_component(qafiya_selector_orig, "QafiyaSelector")
        
        prosody_refiner = ProsodyRefiner(real_llm, prompt_manager)
        qafiya_refiner = QafiyaRefiner(real_llm, prompt_manager)
        refiner_chain_orig = RefinerChain(
            refiners=[prosody_refiner, qafiya_refiner],
            llm=real_llm,
            max_iterations=1
        )
        refiner_chain = capture_component(refiner_chain_orig, "RefinerChain")
        
        poem_evaluator_orig = PoemEvaluator(real_llm)
        poem_evaluator = capture_component(poem_evaluator_orig, "PoemEvaluator")
        
        # Run the workflow
        constraints = Constraints(
            meter=constraints_data["meter"],
            qafiya=constraints_data["qafiya"],
            theme=constraints_data["theme"],
            tone=constraints_data["tone"]
        )
        
        print("Step 1: Enriching constraints...")
        enriched_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
        
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
        import asyncio
        refined_poem, _ = asyncio.run(refiner_chain.refine(
            initial_poem,
            enriched_constraints,
            target_quality=1.0
        ))
        
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
        
        return execution

    def _create_captured_components(self, real_llm, prompt_manager):
        """
        Create all agent components and wrap them with the capture middleware.
        The `capture_component` factory returns a proxy object that intercepts
        and logs all method calls before forwarding them to the original object.
        """
        # Create original refiner instances
        prosody_refiner = ProsodyRefiner(real_llm, prompt_manager)
        qafiya_refiner = QafiyaRefiner(real_llm, prompt_manager)

        # Create the RefinerChain with the original refiners
        refiner_chain_orig = RefinerChain(
            refiners=[prosody_refiner, qafiya_refiner],
            llm=real_llm,
            max_iterations=1 # Limit to one iteration for the test
        )

        # Instantiate other components
        qafiya_selector_orig = QafiyaSelector(real_llm, prompt_manager)
        poem_evaluator_orig = PoemEvaluator(real_llm)

        # --- Wrap all components for capture ---
        # The second argument is a friendly name for the component in the log.
        qafiya_selector = capture_component(qafiya_selector_orig, "QafiyaSelector")
        poem_evaluator = capture_component(poem_evaluator_orig, "PoemEvaluator")
        refiner_chain = capture_component(refiner_chain_orig, "RefinerChain")

        return qafiya_selector, poem_evaluator, refiner_chain

    @pytest.mark.asyncio
    async def test_full_workflow_with_harmony_capture(self, real_llm, prompt_manager, test_data):
        """
        Test the full workflow: constraints -> qafiya -> generation -> evaluation -> refinement -> harmony reasoning.
        The `tmp_path` fixture provides a temporary directory for test outputs.
        """
        # 1. LOAD OR GENERATE EXECUTION
        execution, was_loaded = self._load_or_generate_execution(real_llm, prompt_manager, test_data)
        
        if was_loaded:
            print("Using execution loaded from fixture")
            # For fixture data, we need to set up the capture context
            HarmonyIntegration.start_captured_execution(
                execution.user_prompt, 
                execution.initial_constraints
            )
            # Restore the execution to the capture
            from poet.logging.harmony_capture import get_capture
            capture = get_capture()
            capture._execution = execution
        else:
            print("Using newly generated execution")
            # Execution was already captured during generation

        # 2. CREATE CAPTURED COMPONENTS (only needed if not loaded from fixture)
        if not was_loaded:
            qafiya_selector, poem_evaluator, refiner_chain = self._create_captured_components(real_llm, prompt_manager)
        else:
            # For fixture data, we don't need to recreate components since we're not running the workflow
            qafiya_selector = poem_evaluator = refiner_chain = None

        print(f"\n=== Running Workflow with Harmony Capture for: {execution.user_prompt} ===")

        # 3. RUN THE AGENT WORKFLOW (only if not loaded from fixture)
        if not was_loaded:
            # The logic remains the same, but calls to wrapped components are now being logged.
            constraints = Constraints(
                meter=execution.initial_constraints["meter"],
                qafiya=execution.initial_constraints["qafiya"],
                theme=execution.initial_constraints["theme"],
                tone=execution.initial_constraints["tone"]
            )

            # Step 3.1: Enrich constraints (Captured Call)
            enriched_constraints = qafiya_selector.select_qafiya(constraints, execution.user_prompt)

            # Step 3.2: Use a hardcoded initial poem for test predictability
            initial_poem = LLMPoem(
                verses=[
                    "تَبْكي السَّحابُ عَلَى فِراقِكَ يا شَفَقْ",
                    "وَالْقَلْبُ يَشْكُو مَا أَصابَهُ مِنْ حُرَقْ",
                    "فِي لَيْلَةٍ حَزِنَتْ عَلَيْكَ دُجَى السُّهَا",
                    "وَجْدِي عَلَيْكَ يَزِيدُ مَعْ نَفْحِ العَبَقْ"
                ],
                llm_provider='test', model_name='hardcoded', constraints=constraints.to_dict()
            )
            print(f"Initial Poem:\n{initial_poem}")

            # Step 3.3: Refine the poem (Captured Call)
            # This will internally call the evaluator, which is also captured.
            refined_poem, _ = await refiner_chain.refine(
                initial_poem,
                enriched_constraints,
                target_quality=1.0
            )
            print(f"Refined Poem:\n{refined_poem}")

            # Step 3.4: Run a final evaluation for the report (Captured Call)
            final_evaluation = poem_evaluator.evaluate_poem(
                refined_poem,
                enriched_constraints,
                [EvaluationType.PROSODY, EvaluationType.QAFIYA]
            )
        else:
            print("Using poem and evaluation from fixture data")
            # For fixture data, we can extract the final poem from the execution
            if execution.final_poem:
                refined_poem = LLMPoem.from_dict(execution.final_poem)
            else:
                # Fallback to creating a basic poem if not in fixture
                refined_poem = LLMPoem(
                    verses=["Test verse"],
                    llm_provider='test',
                    model_name='fixture',
                    constraints=execution.initial_constraints
                )

        # 4. COMPLETE AND GENERATE HARMONY REASONING
        # This finalizes the capture, saves the raw data, and generates the reasoning.
        output_dir = Path(__file__).parent / "temp" / "harmony_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        harmony_reasoning = HarmonyIntegration.complete_and_reason(
            llm=real_llm,
            final_poem=refined_poem,
            quality_assessment=execution.quality_assessment if execution.quality_assessment else {"overall": 0.8},
            output_dir=output_dir
        )

        # 5. ASSERT THE RESULTS
        print("\n=== Generated Harmony Reasoning ===")
        print(harmony_reasoning)

        assert harmony_reasoning is not None, "Harmony reasoning should not be empty."
        # Check for key Harmony format elements
        assert "<|channel|>analysis" in harmony_reasoning
        assert "<|call|>" in harmony_reasoning
        assert "<|start|>tool" in harmony_reasoning
        assert "Final Poem:" in harmony_reasoning

        # Verify that the output files were created
        execution = HarmonyIntegration.get_capture().get_execution()
        assert execution is not None, "Execution object should exist."
        raw_file = output_dir / f"{execution.execution_id}_raw.json"
        harmony_file = output_dir / f"{execution.execution_id}_harmony.txt"

        assert raw_file.exists(), "Raw JSON capture file should be created."
        assert harmony_file.exists(), "Harmony reasoning text file should be created."
        assert len(raw_file.read_text(encoding='utf-8')) > 100, "Raw JSON file should not be empty."
        assert harmony_file.read_text(encoding='utf-8') == harmony_reasoning, "Harmony file content should match returned string."