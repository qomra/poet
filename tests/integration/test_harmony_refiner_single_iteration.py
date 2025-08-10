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
    1. Start a capture session.
    2. Wrap all agent components (parser, generator, evaluator, refiners) to automatically capture their calls.
    3. Run the end-to-end poetry generation process.
    4. Complete the capture, which generates a raw execution log (JSON) and a final,
       LLM-generated Harmony reasoning trace.
    5. Assert that the outputs are generated correctly.
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
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        constraints_data = example["agent"]["constraints"]

        # 1. START THE CAPTURED EXECUTION
        # This initializes a new `PipelineExecution` object to store all subsequent calls.
        HarmonyIntegration.start_captured_execution(user_prompt, constraints_data)

        # 2. CREATE CAPTURED COMPONENTS
        # All components returned by this method are now being monitored.
        qafiya_selector, poem_evaluator, refiner_chain = self._create_captured_components(real_llm, prompt_manager)

        print(f"\n=== Running Workflow with Harmony Capture for: {user_prompt} ===")

        # 3. RUN THE AGENT WORKFLOW
        # The logic remains the same, but calls to wrapped components are now being logged.
        constraints = Constraints(
            meter=constraints_data["meter"],
            qafiya=constraints_data["qafiya"],
            theme=constraints_data["theme"],
            tone=constraints_data["tone"]
        )

        # Step 3.1: Enrich constraints (Captured Call)
        enriched_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)

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

        # 4. COMPLETE AND GENERATE HARMONY REASONING
        # This finalizes the capture, saves the raw data, and generates the reasoning.
        #output_dir = tmp_path / "harmony_run"
        output_dir = Path(__file__).parent / "temp" / "harmony_run"
        harmony_reasoning = HarmonyIntegration.complete_and_reason(
            llm=real_llm,
            final_poem=refined_poem,
            quality_assessment=final_evaluation.quality,
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