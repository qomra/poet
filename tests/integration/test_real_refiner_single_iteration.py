# tests/integration/test_real_refiner_single_iteration.py

import os
import pytest
import json
from poet.evaluation.poem import PoemEvaluator, EvaluationType
from poet.generation.poem_generator import SimplePoemGenerator
from poet.refinement import RefinerChain, LineCountRefiner, ProsodyRefiner, QafiyaRefiner, TashkeelRefiner
from poet.analysis.constraint_parser import ConstraintParser
from poet.analysis.qafiya_selector import QafiyaSelector
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.prompts.prompt_manager import PromptManager
from poet.llm.llm_factory import get_real_llm_from_env


@pytest.mark.integration
@pytest.mark.real_data
class TestRealRefinerSingleIteration:
    """Integration tests for single iteration of generation, evaluation, and refinement using real LLMs only"""
    
    @pytest.fixture(scope="class")
    def prompt_manager(self):
        """Create a PromptManager instance"""
        return PromptManager()
    
    @pytest.fixture(scope="class")
    def real_llm(self):
        """Get real LLM - only runs when TEST_REAL_LLMS is set"""
        if os.environ.get("TEST_REAL_LLMS") != "true":
            pytest.skip("Skipping real LLM tests as TEST_REAL_LLMS is not set to 'true'")
        return get_real_llm_from_env()
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Load test data"""
        import json
        from pathlib import Path
        test_file = Path(__file__).parent.parent / "fixtures" / "test_data.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_components(self, real_llm, prompt_manager):
        """Create all components with real LLM"""
        # Create all components with the same LLM
        constraint_parser = ConstraintParser(real_llm, prompt_manager)
        qafiya_selector = QafiyaSelector(real_llm, prompt_manager)
        poem_generator = SimplePoemGenerator(real_llm, prompt_manager)
        poem_evaluator = PoemEvaluator(real_llm)
        
        # Create refiners
        refiners = [
            #TashkeelRefiner(real_llm, prompt_manager),
            #LineCountRefiner(real_llm, prompt_manager),
            #ProsodyRefiner(real_llm, prompt_manager),
            QafiyaRefiner(real_llm, prompt_manager),
            
        ]
        refiner_chain = RefinerChain(refiners, real_llm, max_iterations=5)
        
        return constraint_parser, qafiya_selector, poem_generator, poem_evaluator, refiner_chain
    
    def run_evaluation_and_print_results(self, poem: LLMPoem, poem_evaluator: PoemEvaluator, constraints: Constraints):
        """Run evaluation and print results"""
        evaluation = poem_evaluator.evaluate_poem(
            poem,
            constraints,
            [EvaluationType.LINE_COUNT, EvaluationType.PROSODY, EvaluationType.QAFIYA, EvaluationType.TASHKEEL]
        )
        print(f"Evaluation results:")
        print(f"  Overall quality score: {evaluation.quality.overall_score}")
        print(f"  Is acceptable: {evaluation.quality.is_acceptable}")
        print(f"  Prosody validation: {evaluation.quality.prosody_validation.overall_valid}")
        print(f"  Qafiya validation: {evaluation.quality.qafiya_validation.overall_valid}")
        print(f"  Tashkeel validation: {evaluation.quality.tashkeel_validation.overall_valid}")
        print(f"  Line count validation: {evaluation.quality.line_count_validation.is_valid}")

        if evaluation.quality.prosody_issues:
            print(f"  Prosody issues: {evaluation.quality.prosody_issues}")
        if evaluation.quality.qafiya_issues:
            print(f"  Qafiya issues: {evaluation.quality.qafiya_issues}")
        if evaluation.quality.tashkeel_issues:
            print(f"  Tashkeel issues: {evaluation.quality.tashkeel_issues}")
        if evaluation.quality.line_count_issues:
            print(f"  Line count issues: {evaluation.quality.line_count_issues}")

    @pytest.mark.asyncio
    async def test_single_iteration_example_1(self, real_llm, prompt_manager, test_data):
        """Test single iteration workflow for example 1 (غزل poem) - generate, evaluate, and refine"""
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Create components
        constraint_parser, qafiya_selector, poem_generator, poem_evaluator, refiner_chain = self._create_components(real_llm, prompt_manager)
        
        print(f"\n=== Example 1: Single Iteration Workflow ===")
        print(f"User Prompt: {user_prompt}")
        
        # Step 1: Parse constraints (if needed)
        print(f"\nStep 1: Parsing constraints...")
        constraints = Constraints(
            meter=expected_constraints["meter"],
            qafiya=expected_constraints["qafiya"],
            line_count=expected_constraints["line_count"],
            theme=expected_constraints["theme"],
            tone=expected_constraints["tone"]
        )
        print(f"Constraints: {constraints}")
        
        # Step 2: Select qafiya (complete missing components)
        print(f"\nStep 2: Selecting qafiya...")
        enriched_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
        print(f"Enriched constraints: {enriched_constraints}")
        
        # Step 3: Generate initial poem
        print(f"\nStep 3: Generating initial poem...")
        #initial_poem = poem_generator.generate_poem(enriched_constraints)
        """
        hard coded the initial poem
        1. تَبْكي السَّحابُ عَلَى فِراقِكَ يا شَفَقْ
        2. وَالْقَلْبُ يَشْكُو مَا أَصابَهُ مِنْ حُرَقْ
        3. فِي لَيْلَةٍ حَزِنَتْ عَلَيْكَ دُجَى السُّهَا
        4. وَجْدِي عَلَيْكَ يَزِيدُ مَعْ نَفْحِ العَبَقْ
        """
        initial_poem = LLMPoem(
            verses=[
                "تَبْكي السَّحابُ عَلَى فِراقِكَ يا شَفَقْ",
                "وَالْقَلْبُ يَشْكُو مَا أَصابَهُ مِنْ حُرَقْ",
                "فِي لَيْلَةٍ حَزِنَتْ عَلَيْكَ دُجَى السُّهَا",
                "وَجْدِي عَلَيْكَ يَزِيدُ مَعْ نَفْحِ العَبَقْ"
            ],
            llm_provider=real_llm.__class__.__name__,
            model_name=getattr(real_llm.config, 'model_name', 'unknown'),
            constraints=constraints.to_dict()
        )
        # Verify initial poem generation
        assert isinstance(initial_poem, LLMPoem)
        assert len(initial_poem.verses) > 0
        assert len(initial_poem.verses) % 2 == 0  # Should be even number (complete baits)
        assert all(isinstance(verse, str) for verse in initial_poem.verses)
        assert all(len(verse.strip()) > 0 for verse in initial_poem.verses)
        
        print(f"Initial poem generated:")
        print(f"  LLM Provider: {initial_poem.llm_provider}")
        print(f"  Model: {initial_poem.model_name}")
        print(f"  Verses: {len(initial_poem.verses)}")
        for i, verse in enumerate(initial_poem.verses, 1):
            print(f"    {i}. {verse}")
        
        
        # run evaluation
        print(f"\nStep 4: Running evaluation...")
        self.run_evaluation_and_print_results(initial_poem, poem_evaluator, enriched_constraints)
        
        # Step 4: Run single iteration of refinement
        print(f"\nStep 4: Running single iteration of refinement...")
        refined_poem, refinement_history = await refiner_chain.refine(
            initial_poem,
            enriched_constraints,
            target_quality=0.9
        )

        # step 5: run last evaluation
        print(f"\nStep 5: Running last evaluation...")
        self.run_evaluation_and_print_results(refined_poem, poem_evaluator, enriched_constraints)
        
        # Verify refinement results
        assert refined_poem is not None
        assert isinstance(refined_poem, LLMPoem)
        assert len(refined_poem.verses) > 0
        assert len(refined_poem.verses) % 2 == 0
        
        print(f"Refinement completed:")
        print(f"  Refinement steps: {len(refinement_history)}")
        print(f"  Refined poem verses: {len(refined_poem.verses)}")
        
        # Print the refined poem
        print(f"Refined poem:")
        for i, verse in enumerate(refined_poem.verses, 1):
            print(f"  {i}. {verse}")
        
        # Print refinement history
        print(f"Refinement history:")
        for i, step in enumerate(refinement_history, 1):
            print(f"  Step {i}: {step.refiner_name} (iteration {step.iteration})")
            print(f"    Quality before: {step.quality_before:.3f}")
            if step.quality_after is not None:
                print(f"    Quality after: {step.quality_after:.3f}")
                print(f"    Improvement: {step.quality_after - step.quality_before:.3f}")
            else:
                print(f"    Quality after: Not calculated yet")
                print(f"    Improvement: Not calculated yet")
        
        
        # Step 7: Generate refinement summary
        print(f"\nStep 7: Generating refinement summary...")
        summary = refiner_chain.get_refinement_summary(refinement_history)
        
        print(f"Refinement summary:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Refiners used: {summary['refiners_used']}")
        print(f"  Quality improvement: {summary['quality_improvement']:.3f}")
        print(f"  Iterations: {summary['iterations']}")
        
    