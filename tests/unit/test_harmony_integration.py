"""
Unit tests for HarmonyIntegration and HarmonyCompiler using captured fixture data.
This allows testing without running the full refiner workflow.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import os

from poet.logging.integration import HarmonyIntegration
from poet.compiler.harmony import HarmonyCompiler
from poet.logging.harmony_capture import PipelineExecution, CapturedCall
from poet.llm.base_llm import BaseLLM


class TestHarmonyIntegration:
    """Test HarmonyIntegration functionality"""
    
    def test_start_captured_execution(self):
        """Test starting a new captured execution"""
        user_prompt = "اكتب قصيدة غزلية"
        constraints = {"meter": "بحر الكامل", "theme": "غزل"}
        
        # Start execution
        HarmonyIntegration.start_captured_execution(user_prompt, constraints)
        
        # Verify execution was started
        from poet.logging.harmony_capture import get_capture
        capture = get_capture()
        execution = capture.get_execution()
        
        assert execution is not None
        assert execution.user_prompt == user_prompt
        assert execution.initial_constraints == constraints
        assert len(execution.calls) == 0
    
    def test_instrument_component(self):
        """Test instrumenting a component with capture decorators"""
        # Create a mock component
        mock_component = Mock()
        mock_component.test_method = lambda x: x * 2
        
        # Instrument it
        HarmonyIntegration.instrument_component(mock_component, "TestComponent", "test")
        
        # Verify the method was wrapped
        assert hasattr(mock_component.test_method, '__wrapped__')
        assert mock_component.test_method(5) == 10


class TestHarmonyCompiler:
    """Test HarmonyCompiler functionality"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM"""
        llm = Mock(spec=BaseLLM)
        llm.generate.return_value = """<|start|>system<|message|>You are an Arabic Poetry Generation Agent.
Knowledge cutoff: 2024-06
Current date: 2025-01-10
Reasoning: high
# Valid channels: analysis, commentary, final.<|end|>

<|start|>assistant<|channel|>analysis<|message|>I'll analyze this poetry generation pipeline execution and reconstruct the reasoning process.<|end|>

<|start|>assistant<|channel|>final<|message|>Here is the final analysis of the poetry generation pipeline.<|end|>"""
        return llm
    
    @pytest.fixture
    def sample_execution(self):
        """Create a sample execution for testing"""
        execution = PipelineExecution(
            execution_id="test-123",
            user_prompt="اكتب قصيدة غزلية",
            initial_constraints={"meter": "بحر الكامل", "theme": "غزل"}
        )
        
        # Add some sample calls
        call1 = CapturedCall(
            component_name="QafiyaSelector",
            method_name="select_qafiya",
            call_type="enrich",
            inputs={"constraints": {"meter": "بحر الكامل"}},
            outputs={"qafiya": "ق", "qafiya_type": "متواتر"},
            duration_ms=150,
            success=True
        )
        
        call2 = CapturedCall(
            component_name="RefinerChain",
            method_name="refine",
            call_type="refine",
            inputs={"poem": "test_poem", "constraints": "test_constraints"},
            outputs={"refined_poem": "refined_test_poem"},
            duration_ms=300,
            success=True
        )
        
        execution.add_call(call1)
        execution.add_call(call2)
        
        return execution
    
    def test_serialize_output_with_to_dict(self, mock_llm):
        """Test serializing objects with to_dict method"""
        compiler = HarmonyCompiler(mock_llm)
        
        # Test object with to_dict
        class TestObject:
            def to_dict(self):
                return {"key": "value"}
        
        obj = TestObject()
        result = compiler._serialize_output(obj)
        assert result == {"key": "value"}
    
    def test_serialize_output_with_dict_attr(self, mock_llm):
        """Test serializing objects with __dict__ attribute"""
        compiler = HarmonyCompiler(mock_llm)
        
        # Test object with __dict__
        class TestObject:
            def __init__(self):
                self.public_attr = "value"
                self._private_attr = "private"
        
        obj = TestObject()
        result = compiler._serialize_output(obj)
        assert result == {"public_attr": "value"}
        assert "_private_attr" not in result
    
    def test_serialize_output_with_lists_and_dicts(self, mock_llm):
        """Test serializing nested lists and dictionaries"""
        compiler = HarmonyCompiler(mock_llm)
        
        # Test nested structure
        data = {
            "list": [{"key": "value"}, "string"],
            "nested": {"inner": [1, 2, 3]}
        }
        
        result = compiler._serialize_output(data)
        assert result == data  # Should remain unchanged for basic types
    
    def test_format_execution_steps(self, mock_llm, sample_execution):
        """Test formatting execution steps"""
        compiler = HarmonyCompiler(mock_llm)
        steps = compiler._format_execution_steps(sample_execution)
        
        assert "QafiyaSelector.select_qafiya" in steps
        assert "RefinerChain.refine" in steps
        assert "150ms" in steps
        assert "300ms" in steps
        assert "enrich" in steps
        assert "refine" in steps
    
    def test_prompt_manager_integration(self, mock_llm, sample_execution):
        """Test prompt manager integration"""
        compiler = HarmonyCompiler(mock_llm)
        
        # Test that the template is loaded
        template = compiler.prompt_manager.get_template("harmony_structured")
        assert template is not None
        assert template.name == "harmony_structured"
        assert template.category.value == "compiler"
        
        # Test that the template has required parameters
        required_params = ["user_prompt", "initial_constraints", "execution_steps", 
                          "final_poem", "quality_assessment", "conversation_start_date"]
        for param in required_params:
            assert param in template.parameters
    
    def test_generate_structured_harmony(self, mock_llm, sample_execution):
        """Test generating structured harmony data"""
        compiler = HarmonyCompiler(mock_llm)
        structured_data = compiler.generate_structured_harmony(sample_execution)
        
        # Verify LLM was called
        mock_llm.generate.assert_called_once()
        
        # Verify response contains expected content
        assert "system_message" in structured_data
        assert "developer_message" in structured_data
        assert "messages" in structured_data
    
    def test_save_harmony_reasoning(self, mock_llm, tmp_path):
        """Test saving harmony reasoning to file"""
        compiler = HarmonyCompiler(mock_llm)
        test_reasoning = "<|start|>system<|message|>Test content<|end|>"
        output_file = tmp_path / "test_harmony.txt"
        
        compiler.save_harmony_reasoning(test_reasoning, output_file)
        
        assert output_file.exists()
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == test_reasoning


class TestHarmonyCompilerWithFixture:
    """Test HarmonyCompiler using captured fixture data"""
    
    @pytest.fixture
    def fixture_data(self):
        """Load the captured fixture data"""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "harmony_test.json"
        
        if not fixture_path.exists():
            pytest.skip("Harmony fixture not found. Run capture_fixture.py first.")
        
        with open(fixture_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing"""
        llm = Mock(spec=BaseLLM)
        llm.generate.return_value = """
<|start|>system<|message|>You are an Arabic Poetry Generation Agent.
Knowledge cutoff: 2024-06
Current date: 2025-01-10
Reasoning: high
# Valid channels: analysis, commentary, final.<|end|>

<|start|>assistant<|message|>I'll analyze this poetry generation pipeline execution and reconstruct the reasoning process.<|end|>
"""
        return llm
    
    def test_compiler_with_real_fixture(self, mock_llm, fixture_data):
        """Test compiler with real captured fixture data"""
        # Reconstruct PipelineExecution from fixture
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
        
        # Test compiler
        compiler = HarmonyCompiler(mock_llm)
        
        # Test structured harmony generation
        structured_data = compiler.generate_structured_harmony(execution)
        assert "system_message" in structured_data
        assert "developer_message" in structured_data
        assert "messages" in structured_data
        
        # Verify LLM was called
        mock_llm.generate.assert_called_once()
    
    def test_fixture_data_structure(self, fixture_data):
        """Test that fixture data has expected structure"""
        required_fields = [
            "execution_id", "started_at", "user_prompt", 
            "initial_constraints", "calls"
        ]
        
        for field in required_fields:
            assert field in fixture_data, f"Missing required field: {field}"
        
        # Check calls structure
        assert isinstance(fixture_data["calls"], list)
        if fixture_data["calls"]:
            call = fixture_data["calls"][0]
            call_fields = [
                "call_id", "component_name", "method_name", 
                "call_type", "inputs", "outputs"
            ]
            for field in call_fields:
                assert field in call, f"Missing call field: {field}"
        
        print(f"Fixture contains {len(fixture_data['calls'])} captured calls")
        print(f"Execution ID: {fixture_data['execution_id']}")
        print(f"User prompt: {fixture_data['user_prompt']}")


class TestHarmonyCompilerWithRealLLM:
    """Test HarmonyCompiler with real LLM (when available)"""
    
    @pytest.fixture
    def fixture_data(self):
        """Load harmony fixture data for testing"""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "harmony_test.json"
        
        if not fixture_path.exists():
            pytest.skip("Harmony fixture not found. Run capture_fixture.py first.")
        
        with open(fixture_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.mark.skipif(
        not os.getenv("TEST_REAL_LLMS"),
        reason="Real LLM tests require TEST_REAL_LLMS environment variable"
    )
    def test_generate_structured_harmony_real_llm(self, real_llm, fixture_data):
        """Test structured harmony generation with real LLM"""
        
        # Reconstruct PipelineExecution from fixture
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
        
        # Test compiler with real LLM
        compiler = HarmonyCompiler(real_llm)
        
        # Generate structured harmony data
        print(f"\nGenerating structured harmony data with real LLM...")
        print(f"LLM Provider: {real_llm.__class__.__name__}")
        print(f"Execution ID: {execution.execution_id}")
        print(f"User prompt: {execution.user_prompt}")
        print(f"Number of calls: {len(execution.calls)}")
        
        structured_data = compiler.generate_structured_harmony(execution)
        
        # Print the generated structured data
        print(f"\nGenerated Structured Harmony Data:")
        print("=" * 80)
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        print("=" * 80)
        
        # Verify the structured data has expected structure
        assert "system_message" in structured_data
        assert "developer_message" in structured_data
        assert "messages" in structured_data
        
        # Verify it contains information about the execution
        assert execution.user_prompt in structured_data.get("messages", [{}])[0].get("content", "")
        
        print(f"\nStructured harmony generation successful!")
        print(f"Data keys: {list(structured_data.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 