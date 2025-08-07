import pytest
import os
import json
from unittest.mock import Mock, patch
from poet.generation.poem_generator import SimplePoemGenerator, BasePoemGenerator, GenerationError
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.llm.base_llm import MockLLM, LLMConfig


class TestBasePoemGenerator:
    """Test the abstract base class functionality"""
    
    def test_base_class_initialization(self, mock_llm, prompt_manager):
        """Test BasePoemGenerator initialization"""
        # Create a concrete implementation for testing
        class TestGenerator(BasePoemGenerator):
            def generate_poem(self, constraints: Constraints) -> LLMPoem:
                return LLMPoem(verses=["test"], llm_provider="test", model_name="test")
            
            def can_handle_constraints(self, constraints: Constraints) -> bool:
                return True
        
        generator = TestGenerator(mock_llm, prompt_manager)
        
        assert generator.llm == mock_llm
        assert generator.prompt_manager == prompt_manager
        assert generator.logger is not None
    
    def test_base_class_abstract_methods(self):
        """Test that BasePoemGenerator cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BasePoemGenerator(Mock(), Mock())


class TestSimplePoemGenerator:
    """Test SimplePoemGenerator functionality"""
    
    @pytest.fixture
    def simple_generator(self, mock_llm, prompt_manager):
        """Create SimplePoemGenerator instance"""
        return SimplePoemGenerator(mock_llm, prompt_manager)
    
    @pytest.fixture
    def basic_constraints(self):
        """Basic Constraints for testing"""
        return Constraints(
            meter="طويل",
            qafiya="ل",
            line_count=4
        )
    
    @pytest.fixture
    def complex_constraints(self):
        """Complex Constraints with all fields"""
        return Constraints(
            meter="كامل",
            qafiya="م",
            line_count=6,
            theme="قصيدة غزل",
            tone="رومانسية",
            imagery=["الورد", "الليل"],
            keywords=["حب", "فراق"],
            register="فصيح",
            era="كلاسيكي"
        )
    
    def test_initialization(self, mock_llm, prompt_manager):
        """Test SimplePoemGenerator initialization"""
        generator = SimplePoemGenerator(mock_llm, prompt_manager)
        
        assert generator.llm == mock_llm
        assert generator.prompt_manager == prompt_manager
        assert isinstance(generator, BasePoemGenerator)
    
    def test_can_handle_constraints(self, simple_generator, basic_constraints, complex_constraints):
        """Test constraint handling capability"""
        # SimplePoemGenerator should handle all constraints
        assert simple_generator.can_handle_constraints(basic_constraints) is True
        assert simple_generator.can_handle_constraints(complex_constraints) is True
    
    def test_generate_poem_success(self, simple_generator, basic_constraints, mock_llm):
        """Test successful poem generation with mock LLM"""
        # Mock LLM response
        mock_response = '''
        ```json
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
            ]
        }
        ```
        '''
        # Use patch to mock the generate method
        with patch.object(mock_llm, 'generate', return_value=mock_response):
            # Generate poem
            poem = simple_generator.generate_poem(basic_constraints)
            
            # Verify LLM was called
            mock_llm.generate.assert_called_once()
            
            # Verify poem structure
            assert isinstance(poem, LLMPoem)
            assert len(poem.verses) == 4
            assert poem.verses[0] == "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
            assert poem.verses[1] == "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
            assert poem.verses[2] == "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا"
            assert poem.verses[3] == "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
            assert poem.llm_provider == "MockLLM"
            assert poem.model_name == "test-model"
            assert poem.constraints == basic_constraints.to_dict()
    
    def test_generate_poem_with_complex_constraints(self, simple_generator, complex_constraints, mock_llm):
        """Test poem generation with complex constraints"""
        # Mock LLM response
        mock_response = '''
        ```json
        {
            "verses": [
                "يَا لَيْلَةَ الحُبِّ وَالأَحْلامِ",
                "فِي ظِلِّكَ العَذْبِ وَالإِسْلامِ",
                "أَشْعَارُ قَلْبِي تَتَرَنَّمُ",
                "بِحُبِّكِ يَا أَجْمَلَ الأَيَّامِ",
                "فِي كُلِّ نَبْضٍ مِنْ دَمِي",
                "ذِكْرَاكِ تَعْزِفُ فِي أَوْتَارِي"
            ]
        }
        ```
        '''
        with patch.object(mock_llm, 'generate', return_value=mock_response):
            # Generate poem
            poem = simple_generator.generate_poem(complex_constraints)
            
            # Verify poem structure
            assert isinstance(poem, LLMPoem)
            assert len(poem.verses) == 6
            assert poem.constraints == complex_constraints.to_dict()
    
    def test_generate_poem_llm_error(self, simple_generator, basic_constraints, mock_llm):
        """Test poem generation when LLM fails"""
        with patch.object(mock_llm, 'generate', side_effect=Exception("LLM error")):
            with pytest.raises(GenerationError) as exc_info:
                simple_generator.generate_poem(basic_constraints)
            
            assert "Poem generation failed" in str(exc_info.value)
    
    def test_parse_llm_response_valid_json(self, simple_generator):
        """Test parsing valid JSON response"""
        response = '''
        ```json
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
            ]
        }
        ```
        '''
        
        verses = simple_generator._parse_llm_response(response)
        
        assert len(verses) == 2
        assert verses[0] == "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        assert verses[1] == "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
    
    def test_parse_llm_response_plain_json(self, simple_generator):
        """Test parsing plain JSON response without markdown"""
        response = '''
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
            ]
        }
        '''
        
        verses = simple_generator._parse_llm_response(response)
        
        assert len(verses) == 2
        assert verses[0] == "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        assert verses[1] == "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
    
    def test_parse_llm_response_no_json(self, simple_generator):
        """Test parsing response with no JSON"""
        response = "This is not JSON"
        
        with pytest.raises(GenerationError) as exc_info:
            simple_generator._parse_llm_response(response)
        
        assert "No JSON found in response" in str(exc_info.value)
    
    def test_parse_llm_response_invalid_json(self, simple_generator):
        """Test parsing invalid JSON response"""
        response = '''
        ```json
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
            ]
        ```
        '''  # Missing closing brace
        
        with pytest.raises(GenerationError) as exc_info:
            simple_generator._parse_llm_response(response)
        
        assert "Response parsing failed" in str(exc_info.value)
    
    def test_parse_llm_response_no_verses_key(self, simple_generator):
        """Test parsing JSON without verses key"""
        response = '''
        ```json
        {
            "poem": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
            ]
        }
        ```
        '''
        
        with pytest.raises(GenerationError) as exc_info:
            simple_generator._parse_llm_response(response)
        
        assert "No 'verses' key found in JSON response" in str(exc_info.value)
    
    def test_parse_llm_response_verses_not_list(self, simple_generator):
        """Test parsing JSON where verses is not a list"""
        response = '''
        ```json
        {
            "verses": "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        }
        ```
        '''
        
        with pytest.raises(GenerationError) as exc_info:
            simple_generator._parse_llm_response(response)
        
        assert "'verses' must be a list" in str(exc_info.value)
    
    def test_parse_llm_response_empty_verses(self, simple_generator):
        """Test parsing JSON with empty verses list"""
        response = '''
        ```json
        {
            "verses": []
        }
        ```
        '''
        
        with pytest.raises(GenerationError) as exc_info:
            simple_generator._parse_llm_response(response)
        
        assert "Verses list is empty" in str(exc_info.value)
    
    def test_parse_llm_response_empty_verses_after_processing(self, simple_generator):
        """Test parsing JSON with verses that become empty after processing"""
        response = '''
        ```json
        {
            "verses": ["", "   ", null, ""]
        }
        ```
        '''
        
        with pytest.raises(GenerationError) as exc_info:
            simple_generator._parse_llm_response(response)
        
        assert "No valid verses found after processing" in str(exc_info.value)
    
    def test_parse_llm_response_mixed_content(self, simple_generator):
        """Test parsing JSON with mixed content in verses"""
        response = '''
        ```json
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                123,
                "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ",
                "",
                "   "
            ]
        }
        ```
        '''
        
        verses = simple_generator._parse_llm_response(response)
        
        assert len(verses) == 3  # Only valid string verses
        assert verses[0] == "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        assert verses[1] == "123"  # Converted to string
        assert verses[2] == "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"


class TestSimplePoemGeneratorWithRealLLM:
    """Test SimplePoemGenerator with real LLM (when available)"""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_REAL_LLMS"),
        reason="Real LLM tests require TEST_REAL_LLMS environment variable"
    )
    def test_generate_poem_real_llm(self, real_llm, prompt_manager):
        """Test poem generation with real LLM"""
        generator = SimplePoemGenerator(real_llm, prompt_manager)
        
        constraints = Constraints(
            meter="طويل",
            qafiya="ل",
            line_count=4
        )
        
        # Generate poem
        poem = generator.generate_poem(constraints)
        
        # Verify poem structure
        assert isinstance(poem, LLMPoem)
        assert len(poem.verses) == 8
        assert all(isinstance(verse, str) for verse in poem.verses)
        assert all(len(verse.strip()) > 0 for verse in poem.verses)
        assert poem.llm_provider == real_llm.__class__.__name__
        assert poem.constraints == constraints.to_dict()
        
        # Print results for analysis
        print(f"\nGenerated poem with real LLM:")
        print(f"LLM Provider: {poem.llm_provider}")
        print(f"Model: {poem.model_name}")
        print(f"Constraints: {constraints}")
        print(f"Verses:")
        for i, verse in enumerate(poem.verses, 1):
            print(f"  {i}. {verse}")
    
    @pytest.mark.skipif(
        not os.getenv("TEST_REAL_LLMS"),
        reason="Real LLM tests require TEST_REAL_LLMS environment variable"
    )
    def test_generate_poem_complex_constraints_real_llm(self, real_llm, prompt_manager):
        """Test poem generation with complex constraints using real LLM"""
        generator = SimplePoemGenerator(real_llm, prompt_manager)
        
        constraints = Constraints(
            meter="كامل",
            qafiya="م",
            line_count=6,
            theme="قصيدة غزل",
            tone="رومانسية",
            imagery=["الورد", "الليل"],
            keywords=["حب", "فراق"]
        )
        
        # Generate poem
        poem = generator.generate_poem(constraints)
        
        # Verify poem structure
        assert isinstance(poem, LLMPoem)
        assert len(poem.verses) == 12
        assert all(isinstance(verse, str) for verse in poem.verses)
        assert all(len(verse.strip()) > 0 for verse in poem.verses)
        assert poem.constraints == constraints.to_dict()
        
        # Print results for analysis
        print(f"\nGenerated poem with complex constraints:")
        print(f"Constraints: {constraints}")
        print(f"Verses:")
        for i, verse in enumerate(poem.verses, 1):
            print(f"  {i}. {verse}")


class TestSimplePoemGeneratorEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def simple_generator(self, mock_llm, prompt_manager):
        return SimplePoemGenerator(mock_llm, prompt_manager)
    
    def test_generate_poem_none_constraints(self, simple_generator):
        """Test poem generation with None constraints"""
        with pytest.raises(GenerationError) as exc_info:
            simple_generator.generate_poem(None)
        
        assert "Poem generation failed" in str(exc_info.value)
        assert "NoneType" in str(exc_info.value)
    
    def test_generate_poem_empty_constraints(self, simple_generator):
        """Test poem generation with empty constraints"""
        constraints = Constraints()
        
        # Mock LLM response
        mock_response = '''
        ```json
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
            ]
        }
        ```
        '''
        with patch.object(simple_generator.llm, 'generate', return_value=mock_response):
            # Should work with empty constraints (uses defaults)
            poem = simple_generator.generate_poem(constraints)
            
            assert isinstance(poem, LLMPoem)
            assert len(poem.verses) == 2
    
    def test_generate_poem_large_line_count(self, simple_generator, mock_llm):
        """Test poem generation with large line count"""
        constraints = Constraints(
            meter="طويل",
            qafiya="ل",
            line_count=20
        )
        
        # Mock LLM response with many verses
        verses = [f"verse {i}" for i in range(1, 21)]
        mock_response = f'''
        ```json
        {{
            "verses": {json.dumps(verses)}
        }}
        ```
        '''
        with patch.object(mock_llm, 'generate', return_value=mock_response):
            poem = simple_generator.generate_poem(constraints)
            
            assert len(poem.verses) == 20
            assert poem.verses[0] == "verse 1"
            assert poem.verses[19] == "verse 20"
    
    def test_generate_poem_single_verse(self, simple_generator, mock_llm):
        """Test poem generation with single verse"""
        constraints = Constraints(
            meter="طويل",
            qafiya="ل",
            line_count=1
        )
        
        mock_response = '''
        ```json
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
            ]
        }
        ```
        '''
        with patch.object(mock_llm, 'generate', return_value=mock_response):
            poem = simple_generator.generate_poem(constraints)
            
            assert len(poem.verses) == 1
            assert poem.verses[0] == "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"


class TestSimplePoemGeneratorIntegration:
    """Integration tests for SimplePoemGenerator"""
    
    def test_full_generation_workflow(self, mock_llm, prompt_manager):
        """Test complete generation workflow"""
        generator = SimplePoemGenerator(mock_llm, prompt_manager)
        
        constraints = Constraints(
            meter="طويل",
            qafiya="ل",
            line_count=4,
            theme="قصيدة غزل",
            tone="رومانسية"
        )
        
        # Mock LLM response
        mock_response = '''
        ```json
        {
            "verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
            ]
        }
        ```
        '''
        with patch.object(mock_llm, 'generate', return_value=mock_response) as mock_generate:
            # Generate poem
            poem = generator.generate_poem(constraints)
            
            # Verify complete workflow
            assert isinstance(poem, LLMPoem)
            assert len(poem.verses) == 4
            assert poem.constraints == constraints.to_dict()
            assert poem.llm_provider == "MockLLM"
            assert poem.model_name == "test-model"
            
            # Verify prompt was formatted correctly
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert "البحر: طويل" in call_args
            assert "القافية: ل" in call_args
            assert "عدد الأبيات: 4" in call_args
    
    def test_generation_with_prompt_manager_injection(self, mock_llm):
        """Test generation with custom prompt manager"""
        # Create custom prompt manager
        custom_prompt_manager = Mock()
        custom_prompt_manager.format_prompt.return_value = "custom prompt"
        
        generator = SimplePoemGenerator(mock_llm, custom_prompt_manager)
        
        constraints = Constraints(meter="طويل", qafiya="ل", line_count=4)
        
        # Mock LLM response
        with patch.object(mock_llm, 'generate', return_value='''
        ```json
        {
            "verses": ["test verse"]
        }
        ```
        '''):
            poem = generator.generate_poem(constraints)
            
            # Verify custom prompt manager was used
            custom_prompt_manager.format_prompt.assert_called_once_with(
                'simple_poem_generation',
                meter="طويل",
                qafiya="ل",
                line_count=4,
                verse_count=8
            )
            mock_llm.generate.assert_called_once_with("custom prompt")
