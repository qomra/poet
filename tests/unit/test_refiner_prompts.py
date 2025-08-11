# tests/unit/test_refiner_prompts.py

import pytest
from poet.prompts.prompt_manager import PromptManager
from poet.models.constraints import Constraints


class TestRefinerPrompts:
    """Test refiner prompt templates"""
    
    @pytest.fixture
    def prompt_manager(self):
        """Create prompt manager"""
        return PromptManager()
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints"""
        return Constraints(
            meter="بحر الطويل",
            qafiya="ق",
            qafiya_harakah="مكسور",
            qafiya_type="متواتر",
            theme="غزل",
            tone="حزينة",
            line_count=4
        )
    

    def test_prosody_refinement_prompt(self, prompt_manager, sample_constraints):
        """Test prosody refinement prompt formatting"""
        existing_verse = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        context = "وزن خاطئ في البيت الأول"
        
        formatted_prompt = prompt_manager.format_prompt(
            'prosody_refinement',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya,
            qafiya_harakah=sample_constraints.qafiya_harakah,
            qafiya_type=sample_constraints.qafiya_type,
            qafiya_type_description_and_examples=sample_constraints.qafiya_type_description_and_examples,
            theme=sample_constraints.theme, 
            tone=sample_constraints.tone,
            existing_verses=existing_verse,
            context=context,
            meeter_tafeelat="فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِيلُنْ"
        )
        
        # Check that all required parameters are included
        assert sample_constraints.meter in formatted_prompt
        assert sample_constraints.qafiya in formatted_prompt
        assert sample_constraints.theme in formatted_prompt
        assert sample_constraints.qafiya_harakah in formatted_prompt
        assert sample_constraints.tone in formatted_prompt
        assert existing_verse in formatted_prompt
        assert context in formatted_prompt
        
        # Check that it's a valid prompt
        assert "محلل عروضي" in formatted_prompt
        assert "إصلاح الوزن العروضي" in formatted_prompt
        assert "البحر المحدد" in formatted_prompt
    
    def test_qafiya_refinement_prompt(self, prompt_manager, sample_constraints):
        """Test qafiya refinement prompt formatting"""
        existing_verse = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        context = "إصلاح القافية لتكون: قَ"
        entire_poem = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ\nقِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        
        formatted_prompt = prompt_manager.format_prompt(
            'qafiya_refinement',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya,
            qafiya_type='متواتر',
            qafiya_type_description_and_examples='قافية متواترة: متحرك واحد بين ساكنين',
            qafiya_harakah=sample_constraints.qafiya_harakah,
            theme=sample_constraints.theme,
            tone=sample_constraints.tone,
            existing_verses=existing_verse,
            context=context,
            entire_poem=entire_poem,
            meeter_tafeelat="فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِيلُنْ"
        )
        
        # Check that all required parameters are included
        assert sample_constraints.meter in formatted_prompt
        assert sample_constraints.qafiya in formatted_prompt
        assert sample_constraints.qafiya_harakah in formatted_prompt
        assert sample_constraints.theme in formatted_prompt
        assert sample_constraints.tone in formatted_prompt
        assert existing_verse in formatted_prompt
        assert context in formatted_prompt
        assert entire_poem in formatted_prompt

        # Check that it's a valid prompt
        assert "خبيرًا دقيقًا في القوافي العربية" in formatted_prompt
        assert "إصلاح القافية" in formatted_prompt
        assert "القافية المطلوبة" in formatted_prompt
    

    def test_prompt_template_parameters(self, prompt_manager):
        """Test that prompt templates have correct parameters"""

        # Test prosody_refinement template
        template_info = prompt_manager.get_template_info('prosody_refinement')
        expected_params = {
            'meter', 'meeter_tafeelat', 'qafiya', 'qafiya_harakah', 'qafiya_type', 'qafiya_type_description_and_examples', 'theme', 'tone',
            'existing_verses', 'context'
        }
        assert set(template_info['parameters']) == expected_params

        # Test qafiya_refinement template
        template_info = prompt_manager.get_template_info('qafiya_refinement')

        expected_params = {
            'meter', 'meeter_tafeelat', 'qafiya', 'qafiya_harakah', 'qafiya_type', 'qafiya_type_description_and_examples',
            'theme', 'tone', 'entire_poem', 'existing_verses', 'context'
        }
        assert set(template_info['parameters']) == expected_params
    
    def test_prompt_template_metadata(self, prompt_manager):
        """Test that prompt templates have correct metadata"""


        # Test prosody_refinement template
        template_info = prompt_manager.get_template_info('prosody_refinement')
        assert template_info['category'] == 'refinement'
        assert template_info['metadata']['purpose'] == 'prosody_fix'
        assert template_info['metadata']['language'] == 'arabic'

        # Test qafiya_refinement template
        template_info = prompt_manager.get_template_info('qafiya_refinement')
        assert template_info['category'] == 'refinement'
        assert template_info['metadata']['purpose'] == 'qafiya_fix'
        assert template_info['metadata']['language'] == 'arabic'

    
    def test_prompt_template_listing(self, prompt_manager):
        """Test that all refiner templates are available"""
        templates = prompt_manager.list_templates()

        # Check that all refiner templates are present
        assert 'prosody_refinement' in templates
        assert 'qafiya_refinement' in templates
        assert 'simple_poem_generation' in templates 