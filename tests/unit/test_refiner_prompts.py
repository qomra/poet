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
            qafiya_pattern="قَ",
            theme="غزل",
            tone="حزينة",
            line_count=4
        )
    
    def test_verse_completion_prompt(self, prompt_manager, sample_constraints):
        """Test verse completion prompt formatting"""
        existing_verses = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ\nبِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
        
        formatted_prompt = prompt_manager.format_prompt(
            'verse_completion',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya,
            qafiya_pattern=sample_constraints.qafiya_pattern,
            theme=sample_constraints.theme,
            tone=sample_constraints.tone,
            existing_verses=existing_verses,
            verses_to_add=2
        )
        
        # Check that all required parameters are included
        assert sample_constraints.meter in formatted_prompt
        assert sample_constraints.qafiya in formatted_prompt
        assert sample_constraints.qafiya_pattern in formatted_prompt
        assert sample_constraints.theme in formatted_prompt
        assert sample_constraints.tone in formatted_prompt
        assert existing_verses in formatted_prompt
        assert "2" in formatted_prompt  # verses_to_add
        
        # Check that it's a valid prompt
        assert "أنت شاعر عربي مبدع" in formatted_prompt
        assert "إضافة" in formatted_prompt
        assert "بيت جديد" in formatted_prompt
    
    def test_prosody_refinement_prompt(self, prompt_manager, sample_constraints):
        """Test prosody refinement prompt formatting"""
        existing_verse = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        context = "وزن خاطئ في البيت الأول"
        
        formatted_prompt = prompt_manager.format_prompt(
            'prosody_refinement',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya,
            qafiya_pattern=sample_constraints.qafiya_pattern,
            theme=sample_constraints.theme,
            tone=sample_constraints.tone,
            existing_verses=existing_verse,
            context=context,
            meeter_tafeelat="فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِيلُنْ"
        )
        
        # Check that all required parameters are included
        assert sample_constraints.meter in formatted_prompt
        assert sample_constraints.qafiya in formatted_prompt
        assert sample_constraints.qafiya_pattern in formatted_prompt
        assert sample_constraints.theme in formatted_prompt
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
        
        formatted_prompt = prompt_manager.format_prompt(
            'qafiya_refinement',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya,
            qafiya_pattern=sample_constraints.qafiya_pattern,
            theme=sample_constraints.theme,
            tone=sample_constraints.tone,
            existing_verses=existing_verse,
            context=context,
            meeter_tafeelat="فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِيلُنْ"
        )
        
        # Check that all required parameters are included
        assert sample_constraints.meter in formatted_prompt
        assert sample_constraints.qafiya in formatted_prompt
        assert sample_constraints.qafiya_pattern in formatted_prompt
        assert sample_constraints.theme in formatted_prompt
        assert sample_constraints.tone in formatted_prompt
        assert existing_verse in formatted_prompt
        assert context in formatted_prompt
        
        # Check that it's a valid prompt
        assert "خبير في القوافي" in formatted_prompt
        assert "إصلاح القافية" in formatted_prompt
        assert "القافية المطلوبة" in formatted_prompt
    
    def test_verse_completion_prompt_updated(self, prompt_manager, sample_constraints):
        """Test updated verse completion prompt with refinement support"""
        existing_verses = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        context = "إضافة أبيات جديدة"
        
        formatted_prompt = prompt_manager.format_prompt(
            'verse_completion',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya,
            qafiya_pattern=sample_constraints.qafiya_pattern,
            theme=sample_constraints.theme,
            tone=sample_constraints.tone,
            context=context,
            existing_verses=existing_verses,
            verses_to_add=2,
            fix_mode=False,
            qafiya_fix=False
        )
        
        # Check that all required parameters are included
        assert sample_constraints.meter in formatted_prompt
        assert sample_constraints.qafiya in formatted_prompt
        assert sample_constraints.qafiya_pattern in formatted_prompt
        assert sample_constraints.theme in formatted_prompt
        assert sample_constraints.tone in formatted_prompt
        assert existing_verses in formatted_prompt
        assert context in formatted_prompt
        assert "2" in formatted_prompt  # verses_to_add
        
        # Check that it's a valid prompt
        assert "أنت شاعر عربي مبدع" in formatted_prompt
        assert "إضافة" in formatted_prompt
        assert "بيت جديد" in formatted_prompt
    
    def test_prompt_template_parameters(self, prompt_manager):
        """Test that prompt templates have correct parameters"""
        # Test verse_completion template
        template_info = prompt_manager.get_template_info('verse_completion')
        expected_params = {
            'meter', 'qafiya', 'qafiya_pattern', 'theme', 'tone',
            'existing_verses', 'verses_to_add'
        }
        assert set(template_info['parameters']) == expected_params

        # Test prosody_refinement template
        template_info = prompt_manager.get_template_info('prosody_refinement')
        expected_params = {
            'meter', 'qafiya', 'qafiya_pattern', 'theme', 'tone',
            'existing_verses', 'context', 'meeter_tafeelat'
        }
        assert set(template_info['parameters']) == expected_params

        # Test qafiya_refinement template
        template_info = prompt_manager.get_template_info('qafiya_refinement')
        expected_params = {
            'meter', 'qafiya', 'qafiya_pattern', 'theme', 'tone',
            'existing_verses', 'context', 'meeter_tafeelat'
        }
        assert set(template_info['parameters']) == expected_params
    
    def test_prompt_template_metadata(self, prompt_manager):
        """Test that prompt templates have correct metadata"""
        # Test verse_completion template
        template_info = prompt_manager.get_template_info('verse_completion')
        assert template_info['category'] == 'generation'
        assert template_info['metadata']['purpose'] == 'verse_completion'
        assert template_info['metadata']['language'] == 'arabic'

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
    
    def test_prompt_template_validation(self, prompt_manager, sample_constraints):
        """Test that prompt templates validate parameters correctly"""
        # Test valid parameters
        assert prompt_manager.validate_template('verse_completion',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya,
            qafiya_pattern=sample_constraints.qafiya_pattern,
            theme=sample_constraints.theme,
            tone=sample_constraints.tone,
            existing_verses="test",
            verses_to_add=2
        )
        
        # Test invalid parameters (missing required)
        assert not prompt_manager.validate_template('verse_completion',
            meter=sample_constraints.meter,
            qafiya=sample_constraints.qafiya
            # Missing other required parameters
        )
    
    def test_prompt_template_listing(self, prompt_manager):
        """Test that all refiner templates are available"""
        templates = prompt_manager.list_templates()

        # Check that all refiner templates are present
        assert 'verse_completion' in templates
        assert 'prosody_refinement' in templates
        assert 'qafiya_refinement' in templates
        assert 'simple_poem_generation' in templates 