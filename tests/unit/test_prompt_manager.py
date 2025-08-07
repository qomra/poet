import pytest
from poet.prompts.prompt_manager import PromptCategory


class TestPromptManager:
    
    def test_get_template_unified_extraction(self, prompt_manager):
        """Test getting unified_extraction template"""
        template = prompt_manager.get_template('unified_extraction')
        assert template.name == 'unified_extraction'
        assert template.category == PromptCategory.CONSTRAINT_INFERENCE
        assert 'user_prompt' in template.parameters
    
    def test_get_template_simple_poem_generation(self, prompt_manager):
        """Test getting simple_poem_generation template"""
        template = prompt_manager.get_template('simple_poem_generation')
        assert template.name == 'simple_poem_generation'
        assert template.category == PromptCategory.GENERATION
        assert 'line_count' in template.parameters
        assert 'meter' in template.parameters
    
    def test_get_template_imagery_creation(self, prompt_manager):
        """Test getting imagery_creation template"""
        template = prompt_manager.get_template('imagery_creation')
        assert template.name == 'imagery_creation'
        assert template.category == PromptCategory.GENERATION
        assert 'theme' in template.parameters
        assert 'tone' in template.parameters
    
    def test_get_template_qafiya_validation(self, prompt_manager):
        """Test getting qafiya_validation template"""
        template = prompt_manager.get_template('qafiya_validation')
        assert template.name == 'qafiya_validation'
        assert template.category == PromptCategory.EVALUATION
        assert 'verses' in template.parameters
        assert 'expected_qafiya' in template.parameters
    
    def test_get_template_tashkeel(self, prompt_manager):
        """Test getting tashkeel template"""
        template = prompt_manager.get_template('tashkeel')
        assert template.name == 'tashkeel'
        assert template.category == PromptCategory.EVALUATION
        assert 'text' in template.parameters
    
    def test_get_template_not_found(self, prompt_manager):
        """Test getting a non-existent template raises KeyError"""
        with pytest.raises(KeyError, match="Template 'nonexistent' not found"):
            prompt_manager.get_template('nonexistent')
    
    def test_get_templates_by_category_constraint_inference(self, prompt_manager):
        """Test getting constraint_inference templates"""
        templates = prompt_manager.get_templates_by_category(PromptCategory.CONSTRAINT_INFERENCE)
        assert len(templates) == 1
        template_names = [t.name for t in templates]
        assert 'unified_extraction' in template_names
    
    def test_get_templates_by_category_generation(self, prompt_manager):
        """Test getting generation templates"""
        templates = prompt_manager.get_templates_by_category(PromptCategory.GENERATION)
        assert len(templates) == 3
        template_names = [t.name for t in templates]
        assert 'simple_poem_generation' in template_names
        assert 'imagery_creation' in template_names
        assert 'verse_completion' in template_names
    
    def test_get_templates_by_category_evaluation(self, prompt_manager):
        """Test getting evaluation templates"""
        templates = prompt_manager.get_templates_by_category(PromptCategory.EVALUATION)
        assert len(templates) == 3  # qafiya_validation, tashkeel
        template_names = [t.name for t in templates]
        assert 'qafiya_validation' in template_names
        assert 'tashkeel' in template_names
    
    def test_list_templates(self, prompt_manager):
        """Test listing all available templates"""
        templates = prompt_manager.list_templates()
        assert len(templates) == 14  # Updated to match actual count
        expected_templates = [
            'unified_extraction',
            'simple_poem_generation', 'imagery_creation', 'verse_completion',
            'qafiya_validation', 'tashkeel',
            'bahr_selection', 'qafiya_completion'
        ]
        for template in expected_templates:
            assert template in templates
    
    def test_format_prompt_unified_extraction(self, prompt_manager):
        """Test formatting unified_extraction prompt"""
        formatted = prompt_manager.format_prompt('unified_extraction', user_prompt='أريد قصيدة غزل حزينة')
        assert 'خبير في الشعر العربي والعروض' in formatted
        assert 'استخراج جميع القيود والمتطلبات الشعرية' in formatted
        assert 'القيود العروضية' in formatted
        assert 'القيود الموضوعية' in formatted
        assert '```json' in formatted
    
    def test_format_prompt_simple_poem_generation(self, prompt_manager):
        """Test formatting simple_poem_generation prompt"""
        formatted = prompt_manager.format_prompt('simple_poem_generation',
                                                meter='بحر الكامل',
                                                qafiya='ق',
                                                line_count=4,
                                                verse_count=8)
        assert 'بحر الكامل' in formatted
        assert 'ق' in formatted
        assert '4' in formatted
    
    def test_format_prompt_imagery_creation(self, prompt_manager):
        """Test formatting imagery_creation prompt"""
        formatted = prompt_manager.format_prompt('imagery_creation',
                                                theme='الطبيعة',
                                                tone='هادئ',
                                                imagery_type='تشبيهات',
                                                cultural_context='عربي',
                                                requirements='صور حسية')
        assert 'الطبيعة' in formatted
        assert 'هادئ' in formatted
        assert 'تشبيهات' in formatted
    
    def test_format_prompt_qafiya_validation(self, prompt_manager):
        """Test formatting qafiya_validation prompt"""
        formatted = prompt_manager.format_prompt('qafiya_validation',
                                                verses='بيت شعري تجريبي',
                                                expected_qafiya='ق')
        assert 'بيت شعري تجريبي' in formatted
        assert 'ق' in formatted
    
    def test_format_prompt_tashkeel(self, prompt_manager):
        """Test formatting tashkeel prompt"""
        formatted = prompt_manager.format_prompt('tashkeel',
                                                text='نص شعري للتشكيل')
        assert 'نص شعري للتشكيل' in formatted
    
    def test_format_prompt_missing_parameters(self, prompt_manager):
        """Test formatting prompt with missing parameters raises ValueError"""
        with pytest.raises(ValueError, match="Missing required parameters"):
            prompt_manager.format_prompt('unified_extraction')  # Missing user_prompt
    
    def test_get_template_info_unified_extraction(self, prompt_manager):
        """Test getting unified_extraction template info"""
        info = prompt_manager.get_template_info('unified_extraction')
        assert info['name'] == 'unified_extraction'
        assert info['category'] == 'constraint_inference'
        assert 'user_prompt' in info['parameters']
        assert 'version' in info['metadata']
    
    def test_get_template_info_simple_poem_generation(self, prompt_manager):
        """Test getting simple_poem_generation template info"""
        info = prompt_manager.get_template_info('simple_poem_generation')
        assert info['name'] == 'simple_poem_generation'
        assert info['category'] == 'generation'
        assert 'line_count' in info['parameters']
        assert 'meter' in info['parameters']
    
    def test_get_template_info_not_found(self, prompt_manager):
        """Test getting template info for non-existent template raises KeyError"""
        with pytest.raises(KeyError, match="Template 'nonexistent' not found"):
            prompt_manager.get_template_info('nonexistent')
    
    def test_template_format_method(self, prompt_manager):
        """Test template format method directly"""
        template = prompt_manager.get_template('unified_extraction')
        formatted = template.format(user_prompt='أريد قصيدة غزل حزينة')
        assert 'أريد قصيدة غزل حزينة' in formatted
        assert 'خبير في الشعر العربي والعروض' in formatted
    
    def test_template_format_missing_parameters(self, prompt_manager):
        """Test template format method with missing parameters raises ValueError"""
        template = prompt_manager.get_template('unified_extraction')
        with pytest.raises(ValueError, match="Missing required parameters"):
            template.format()  # Missing user_prompt 