import pytest
from poet.prompts.prompt_manager import PromptCategory


class TestPromptManager:
    
    def test_get_template_unified_extraction(self, prompt_manager):
        """Test getting unified_extraction template"""
        template = prompt_manager.get_template('unified_extraction')
        assert template.name == 'unified_extraction'
        assert template.category == PromptCategory.CONSTRAINT_INFERENCE
        assert 'user_prompt' in template.parameters
    
    def test_get_template_verse_generation(self, prompt_manager):
        """Test getting verse_generation template"""
        template = prompt_manager.get_template('verse_generation')
        assert template.name == 'verse_generation'
        assert template.category == PromptCategory.GENERATION
        assert 'theme' in template.parameters
        assert 'meter' in template.parameters
    
    def test_get_template_imagery_creation(self, prompt_manager):
        """Test getting imagery_creation template"""
        template = prompt_manager.get_template('imagery_creation')
        assert template.name == 'imagery_creation'
        assert template.category == PromptCategory.GENERATION
        assert 'theme' in template.parameters
        assert 'tone' in template.parameters
    
    def test_get_template_prosody_check(self, prompt_manager):
        """Test getting prosody_check template"""
        template = prompt_manager.get_template('prosody_check')
        assert template.name == 'prosody_check'
        assert template.category == PromptCategory.EVALUATION
        assert 'poem_text' in template.parameters
        assert 'expected_meter' in template.parameters
    
    def test_get_template_semantic_evaluation(self, prompt_manager):
        """Test getting semantic_evaluation template"""
        template = prompt_manager.get_template('semantic_evaluation')
        assert template.name == 'semantic_evaluation'
        assert template.category == PromptCategory.EVALUATION
        assert 'poem_text' in template.parameters
        assert 'intended_theme' in template.parameters
    
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
        assert 'verse_generation' in template_names
        assert 'imagery_creation' in template_names
    
    def test_get_templates_by_category_evaluation(self, prompt_manager):
        """Test getting evaluation templates"""
        templates = prompt_manager.get_templates_by_category(PromptCategory.EVALUATION)
        assert len(templates) == 4  # prosody_check, semantic_evaluation, qafiya_validation, tashkeel
        template_names = [t.name for t in templates]
        assert 'prosody_check' in template_names
        assert 'semantic_evaluation' in template_names
        assert 'qafiya_validation' in template_names
        assert 'tashkeel' in template_names
    
    def test_list_templates(self, prompt_manager):
        """Test listing all available templates"""
        templates = prompt_manager.list_templates()
        assert len(templates) == 12  # Updated to include bahr_selection
        expected_templates = [
            'unified_extraction',
            'verse_generation', 'imagery_creation',
            'prosody_check', 'semantic_evaluation', 'qafiya_validation', 'tashkeel',
            'bahr_selection'
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
    
    def test_format_prompt_verse_generation(self, prompt_manager):
        """Test formatting verse_generation prompt"""
        formatted = prompt_manager.format_prompt('verse_generation', 
                                                theme='الحب', 
                                                meter='بحر الكامل',
                                                tone='رومانسي',
                                                verse_count='4',
                                                rhyme_scheme='أأأأ',
                                                additional_requirements='استخدام التشبيهات')
        assert 'شاعر عربي مبدع' in formatted
        assert 'كتابة أبيات شعرية' in formatted
        assert 'الوزن الشعري المحدد' in formatted
    
    def test_format_prompt_imagery_creation(self, prompt_manager):
        """Test formatting imagery_creation prompt"""
        formatted = prompt_manager.format_prompt('imagery_creation',
                                                theme='الطبيعة',
                                                tone='هادئ',
                                                cultural_context='عربي',
                                                imagery_type='تشبيهات',
                                                requirements='صور حسية')
        assert 'خبير في الصور البلاغية والاستعارات' in formatted
        assert 'تشبيهات مناسبة' in formatted
        assert 'استعارات مبتكرة' in formatted
    
    def test_format_prompt_prosody_check(self, prompt_manager):
        """Test formatting prosody_check prompt"""
        formatted = prompt_manager.format_prompt('prosody_check',
                                                poem_text='بيت شعري تجريبي',
                                                expected_meter='بحر الكامل')
        assert 'خبير في العروض العربي والتحليل العروضي' in formatted
        assert 'فحص صحة الوزن والإيقاع' in formatted
        assert 'صحة الوزن العروضي' in formatted
    
    def test_format_prompt_semantic_evaluation(self, prompt_manager):
        """Test formatting semantic_evaluation prompt"""
        formatted = prompt_manager.format_prompt('semantic_evaluation',
                                                poem_text='نص شعري للتقييم',
                                                intended_theme='الحب',
                                                intended_tone='رومانسي',
                                                language_level='فصيح')
        assert 'ناقد أدبي متخصص في الشعر العربي' in formatted
        assert 'تقييم الجودة الأدبية والمعنى' in formatted
        assert 'وضوح المعنى والرسالة' in formatted
    
    def test_format_prompt_missing_parameters(self, prompt_manager):
        """Test formatting a prompt with missing parameters raises ValueError"""
        with pytest.raises(ValueError, match="Missing required parameters"):
            prompt_manager.format_prompt('unified_extraction')
    
    def test_get_template_info_unified_extraction(self, prompt_manager):
        """Test getting unified_extraction template info"""
        info = prompt_manager.get_template_info('unified_extraction')
        assert info['name'] == 'unified_extraction'
        assert info['category'] == 'constraint_inference'
        assert 'user_prompt' in info['parameters']
        assert info['metadata']['language'] == 'arabic'
    
    def test_get_template_info_verse_generation(self, prompt_manager):
        """Test getting verse_generation template info"""
        info = prompt_manager.get_template_info('verse_generation')
        assert info['name'] == 'verse_generation'
        assert info['category'] == 'generation'
        assert 'theme' in info['parameters']
        assert 'meter' in info['parameters']
        assert info['metadata']['language'] == 'arabic'
    
    def test_get_template_info_not_found(self, prompt_manager):
        """Test getting info for non-existent template raises KeyError"""
        with pytest.raises(KeyError, match="Template 'nonexistent' not found"):
            prompt_manager.get_template_info('nonexistent')
    
    def test_template_format_method(self, prompt_manager):
        """Test the format method of PromptTemplate class"""
        template = prompt_manager.get_template('unified_extraction')
        formatted = template.format(user_prompt='تحليل خاص')
        assert 'خبير في الشعر العربي والعروض' in formatted
        assert 'تحليل خاص' in formatted
    
    def test_template_format_missing_parameters(self, prompt_manager):
        """Test template format with missing parameters"""
        template = prompt_manager.get_template('unified_extraction')
        with pytest.raises(ValueError, match="Missing required parameters"):
            template.format() 