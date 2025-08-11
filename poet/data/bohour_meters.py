# poet/data/bohour_meters.py

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from poet.models.constraints import Constraints
from poet.utils.bohour import bohours_list
from poet.utils.bohour.bahr import Bahr
from poet.utils.bohour.tafeela import Tafeela


@dataclass
class MeterInfo:
    """Information about a specific meter/bahr"""
    name: str
    arabic_name: str
    tafeelat: List[str]
    sub_bahrs: List[str]
    description: str
    examples: List[str]
    common_themes: List[str]
    difficulty_level: str  # easy, medium, hard


class BohourMetersManager:
    """
    Data manager for Arabic poetry meters (bohour) that can enrich Constraints
    with meter tafeelat and related information.
    """
    
    def __init__(self):
        """Initialize the meters manager"""
        self._meters_cache: Dict[str, MeterInfo] = {}
        self._name_mapping: Dict[str, str] = {}
        self._initialize_meters()
    
    def _initialize_meters(self):
        """Initialize meter information from bohour library"""
        # Arabic names mapping
        arabic_names = {
            "Taweel": "بحر الطويل",
            "Madeed": "بحر المديد", 
            "Baseet": "بحر البسيط",
            "Wafer": "بحر الوافر",
            "Kamel": "بحر الكامل",
            "Hazaj": "بحر الهزج",
            "Rajaz": "بحر الرجز",
            "Ramal": "بحر الرمل",
            "Saree": "بحر السريع",
            "Munsareh": "بحر المنسرح",
            "Khafeef": "بحر الخفيف",
            "Mudhare": "بحر المضارع",
            "Muqtadheb": "بحر المقتضب",
            "Mujtath": "بحر المجتث",
            "Mutakareb": "بحر المتقارب",
            "Mutadarak": "بحر المحدث",
            # Sub-bahrs
            "BaseetMajzoo": "بحر البسيط المجزوء",
            "BaseetMukhalla": "بحر البسيط المخلى",
            "WaferMajzoo": "بحر الوافر المجزوء",
            "KamelMajzoo": "بحر الكامل المجزوء",
            "RajazMajzoo": "بحر الرجز المجزوء",
            "RajazMashtoor": "بحر الرجز المشطور",
            "RajazManhook": "بحر الرجز المنهوك",
            "RamalMajzoo": "بحر الرمل المجزوء",
            "SareeMashtoor": "بحر السريع المشطور",
            "MunsarehManhook": "بحر المنسرح المنهوك",
            "KhafeefMajzoo": "بحر الخفيف المجزوء",
            "MutakarebMajzoo": "بحر المتقارب المجزوء",
            "MutadarakMajzoo": "بحر المحدث المجزوء",
            "MutadarakMashtoor": "بحر المحدث المشطور",
        }
        
        # Descriptions and examples
        meter_descriptions = {
            "Taweel": "أطول بحور الشعر العربي، مناسب للقصائد الطويلة والملاحم",
            "Madeed": "بحر متوسط الطول، مناسب للغزل والمدح",
            "Baseet": "بحر واسع ومتعدد الأغراض، مناسب لجميع أنواع الشعر",
            "Wafer": "بحر قوي وموسيقي، مناسب للحماسة والفخر",
            "Kamel": "بحر كامل ومتوازن، مناسب لجميع الأغراض الشعرية",
            "Hazaj": "بحر خفيف وموسيقي، مناسب للغزل والوصف",
            "Rajaz": "بحر سريع وخفيف، مناسب للهجاء والسخرية",
            "Ramal": "بحر متوسط، مناسب للغزل والوصف",
            "Saree": "بحر سريع، مناسب للهجاء والحماسة",
            "Munsareh": "بحر منسرح، مناسب للوصف والغزل",
            "Khafeef": "بحر خفيف، مناسب للغزل والوصف",
            "Mudhare": "بحر مضارع، مناسب للغزل والوصف",
            "Muqtadheb": "بحر مقتضب، مناسب للهجاء والسخرية",
            "Mujtath": "بحر مجتث، مناسب للغزل والوصف",
            "Mutakareb": "بحر متقارب، مناسب للغزل والوصف",
            "Mutadarak": "بحر محدث، مناسب للغزل والوصف",
        }
        
        # Common themes for each meter
        meter_themes = {
            "Taweel": ["ملاحم", "قصائد طويلة", "مدح", "فخر"],
            "Madeed": ["غزل", "مدح", "وصف"],
            "Baseet": ["جميع الأغراض", "غزل", "مدح", "هجاء", "وصف"],
            "Wafer": ["حماسة", "فخر", "مدح"],
            "Kamel": ["جميع الأغراض", "غزل", "مدح", "وصف"],
            "Hazaj": ["غزل", "وصف", "رثاء"],
            "Rajaz": ["هجاء", "سخرية", "وصف"],
            "Ramal": ["غزل", "وصف", "رثاء"],
            "Saree": ["هجاء", "حماسة", "وصف"],
            "Munsareh": ["وصف", "غزل", "رثاء"],
            "Khafeef": ["غزل", "وصف", "رثاء"],
            "Mudhare": ["غزل", "وصف"],
            "Muqtadheb": ["هجاء", "سخرية"],
            "Mujtath": ["غزل", "وصف"],
            "Mutakareb": ["غزل", "وصف"],
            "Mutadarak": ["غزل", "وصف"],
        }
        
        # Difficulty levels
        difficulty_levels = {
            "Taweel": "medium",
            "Madeed": "easy",
            "Baseet": "medium",
            "Wafer": "easy",
            "Kamel": "medium",
            "Hazaj": "easy",
            "Rajaz": "easy",
            "Ramal": "medium",
            "Saree": "hard",
            "Munsareh": "hard",
            "Khafeef": "medium",
            "Mudhare": "easy",
            "Muqtadheb": "hard",
            "Mujtath": "medium",
            "Mutakareb": "medium",
            "Mutadarak": "medium",
        }
        
        # Examples for each meter
        meter_examples = {
            "Taweel": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
            ],
            "Madeed": [
                "يَا لَيْلَةَ الحُبِّ وَالأَحْلامِ",
                "فِي ظِلِّكَ العَذْبِ وَالإِسْلامِ"
            ],
            "Baseet": [
                "أَلاَ يَا حَادِي العِيسِ قِفْ بِالرَّكْبِ",
                "وَسَلِّمْ عَلَى أَهْلِ الحِمَى مِنْ أَهْلِ الحُبِّ"
            ],
            "Wafer": [
                "يَا أَيُّهَا المَلِكُ المُعَظَّمُ شَأْنُهُ",
                "فِي كُلِّ أَرْضٍ ذِكْرُهُ يَتَرَدَّدُ"
            ],
            "Kamel": [
                "وَمُتَيَّمٍ جَرَحَ الفُراقُ فُؤادَهُ",
                "فَالدَّمْعُ مِنْ أَجْفانِهِ يَتَدَفَّقُ"
            ],
            "Hazaj": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Rajaz": [
                "تَمَكَّنَ هَذا الدَّهْرُ مِمَّا يَسُوءُني",
                "وَلَجَّ فَما يَخْلي صَفاتِيَ مِنْ قَرْعِ"
            ],
            "Ramal": [
                "أَلاَ يَا حَادِي العِيسِ قِفْ بِالرَّكْبِ",
                "وَسَلِّمْ عَلَى أَهْلِ الحِمَى مِنْ أَهْلِ الحُبِّ"
            ],
            "Saree": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Munsareh": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Khafeef": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Mudhare": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Muqtadheb": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Mujtath": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Mutakareb": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
            "Mutadarak": [
                "يَا رَاكِباً مِنْ عِنْدَنَا مُتَيَمَّماً",
                "قِفْ بِالرَّكْبِ وَاسْأَلْ عَنْ أَحْوالِنَا"
            ],
        }
        
        # Build meter information for each bahr class
        for bahr_class in bohours_list:
            bahr_name = bahr_class.__name__
            arabic_name = arabic_names.get(bahr_name, bahr_name)
            
            # Get tafeelat names
            tafeelat_names = []
            if hasattr(bahr_class, 'tafeelat'):
                for tafeela_class in bahr_class.tafeelat:
                    if hasattr(tafeela_class, 'name'):
                        tafeelat_names.append(tafeela_class.name)
            
            # Get sub-bahrs names
            sub_bahrs_names = []
            if hasattr(bahr_class, 'sub_bahrs'):
                for sub_bahr_class in bahr_class.sub_bahrs:
                    sub_bahrs_names.append(arabic_names.get(sub_bahr_class.__name__, sub_bahr_class.__name__))
            
            # Create MeterInfo
            meter_info = MeterInfo(
                name=bahr_name,
                arabic_name=arabic_name,
                tafeelat=tafeelat_names,
                sub_bahrs=sub_bahrs_names,
                description=meter_descriptions.get(bahr_name, ""),
                examples=meter_examples.get(bahr_name, []),
                common_themes=meter_themes.get(bahr_name, []),
                difficulty_level=difficulty_levels.get(bahr_name, "medium")
            )
            
            # Cache the meter info
            self._meters_cache[bahr_name] = meter_info
            self._meters_cache[arabic_name] = meter_info
            
            # Create name mappings
            self._name_mapping[bahr_name] = arabic_name
            self._name_mapping[arabic_name] = bahr_name
        
        # Also add sub-bahrs as standalone meters
        for bahr_class in bohours_list:
            if hasattr(bahr_class, 'sub_bahrs'):
                for sub_bahr_class in bahr_class.sub_bahrs:
                    sub_bahr_name = sub_bahr_class.__name__
                    sub_arabic_name = arabic_names.get(sub_bahr_name, sub_bahr_name)
                    
                    # Get tafeelat names for sub-bahr
                    sub_tafeelat_names = []
                    if hasattr(sub_bahr_class, 'tafeelat'):
                        for tafeela_class in sub_bahr_class.tafeelat:
                            if hasattr(tafeela_class, 'name'):
                                sub_tafeelat_names.append(tafeela_class.name)
                    
                    # Create MeterInfo for sub-bahr
                    sub_meter_info = MeterInfo(
                        name=sub_bahr_name,
                        arabic_name=sub_arabic_name,
                        tafeelat=sub_tafeelat_names,
                        sub_bahrs=[],  # Sub-bahrs don't have their own sub-bahrs
                        description=f"مجزوء من {arabic_names.get(bahr_class.__name__, bahr_class.__name__)}",
                        examples=meter_examples.get(bahr_name, []),  # Use parent's examples
                        common_themes=meter_themes.get(bahr_name, []),  # Use parent's themes
                        difficulty_level="hard"  # Sub-bahrs are generally harder
                    )
                    
                    # Cache the sub-bahr meter info
                    self._meters_cache[sub_bahr_name] = sub_meter_info
                    self._meters_cache[sub_arabic_name] = sub_meter_info
                    
                    # Create name mappings for sub-bahr
                    self._name_mapping[sub_bahr_name] = sub_arabic_name
                    self._name_mapping[sub_arabic_name] = sub_bahr_name
    
    def get_meter_info(self, meter_name: str) -> Optional[MeterInfo]:
        """
        Get information about a specific meter.
        
        Args:
            meter_name: Name of the meter (Arabic or English)
            
        Returns:
            MeterInfo object or None if not found
        """
        return self._meters_cache.get(meter_name)
    
    def get_all_meters(self) -> List[MeterInfo]:
        """Get all available meters"""
        # Return unique meters (avoid duplicates from Arabic/English names)
        seen_names = set()
        meters = []
        for meter_info in self._meters_cache.values():
            if meter_info.name not in seen_names:
                meters.append(meter_info)
                seen_names.add(meter_info.name)
        return meters
    
    def search_meters(self, query: str) -> List[MeterInfo]:
        """
        Search for meters by name, description, or themes.
        
        Args:
            query: Search query
            
        Returns:
            List of matching meters
        """
        query = query.lower()
        results = []
        
        for meter_info in self.get_all_meters():
            # Search in name
            if query in meter_info.name.lower() or query in meter_info.arabic_name.lower():
                results.append(meter_info)
                continue
            
            # Search in description
            if query in meter_info.description.lower():
                results.append(meter_info)
                continue
            
            # Search in themes
            for theme in meter_info.common_themes:
                if query in theme.lower():
                    results.append(meter_info)
                    break
        
        return results
    
    def get_meters_by_theme(self, theme: str) -> List[MeterInfo]:
        """
        Get meters suitable for a specific theme.
        
        Args:
            theme: Theme to search for (e.g., "غزل", "مدح", "هجاء")
            
        Returns:
            List of meters suitable for the theme
        """
        theme = theme.lower()
        results = []
        
        for meter_info in self.get_all_meters():
            for meter_theme in meter_info.common_themes:
                if theme in meter_theme.lower():
                    results.append(meter_info)
                    break
        
        return results
    
    def get_meters_by_difficulty(self, difficulty: str) -> List[MeterInfo]:
        """
        Get meters by difficulty level.
        
        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            
        Returns:
            List of meters with the specified difficulty
        """
        return [meter for meter in self.get_all_meters() if meter.difficulty_level == difficulty]
    
    def enrich_constraints(self, constraints: Constraints) -> Constraints:
        """
        Enrich constraints with meter tafeelat and related information.
        Handles both main bahrs and sub-bahrs.
        
        Args:
            constraints: Original constraints
            
        Returns:
            Enriched constraints with meeter_tafeelat field populated
        """
        if not constraints.meter:
            return constraints
        
        # First, try to get meter info directly (handles both main and sub-bahrs)
        meter_info = self.get_meter_info(constraints.meter)
        
        # If not found, check if it's a sub-bahr of a main bahr
        if not meter_info:
            # Try to find a main bahr that has this as a sub-bahr
            for main_meter in self.get_all_meters():
                if constraints.meter in main_meter.sub_bahrs:
                    # Found a main bahr that has this sub-bahr
                    # Get the sub-bahr info directly from the bohour library
                    sub_bahr_info = self._get_sub_bahr_info(constraints.meter)
                    if sub_bahr_info:
                        meter_info = sub_bahr_info
                        break
        
        if not meter_info:
            return constraints
        
        # Create enriched constraints
        enriched_constraints = Constraints(
            meter=constraints.meter,
            meeter_tafeelat=" ".join(meter_info.tafeelat),  # Join tafeelat with spaces
            qafiya=constraints.qafiya,
            qafiya_harakah=constraints.qafiya_harakah,
            qafiya_type=constraints.qafiya_type,
            line_count=constraints.line_count,
            theme=constraints.theme,
            tone=constraints.tone,
            imagery=constraints.imagery,
            keywords=constraints.keywords,
            sections=constraints.sections,
            register=constraints.register,
            era=constraints.era,
            poet_style=constraints.poet_style,
            ambiguities=constraints.ambiguities,
            original_prompt=constraints.original_prompt
        )
        
        return enriched_constraints
    
    def _get_sub_bahr_info(self, sub_bahr_name: str) -> Optional[MeterInfo]:
        """
        Get information about a specific sub-bahr directly from the bohour library.
        
        Args:
            sub_bahr_name: Name of the sub-bahr (Arabic)
            
        Returns:
            MeterInfo object for the sub-bahr or None if not found
        """
        # Map Arabic sub-bahr names to their class names
        sub_bahr_mapping = {
            "بحر الكامل المجزوء": "KamelMajzoo",
            "بحر الوافر المجزوء": "WaferMajzoo", 
            "بحر البسيط المجزوء": "BaseetMajzoo",
            "بحر البسيط المخلى": "BaseetMukhalla",
            "بحر الرجز المجزوء": "RajazMajzoo",
            "بحر الرجز المشطور": "RajazMashtoor",
            "بحر الرجز المنهوك": "RajazManhook",
            "بحر الرمل المجزوء": "RamalMajzoo",
            "بحر السريع المشطور": "SareeMashtoor",
            "بحر المنسرح المنهوك": "MunsarehManhook",
            "بحر الخفيف المجزوء": "KhafeefMajzoo",
            "بحر المتقارب المجزوء": "MutakarebMajzoo",
            "بحر المحدث المجزوء": "MutadarakMajzoo",
            "بحر المحدث المشطور": "MutadarakMashtoor",
        }
        
        class_name = sub_bahr_mapping.get(sub_bahr_name)
        if not class_name:
            return None
        
        # Get the class from bohour library
        try:
            from poet.utils.bohour.bahr import (
                KamelMajzoo, WaferMajzoo, BaseetMajzoo, BaseetMukhalla,
                RajazMajzoo, RajazMashtoor, RajazManhook, RamalMajzoo,
                SareeMashtoor, MunsarehManhook, KhafeefMajzoo,
                MutakarebMajzoo, MutadarakMajzoo, MutadarakMashtoor
            )
            
            # Map class names to actual classes
            bahr_classes = {
                "KamelMajzoo": KamelMajzoo,
                "WaferMajzoo": WaferMajzoo,
                "BaseetMajzoo": BaseetMajzoo,
                "BaseetMukhalla": BaseetMukhalla,
                "RajazMajzoo": RajazMajzoo,
                "RajazMashtoor": RajazMashtoor,
                "RajazManhook": RajazManhook,
                "RamalMajzoo": RamalMajzoo,
                "SareeMashtoor": SareeMashtoor,
                "MunsarehManhook": MunsarehManhook,
                "KhafeefMajzoo": KhafeefMajzoo,
                "MutakarebMajzoo": MutakarebMajzoo,
                "MutadarakMajzoo": MutadarakMajzoo,
                "MutadarakMashtoor": MutadarakMashtoor,
            }
            
            bahr_class = bahr_classes.get(class_name)
            if not bahr_class:
                return None
            
            # Create MeterInfo for the sub-bahr
            tafeelat_names = []
            if hasattr(bahr_class, 'tafeelat'):
                for tafeela_class in bahr_class.tafeelat:
                    if hasattr(tafeela_class, 'name'):
                        tafeelat_names.append(tafeela_class.name)
            
            # Get description and themes from parent bahr or use defaults
            parent_bahr_name = self._get_parent_bahr_name(sub_bahr_name)
            parent_info = self.get_meter_info(parent_bahr_name) if parent_bahr_name else None
            
            description = f"مجزوء من {parent_bahr_name}" if parent_info else f"مجزوء من {sub_bahr_name}"
            themes = parent_info.common_themes if parent_info else ["جميع الأغراض"]
            difficulty = parent_info.difficulty_level if parent_info else "medium"
            examples = parent_info.examples if parent_info else []
            
            return MeterInfo(
                name=class_name,
                arabic_name=sub_bahr_name,
                tafeelat=tafeelat_names,
                sub_bahrs=[],  # Sub-bahrs don't have their own sub-bahrs
                description=description,
                examples=examples,
                common_themes=themes,
                difficulty_level=difficulty
            )
            
        except (ImportError, AttributeError):
            return None
    
    def _get_parent_bahr_name(self, sub_bahr_name: str) -> Optional[str]:
        """
        Get the parent bahr name for a given sub-bahr.
        
        Args:
            sub_bahr_name: Name of the sub-bahr
            
        Returns:
            Parent bahr name or None if not found
        """
        parent_mapping = {
            "بحر الكامل المجزوء": "بحر الكامل",
            "بحر الوافر المجزوء": "بحر الوافر",
            "بحر البسيط المجزوء": "بحر البسيط",
            "بحر البسيط المخلى": "بحر البسيط",
            "بحر الرجز المجزوء": "بحر الرجز",
            "بحر الرجز المشطور": "بحر الرجز",
            "بحر الرجز المنهوك": "بحر الرجز",
            "بحر الرمل المجزوء": "بحر الرمل",
            "بحر السريع المشطور": "بحر السريع",
            "بحر المنسرح المنهوك": "بحر المنسرح",
            "بحر الخفيف المجزوء": "بحر الخفيف",
            "بحر المتقارب المجزوء": "بحر المتقارب",
            "بحر المحدث المجزوء": "بحر المحدث",
            "بحر المحدث المشطور": "بحر المحدث",
        }
        
        return parent_mapping.get(sub_bahr_name)
    
    def suggest_meter_for_theme(self, theme: str, difficulty: Optional[str] = None) -> List[MeterInfo]:
        """
        Suggest meters suitable for a given theme and optional difficulty.
        
        Args:
            theme: Theme of the poem
            difficulty: Optional difficulty level filter
            
        Returns:
            List of suggested meters
        """
        # Get meters for the theme
        theme_meters = self.get_meters_by_theme(theme)
        
        # Filter by difficulty if specified
        if difficulty:
            theme_meters = [meter for meter in theme_meters if meter.difficulty_level == difficulty]
        
        # Sort by difficulty (easy first)
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        theme_meters.sort(key=lambda m: difficulty_order.get(m.difficulty_level, 1))
        
        return theme_meters
    
    def get_meter_tafeelat(self, meter_name: str) -> List[str]:
        """
        Get tafeelat for a specific meter.
        
        Args:
            meter_name: Name of the meter
            
        Returns:
            List of tafeelat names
        """
        meter_info = self.get_meter_info(meter_name)
        return meter_info.tafeelat if meter_info else []
    
    def get_sub_bahrs(self, meter_name: str) -> List[str]:
        """
        Get sub-bahrs for a specific meter.
        
        Args:
            meter_name: Name of the meter
            
        Returns:
            List of sub-bahr names
        """
        meter_info = self.get_meter_info(meter_name)
        return meter_info.sub_bahrs if meter_info else []
    
    def validate_meter(self, meter_name: str) -> bool:
        """
        Validate if a meter name is recognized.
        
        Args:
            meter_name: Name of the meter to validate
            
        Returns:
            True if meter is recognized, False otherwise
        """
        return meter_name in self._meters_cache
    
    def get_meter_examples(self, meter_name: str) -> List[str]:
        """
        Get example verses for a specific meter.
        
        Args:
            meter_name: Name of the meter
            
        Returns:
            List of example verses
        """
        meter_info = self.get_meter_info(meter_name)
        return meter_info.examples if meter_info else []
    
    def get_meter_themes(self, meter_name: str) -> List[str]:
        """
        Get common themes for a specific meter.
        
        Args:
            meter_name: Name of the meter
            
        Returns:
            List of common themes
        """
        meter_info = self.get_meter_info(meter_name)
        return meter_info.common_themes if meter_info else []
