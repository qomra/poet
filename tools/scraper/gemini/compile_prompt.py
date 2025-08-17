import json
import os
from typing import List, Dict, Any

def load_diwan_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the diwan data from JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        List of poem data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_gemini_responses(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter data to only include entries with Gemini responses that have thought maps.
    
    Args:
        data: List of poem data
    
    Returns:
        List of poems with Gemini responses and thought maps
    """
    gemini_entries = []
    
    for item in data:
        ai_response = item.get('ai', {})
        model = ai_response.get('model', '').lower()
        
        # Check if it's a Gemini response with thought map
        if 'gemini' in model and 'thoughtMap' in ai_response and ai_response['thoughtMap']:
            gemini_entries.append(item)
    
    return gemini_entries

def create_analysis_prompt(gemini_entries: List[Dict[str, Any]]) -> str:
    """
    Create a comprehensive analysis prompt with all prompts and thought maps.
    
    Args:
        gemini_entries: List of poems with Gemini responses and thought maps
    
    Returns:
        Formatted analysis prompt
    """
    
    # Analysis instructions
    instructions = """# تحليل أنماط التفكير في Gemini Agent

## الهدف
تحليل أنماط التفكير والاستراتيجيات المستخدمة في نموذج Gemini عند كتابة الشعر العربي، لفهم كيفية تحويل هذه العملية إلى agent ذكي.

## المطلوب تحليله

### 1. أنماط البحث والاستكشاف
- ما هي أنواع المعلومات التي يبحث عنها؟
- كيف يبدأ عملية البحث؟
- ما هي المصادر أو الأنماط التي يستكشفها؟

### 2. استراتيجيات التحليل
- كيف يحلل المتطلبات والقيود؟
- ما هي المعايير التي يستخدمها في التقييم؟
- كيف يربط بين العناصر المختلفة؟

### 3. عملية التخطيط والتنظيم
- كيف ينظم أفكاره؟
- ما هي مراحل التخطيط التي يتبعها؟
- كيف يحدد الأولويات؟

### 4. أنماط الإبداع والتوليد
- كيف يولد الأفكار الجديدة؟
- ما هي التقنيات التي يستخدمها للإبداع؟
- كيف يتعامل مع القيود الإبداعية؟

### 5. استراتيجيات التحسين والمراجعة
- كيف يحسن النتائج؟
- ما هي معايير الجودة التي يطبقها؟
- كيف يتعامل مع الأخطاء أو النواقص؟

### 6. أنماط التفكير المنطقي
- كيف يربط السبب بالنتيجة؟
- ما هي القواعد المنطقية التي يتبعها؟
- كيف يتخذ القرارات؟

## البيانات للتحليل

"""
    
    # Add each prompt and thought map
    for i, entry in enumerate(gemini_entries, 1):
        prompt_text = entry.get('prompt', {}).get('text', '')
        thought_map = entry.get('ai', {}).get('thoughtMap', '')
        poem_id = entry.get('poem_id', 'Unknown')
        
        section = f"""
### مثال {i} - القصيدة رقم {poem_id}

#### النص التوجيهي (Prompt):
{prompt_text}

#### خريطة التفكير (Thought Map):
{thought_map}

---
"""
        instructions += section
    
    # Add conclusion
    instructions += """
## المطلوب في التحليل النهائي

بناءً على تحليل جميع الأمثلة أعلاه، قم بتقديم:

1. **ملخص شامل** لأنماط التفكير المستخدمة
2. **تصنيف للاستراتيجيات** المستخدمة في كل مرحلة
3. **اقتراح لهيكل Agent** يمكن أن يحاكي هذه الأنماط
4. **خطوات عملية** لتحويل هذه الأنماط إلى agent ذكي
5. **أدوات وتقنيات** مقترحة لتنفيذ كل استراتيجية

الهدف النهائي هو فهم كيفية بناء agent يمكنه محاكاة عملية التفكير هذه بشكل منهجي ومنظم.
"""
    
    return instructions

def save_analysis_prompt(content: str, output_file: str):
    """
    Save the analysis prompt to a file.
    
    Args:
        content: The analysis prompt content
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Main function to compile prompts and thought maps for analysis."""
    
    # File paths and parameters
    input_file = "./diwan.json"  # Use the merged file
    output_file = "gemini_thinking_analysis.md"
    n_samples = 20  # Number of samples to analyze
    
    try:
        # Load data
        print("Loading diwan data...")
        diwan_data = load_diwan_data(input_file)
        print(f"Loaded {len(diwan_data)} total entries")
        
        # Filter for Gemini responses with thought maps
        print("Filtering for Gemini responses with thought maps...")
        gemini_entries = filter_gemini_responses(diwan_data)
        print(f"Found {len(gemini_entries)} Gemini entries with thought maps")
        
        if not gemini_entries:
            print("No Gemini entries with thought maps found!")
            return
        
        # Sample the entries
        import random
        if len(gemini_entries) > n_samples:
            print(f"Sampling {n_samples} entries from {len(gemini_entries)} total entries...")
            gemini_entries = random.sample(gemini_entries, n_samples)
            # Sort by poem_id for consistent ordering
            gemini_entries.sort(key=lambda x: x.get('poem_id', 0))
        else:
            print(f"Using all {len(gemini_entries)} entries (less than requested {n_samples})")
        
        # Create analysis prompt
        analysis_prompt = create_analysis_prompt(gemini_entries)
        
        # Save to file
        print(f"Saving analysis prompt to {output_file}...")
        save_analysis_prompt(analysis_prompt, output_file)
        
        print(f"Successfully created analysis prompt with {len(gemini_entries)} examples")
        print(f"Analysis prompt saved to: {output_file}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Total entries processed: {len(diwan_data)}")
        print(f"- Gemini entries with thought maps found: {len(filter_gemini_responses(diwan_data))}")
        print(f"- Samples analyzed: {len(gemini_entries)}")
        print(f"- Analysis prompt created with comprehensive instructions")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
