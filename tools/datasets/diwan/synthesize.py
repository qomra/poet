#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poem Prompt Synthesizer

This script takes extracted poems with Gemini-2.5-pro generated prompts and sends them to GPT-5
to generate poems, creating training data pairs of (prompt, poem) for Arabic poetry generation.

Usage: python synthesize.py --poems_file diwan_with_gemini.json --output_file gpt5_generated_poems.json
"""

import json
import argparse
import os
import time
import random
from typing import List, Dict, Optional, Set, Tuple
import re

# API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Use: pip install openai")

# API Keys - set these as environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser(description="Arabic Poetry Prompt Synthesizer")
    parser.add_argument("--poems_file", type=str, default="diwan_with_gemini.json", 
                       help="JSON file containing extracted poems with Gemini prompts")
    parser.add_argument("--output_file", type=str, default="gpt5_generated_poems.json",
                       help="Output JSON file for prompt-poem pairs")
    parser.add_argument("--sample_size", type=int, default=1,
                       help="Number of random poems to sample and process")
    parser.add_argument("--min_lines", type=int, default=4,
                       help="Minimum number of lines a poem must have to be processed")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls (seconds)")
    parser.add_argument("--augment", action="store_true",
                       help="Augment existing results instead of overwriting")
    return parser.parse_args()

def openai_complete(prompt: str, model_name: str = "gpt-5-2025-08-07") -> str:
    """Complete prompt using OpenAI API"""
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        raise ValueError("OpenAI API not available or API key not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        print(f"  Sending request to {model_name}...")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=15000,
            temperature=1
        )
        response = completion.choices[0].message.content
        print(f"  Response length: {len(response)} characters")
        print(f"  Raw response: '{response}'")
        print(f"  Finish reason: {completion.choices[0].finish_reason}")
        print(f"  Usage: {completion.usage}")
        return response
    except Exception as e:
        print(f"OpenAI API error: {e}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Model: {model_name}")
        print(f"  Prompt length: {len(prompt)} characters")
        print(f"  Prompt preview: {prompt[:200]}...")
        return ""

def load_existing_results(output_file: str) -> Tuple[List[Dict], Set[str]]:
    """Load existing results and return GPT-5 processed poem IDs"""
    existing_results = []
    gpt5_processed_ids = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # Extract GPT-5 processed poem IDs (poems that have openai provider in ai section)
            for result in existing_results:
                if 'poem_id' in result and result.get('ai', {}).get('provider') == 'openai':
                    gpt5_processed_ids.add(result['poem_id'])
            
            print(f"Found {len(existing_results)} existing results")
            print(f"GPT-5 processed: {len(gpt5_processed_ids)} poems")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            print("Starting fresh...")
    
    return existing_results, gpt5_processed_ids

def extract_gemini_poems(poems: List[Dict]) -> List[Dict]:
    """Extract all Gemini-2.5-pro poems from the dataset"""
    gemini_poems = []
    
    for poem in poems:
        poem_id = poem.get("poem_id")
        reference = poem.get("reference", {})
        prompt = poem.get("prompt", {})
        ai = poem.get("ai", {})
        
        gemini_provider = ai.get("provider", "")
        gemini_model = ai.get("model", "")
        
        # Only include if it's a Gemini-2.5-pro poem
        if gemini_provider == "gemini" and "2.5-pro" in gemini_model and ai.get("text", "").strip():
            gemini_poems.append({
                "poem_id": poem_id,
                "reference": reference,
                "prompt": prompt,
                "ai": ai
            })
    
    print(f"Extracted {len(gemini_poems)} Gemini-2.5-pro poems")
    return gemini_poems

def sample_poems(poems: List[Dict], sample_size: int, gpt5_processed_ids: Set[str], min_lines: int = 2) -> List[Dict]:
    """Sample poems randomly, excluding already GPT-5 processed ones and filtering by minimum lines"""
    # Filter out already GPT-5 processed poems and poems with too few lines
    unprocessed_poems = []
    too_few_lines_count = 0
    
    for poem in poems:
        if poem.get('poem_id') not in gpt5_processed_ids:
            poem_text = poem.get('reference', {}).get('poem', '')
            line_count = len([line.strip() for line in poem_text.split('\n') if line.strip()])
            if line_count >= min_lines:
                unprocessed_poems.append(poem)
            else:
                too_few_lines_count += 1
    
    already_gpt5_processed_count = len([p for p in poems if p.get('poem_id') in gpt5_processed_ids])
    
    print(f"Total Gemini poems: {len(poems)}")
    print(f"Already GPT-5 processed: {already_gpt5_processed_count}")
    print(f"Too few lines (< {min_lines}): {too_few_lines_count}")
    print(f"Available for GPT-5 processing: {len(unprocessed_poems)}")
    
    if not unprocessed_poems:
        print("No unprocessed poems available for GPT-5!")
        return []
    
    if sample_size is None or sample_size >= len(unprocessed_poems):
        print(f"Processing all {len(unprocessed_poems)} unprocessed poems with GPT-5")
        return unprocessed_poems
    
    # Random sample without replacement
    sampled_poems = random.sample(unprocessed_poems, sample_size)
    print(f"Randomly sampled {len(sampled_poems)} poems from {len(unprocessed_poems)} available for GPT-5")
    
    return sampled_poems

def load_poems(poems_file: str) -> List[Dict]:
    """Load poems from JSON file"""
    with open(poems_file, 'r', encoding='utf-8') as f:
       all_poems = json.load(f)
    print(f"Loaded {len(all_poems)} poems from {poems_file}")
    return all_poems

def synthesize_poem(poem_data: Dict, completion_func, model_name: str) -> Optional[Dict]:
    """Synthesize a prompt for a single poem"""
    poem_id = poem_data.get("poem_id", "")
    prompt_text = poem_data.get("prompt", {}).get("text", "")
    
    base_prompt = "أنت شاعر يبني القصائد بناءً على المتطلبات المحددة.\n\n" \
                  "المتطلبات:\n" \
                  f"{prompt_text}\n\n" \
                  "اكتب كل شطر في سطر جديد. لا تكتب أي شيء آخر غير القصيدة ولا حتى مقدمات أو شرح.\n\n"
    
    try:
        # Get AI-generated poem using GPT-5
        generated_poem = completion_func(base_prompt, model_name)
        
        if not generated_poem.strip():
            print(f"Empty response for poem {poem_id}")
            print(f"  Raw response: '{generated_poem}'")
            return None
        
        # Create a new record for GPT-5, copying the original structure
        result = {
            "poem_id": poem_id,
            "reference": poem_data.get("reference", {}),
            "prompt": poem_data.get("prompt", {}),
            "ai": {
                "text": generated_poem.strip(),
                "provider": "openai",
                "model": model_name
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing poem {poem_id}: {e}")
        return None

def save_results(results: List[Dict], existing_results: List[Dict], output_file: str, augment: bool = True):
    """Save results to JSON file"""
    if augment:
        # Combine existing and new results
        all_results = existing_results + results
    else:
        all_results = results
    
    # Save JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

def save_incremental(new_result: Dict, existing_results: List[Dict], output_file: str):
    """Save a single new result incrementally"""
    try:
        # Add new result to existing ones
        all_results = existing_results + [new_result]
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving incremental update: {e}")
        return False

def main():
    """Main function"""
    args = parse_args()

    # Always load existing results
    existing_results, gpt5_processed_ids = load_existing_results(args.output_file)
    
    # Load poems
    all_poems = load_poems(args.poems_file)
    if not all_poems:
        print("No poems to process")
        return
    
    # Extract all Gemini-2.5-pro poems first
    gemini_poems = extract_gemini_poems(all_poems)
    if not gemini_poems:
        print("No Gemini-2.5-pro poems found")
        return
    
    # Always load existing results first, then add Gemini poems if needed
    if os.path.exists(args.output_file) and existing_results:
        # File exists and has content - preserve existing results
        existing_poem_ids = {result['poem_id'] for result in existing_results}
        missing_gemini_poems = [poem for poem in gemini_poems if poem['poem_id'] not in existing_poem_ids]
        
        if missing_gemini_poems:
            print(f"Adding {len(missing_gemini_poems)} missing Gemini poems to existing results...")
            existing_results.extend(missing_gemini_poems)
            # Save the updated results
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
            print(f"Updated file with {len(existing_results)} total poems")
        else:
            print("All Gemini poems already exist in the file")
    else:
        # No existing file or no content - create new file with all Gemini poems
        print("Creating new file with all Gemini-2.5-pro poems...")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(gemini_poems, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(gemini_poems)} Gemini-2.5-pro poems to {args.output_file}")
        existing_results = gemini_poems
    
    # Sample Gemini poems for GPT-5 processing (excluding already GPT-5 processed ones)
    poems_to_process = sample_poems(gemini_poems, args.sample_size, gpt5_processed_ids, args.min_lines)
    if not poems_to_process:
        print("No unprocessed Gemini poems available for GPT-5 generation")
        return

    print(f"Processing {len(poems_to_process)} Gemini poems with GPT-5...")
    print("Saving after each successful generation...")
    
    results = []
    successful_count = 0
    model = "gpt-5-2025-08-07"
    
    for i, poem_data in enumerate(poems_to_process, 1):
        poem_id = poem_data.get('poem_id', 'N/A')
        print(f"Processing poem {i}/{len(poems_to_process)} (ID: {poem_id})")
        
        print(f"Using provider: openai, model: {model}")
        result = synthesize_poem(
            poem_data, 
            openai_complete,
            model_name=model
        )
        
        if result:
            results.append(result)
            successful_count += 1
            
            # Save incrementally after each successful generation
            if save_incremental(result, existing_results, args.output_file):
                print(f"✓ Generated and saved GPT-5 poem for {poem_id} ({successful_count} total)")
                # Update existing_results to include the new result for next iteration
                existing_results.append(result)
            else:
                print(f"✓ Generated GPT-5 poem for {poem_id} but failed to save")
        else:
            print(f"✗ Failed to generate GPT-5 poem for {poem_id}")
        
        # Add delay between API calls
        if i < len(poems_to_process):
            time.sleep(args.delay)
    
    print(f"\nCompleted! Generated GPT-5 poems for {len(results)}/{len(poems_to_process)} Gemini poems")
    print(f"All results saved incrementally to {args.output_file}")
    
    # Final save with summary
    if results:
        final_total = len(existing_results)
        print(f"\nFinal dataset size: {final_total} poems (Gemini + GPT-5)")
    else:
        print("No new GPT-5 results generated")

if __name__ == "__main__":
    main()
