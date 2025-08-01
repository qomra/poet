import json
import os
import re
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional

def extract_content_from_html(html_content: str) -> Dict[str, str]:
    """
    Extract response and thinking content from HTML file.
    
    Args:
        html_content: HTML content as string
    
    Returns:
        Dictionary with 'response' and 'thinking' keys
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract response content
    response_content = ""
    response_div = soup.find('div', class_='response')
    if response_div:
        # Find the actual response text (not the thinking part)
        # Look for message-content with class 'model-response-text'
        response_text_div = response_div.find('div', class_='model-response-text')
        if response_text_div:
            # Get the markdown content
            markdown_div = response_text_div.find('div', class_='markdown')
            if markdown_div:
                # Convert HTML to text with proper line breaks
                response_content = convert_html_to_text(markdown_div)
        else:
            # Alternative: look for any markdown div in response that's not in thinking
            markdown_divs = response_div.find_all('div', class_='markdown')
            for markdown_div in markdown_divs:
                # Check if this markdown div is not inside the thinking section
                # AND not inside model-thoughts
                if (not markdown_div.find_parent('div', class_='thinking') and 
                    not markdown_div.find_parent('model-thoughts')):
                    response_content = convert_html_to_text(markdown_div)
                    break
    
    # Extract thinking content
    thinking_content = ""
    thinking_div = soup.find('div', class_='thinking')
    if thinking_div:
        # Find the markdown content in thinking section
        markdown_div = thinking_div.find('div', class_='markdown')
        if markdown_div:
            raw_thinking = convert_html_to_text(markdown_div)
            thinking_content = format_thinking_content(raw_thinking)
    
    return {
        'response': response_content,
        'thinking': thinking_content
    }

def convert_html_to_text(element) -> str:
    """
    Convert HTML element to text with proper line breaks.
    
    Args:
        element: BeautifulSoup element
    
    Returns:
        Text with proper line breaks
    """
    # Replace <p> tags with newlines
    for p in element.find_all('p'):
        p.replace_with('\n' + p.get_text() + '\n')
    
    # Replace <br> tags with newlines
    for br in element.find_all('br'):
        br.replace_with('\n')
    
    # Get the text and clean it up
    text = element.get_text()
    
    # Clean up multiple newlines and whitespace
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]  # Remove empty lines
    
    return '\n'.join(lines)

def format_thinking_content(text: str) -> str:
    """
    Format thinking content to be more readable by adding line breaks between sections.
    
    Args:
        text: Raw thinking text
    
    Returns:
        Formatted thinking text with proper spacing
    """
    # Split by common thinking section patterns
    sections = []
    current_section = ""
    
    lines = text.split('\n')
    for line in lines:
        # Check if this line starts a new section (contains "strong" or starts with common patterns)
        if (line.strip().startswith('**') or 
            any(keyword in line.lower() for keyword in ['establishing', 'pinpointing', 'commencing', 'diving', 'focusing', 'analyzing', 'crafting', 'building', 'refining', 'expanding', 'integrating', 'composing', 'generating', 'investigating', 'uncovering', 'grasping', 'initiating'])):
            if current_section:
                sections.append(current_section.strip())
            current_section = line
        else:
            current_section += '\n' + line if current_section else line
    
    # Add the last section
    if current_section:
        sections.append(current_section.strip())
    
    # Join sections with double line breaks for better readability
    return '\n\n'.join(sections)

def get_poem_id_from_filename(filename: str) -> Optional[int]:
    """
    Extract poem ID from filename like 'poem_123_response.html'
    
    Args:
        filename: HTML filename
    
    Returns:
        Poem ID as integer or None if not found
    """
    match = re.search(r'poem_(\d+)_response\.html', filename)
    if match:
        return int(match.group(1))
    return None

def load_diwan_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the original diwan.json data.
    
    Args:
        file_path: Path to diwan.json
    
    Returns:
        List of poem data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_gemini_responses(diwan_data: List[Dict[str, Any]], html_folder: str) -> List[Dict[str, Any]]:
    """
    Merge Gemini responses with original diwan data.
    
    Args:
        diwan_data: Original diwan data
        html_folder: Folder containing HTML files
    
    Returns:
        Updated diwan data with Gemini responses
    """
    # Create a mapping of poem_id to diwan entries
    poem_groups = {}
    for item in diwan_data:
        poem_id = item.get('poem_id')
        if poem_id not in poem_groups:
            poem_groups[poem_id] = []
        poem_groups[poem_id].append(item)
    
    # Process HTML files
    html_files = [f for f in os.listdir(html_folder) if f.endswith('_response.html')]
    added_count = 0
    
    for html_file in html_files:
        poem_id = get_poem_id_from_filename(html_file)
        if poem_id is None:
            print(f"Could not extract poem ID from filename: {html_file}")
            continue
        
        if poem_id not in poem_groups:
            print(f"Poem ID {poem_id} not found in diwan data")
            continue
        
        # Read HTML file
        html_path = os.path.join(html_folder, html_file)
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            print(f"Error reading {html_file}: {e}")
            continue
        
        # Extract content
        content = extract_content_from_html(html_content)
        
        if not content['response']:
            print(f"No response content found in {html_file}")
            # Debug: let's see what's in the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            response_div = soup.find('div', class_='response')
            if response_div:
                print(f"Found response div, looking for markdown content...")
                markdown_divs = response_div.find_all('div', class_='markdown')
                print(f"Found {len(markdown_divs)} markdown divs in response")
                for i, md in enumerate(markdown_divs):
                    parent_thinking = md.find_parent('div', class_='thinking')
                    parent_thoughts = md.find_parent('model-thoughts')
                    print(f"Markdown div {i}: thinking_parent={parent_thinking is not None}, thoughts_parent={parent_thoughts is not None}")
                    print(f"  Content: {md.get_text(strip=True)[:100]}...")
            continue
        
        # Debug: show what we extracted
        print(f"Extracted from {html_file}:")
        print(f"Response length: {len(content['response'])} chars")
        print(f"Response preview: {content['response'][:200]}...")
        print(f"Thinking length: {len(content['thinking'])} chars")
        print(f"Thinking preview: {content['thinking'][:200]}...")
        print("-" * 50)
        
        # Add Gemini response to the first entry for this poem_id
        poem_entries = poem_groups[poem_id]
        if poem_entries:
            # Check if there's already a Gemini response in the original data
            has_gemini = False
            for entry in poem_entries:
                ai_response = entry.get('ai', {})
                model = ai_response.get('model', '').lower()
                if 'gemini' in model:
                    has_gemini = True
                    break
            
            if not has_gemini:
                # Copy the first entry and modify only the ai field
                original_entry = poem_entries[0]
                new_entry = original_entry.copy()
                new_entry['ai'] = {
                    'provider': 'gemini',
                    'model': 'gemini-2.5-pro',
                    'text': content['response'],
                    'thoughtMap': content['thinking']
                }
                diwan_data.append(new_entry)
                added_count += 1
                print(f"Added Gemini response for poem {poem_id}")
            else:
                print(f"Poem {poem_id} already has Gemini response in original data, skipping")
    
    print(f"Total Gemini responses added: {added_count}")
    return diwan_data

def main():
    """Main function to parse and merge data."""
    # File paths
    diwan_file = "../../labeler/diwan.json"
    html_folder = "./html"  # Current directory where HTML files are saved
    output_file = "diwan_with_gemini.json"
    
    try:
        # Load original data
        print("Loading original diwan data...")
        diwan_data = load_diwan_data(diwan_file)
        print(f"Loaded {len(diwan_data)} entries from diwan.json")
        
        # Merge Gemini responses
        print("Processing HTML files and merging responses...")
        updated_data = merge_gemini_responses(diwan_data, html_folder)
        
        # Save updated data
        print(f"Saving updated data to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully saved {len(updated_data)} entries to {output_file}")
        
        # Print summary
        original_count = len(diwan_data)
        new_count = len(updated_data)
        added_count = new_count - original_count
        print(f"Summary: {added_count} new Gemini responses added to the dataset")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
