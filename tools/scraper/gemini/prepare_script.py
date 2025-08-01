import json
import random
from typing import List, Dict, Any

def load_and_process_data(file_path: str, n_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Load JSON file, sort by poem_id, find poems with Claude but no Gemini responses,
    and return random n samples with prepared prompts.
    
    Args:
        file_path: Path to the JSON file
        n_samples: Number of random samples to return
    
    Returns:
        List of dictionaries with prepared prompts
    """
    # Load JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Sort by poem_id
    data.sort(key=lambda x: x.get('poem_id', 0))
    
    # Group by poem_id
    poem_groups = {}
    for item in data:
        poem_id = item.get('poem_id')
        if poem_id not in poem_groups:
            poem_groups[poem_id] = []
        poem_groups[poem_id].append(item)
    
    # Find poems with Claude but no Gemini responses
    target_poems = []
    for poem_id, items in poem_groups.items():
        has_claude = False
        has_gemini = False
        
        for item in items:
            ai_response = item.get('ai', {})
            provider = ai_response.get('provider', '').lower()
            model = ai_response.get('model', '').lower()
            
            if 'claude-sonnet-4-20250514' in model:
                has_claude = True
            if 'gemini-2.5-pro' in model:
                has_gemini = True
        
        if has_claude and not has_gemini:
            # Get the item with Claude response
            for item in items:
                ai_response = item.get('ai', {})
                model = ai_response.get('model', '')
                if 'claude-sonnet-4-20250514' in model:
                    target_poems.append(item)
                    break
    
    # Randomly sample n items
    if len(target_poems) < n_samples:
        print(f"Warning: Only {len(target_poems)} poems found, returning all")
        n_samples = len(target_poems)
    
    # sample first n_samples
    selected_poems = target_poems[:n_samples]
    
    # Prepare prompts
    prepared_data = []
    for poem in selected_poems:
        reference = poem.get('reference', {})
        prompt_data = poem.get('prompt', {})
        
        prepared_item = {
            'poem_id': poem.get('poem_id'),
            'prompt_text': prompt_data.get('text', '')
        }
        prepared_data.append(prepared_item)
    
    return prepared_data

def save_to_js_file(data: List[Dict[str, Any]], output_file: str):
    """
    Save the prepared data as a JavaScript array.
    
    Args:
        data: List of prepared data dictionaries
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('const poemsForGemini = ')
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write(';\n\n')
        f.write('export default poemsForGemini;\n')

def create_scraper_js(data: List[Dict[str, Any]], output_file: str):
    """
    Create a scraper.js file that contains a list of prompts and places first prompt in textarea.
    
    Args:
        data: List of prepared data dictionaries
        output_file: Output file path
    """
    if not data:
        print("No data to create scraper for")
        return
    
    # Create the prompts list
    prompts_list = []
    for item in data:
        # Add instruction to output only the poem without introductory text
        enhanced_prompt = item['prompt_text'] + "\n\nمهم: اكتب القصيدة مباشرة بدون أي مقدمة أو كلمات تمهيدية مثل 'بالتأكيد' أو 'إليك' أو 'هنا'. ابدأ القصيدة فوراً."
        
        prompts_list.append({
            "poem_id": item['poem_id'],
            "prompt": enhanced_prompt
        })
    
    scraper_script = f'''// Gemini Scraper Script
// This script finds the edit button, places the first prompt in the textarea,
// waits for generation, clicks "Show thinking", and downloads the page

(function() {{
    'use strict';
    
    // List of prompts to process
    const promptsList = {json.dumps(prompts_list, ensure_ascii=False, indent=2)};
    
    // Wait for page to load
    function waitForElement(selector, timeout = 10000) {{
        return new Promise((resolve, reject) => {{
            const startTime = Date.now();
            
            const checkElement = () => {{
                const element = document.querySelector(selector);
                if (element) {{
                    resolve(element);
                }} else if (Date.now() - startTime > timeout) {{
                    reject(new Error(`Element ${{selector}} not found within ${{timeout}}ms`));
                }} else {{
                    setTimeout(checkElement, 100);
                }}
            }};
            
            checkElement();
        }});
    }}
    
    // Wait for generation to complete
    function waitForGenerationComplete(timeout = 60000) {{
        return new Promise((resolve, reject) => {{
            const startTime = Date.now();
            
            const checkGeneration = () => {{
                // Look for the avatar animation with completed status
                const completedAvatar = document.querySelector('[data-test-lottie-animation-status="completed"]');
                
                if (completedAvatar) {{
                    console.log('Found completed avatar animation, waiting to ensure it is stable...');
                    // Wait a bit to ensure the completed status is stable
                    setTimeout(() => {{
                        const stillCompleted = document.querySelector('[data-test-lottie-animation-status="completed"]');
                        if (stillCompleted) {{
                            console.log('Completed status confirmed stable');
                            resolve();
                        }} else {{
                            console.log('Completed status disappeared, continuing to wait...');
                            setTimeout(checkGeneration, 1000);
                        }}
                    }}, 2000);
                    return;
                }}
                
                if (Date.now() - startTime > timeout) {{
                    reject(new Error('Generation timeout'));
                }} else {{
                    setTimeout(checkGeneration, 1000);
                }}
            }};
            
            checkGeneration();
        }});
    }}
    
    // Wait for thinking to be shown
    function waitForThinkingShown(timeout = 30000) {{
        return new Promise((resolve, reject) => {{
            const startTime = Date.now();
            
            const checkThinking = () => {{
                // Look for the specific thoughts content element
                const thinkingContent = document.querySelector('[data-test-id="thoughts-content"]');
                
                if (thinkingContent && thinkingContent.textContent.trim().length > 0) {{
                    console.log('Found thinking content:', thinkingContent.textContent.substring(0, 100) + '...');
                    resolve();
                    return;
                }}
                
                if (Date.now() - startTime > timeout) {{
                    reject(new Error('Thinking display timeout'));
                }} else {{
                    setTimeout(checkThinking, 1000);
                }}
            }};
            
            checkThinking();
        }});
    }}
    
    // Extract and download response and thinking content
    function downloadResponseAndThinking(poemId) {{
        // Find the response content
        const responseContent = document.querySelector('.response-content');
        if (!responseContent) {{
            console.error('Response content not found');
            return;
        }}
        
        // Find the thinking content (thoughts)
        const thinkingContent = document.querySelector('[data-test-id="thoughts-content"]');
        
        // Create a simple HTML structure with just the content we want
        let htmlContent = `<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>Poem ${{poemId}} - Response and Thinking</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .response {{ margin-bottom: 30px; }}
        .thinking {{ margin-top: 30px; }}
        .section-title {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
    </style>
</head>
<body>
    <div class="response">
        <div class="section-title">Response:</div>
        ${{responseContent.innerHTML}}
    </div>`;
        
        if (thinkingContent) {{
            htmlContent += `
    <div class="thinking">
        <div class="section-title">Thinking Process:</div>
        ${{thinkingContent.innerHTML}}
    </div>`;
        }}
        
        htmlContent += `
</body>
</html>`;
        
        const blob = new Blob([htmlContent], {{ type: 'text/html' }});
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `poem_${{poemId}}_response.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`Response and thinking downloaded as poem_${{poemId}}_response.html`);
    }}
    
    // Process a single poem
    async function processPoem(poem) {{
        try {{
            console.log('Processing poem ID:', poem.poem_id);
            
            // First, find and click the edit button
            const editButton = await waitForElement('button[data-test-id="prompt-edit-button"], button[aria-label="Edit"], button[data-test-id="edit-button"]');
            console.log('Found edit button:', editButton);
            editButton.click();
            console.log('Edit button clicked');
            
            // Wait a bit for the edit mode to activate
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Now find the editable textarea
            const textarea = await waitForElement('textarea[aria-label="Edit prompt"], textarea.mat-mdc-input-element');
            console.log('Found editable textarea:', textarea);
            
            // Set the prompt text
            const promptText = poem.prompt;
            
            // Clear existing content and set new prompt
            textarea.value = promptText;
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.dispatchEvent(new Event('change', {{ bubbles: true }}));
            
            console.log('Prompt text set successfully');
            
            // Wait a bit for the update button to become enabled
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Find and click the update button
            const updateButton = await waitForElement('button.update-button, button[class*="update-button"]');
            console.log('Found update button:', updateButton);
            
            // Check if button is disabled
            if (!updateButton.disabled) {{
                updateButton.click();
                console.log('Update button clicked');
            }} else {{
                console.log('Update button is disabled, waiting...');
                // Wait a bit more and try again
                setTimeout(async () => {{
                    try {{
                        const updateBtn = document.querySelector('button.update-button, button[class*="update-button"]');
                        if (updateBtn && !updateBtn.disabled) {{
                            updateBtn.click();
                            console.log('Update button clicked after waiting');
                        }} else {{
                            console.log('Update button still disabled');
                        }}
                    }} catch (error) {{
                        console.log('Error clicking update button:', error.message);
                    }}
                }}, 2000);
            }}
            
            // Wait for generation to complete
            console.log('Waiting for generation to complete...');
            await waitForGenerationComplete();
            console.log('Generation completed');
            
            // Find and click the "Show thinking" button
            console.log('Looking for "Show thinking" button...');
            const showThinkingButton = await waitForElement('button[class*="thoughts-header-button"], div[class*="thoughts-header-button"], [class*="thoughts-header-button"]');
            console.log('Found "Show thinking" button:', showThinkingButton);
            showThinkingButton.click();
            console.log('"Show thinking" button clicked');
            
            // Wait a bit for the button click to take effect
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Wait for thinking to be shown
            console.log('Waiting for thinking to be displayed...');
            await waitForThinkingShown();
            console.log('Thinking displayed');
            
            // Wait a bit more for thinking to fully load
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Download the response and thinking content
            console.log('Downloading response and thinking content...');
            downloadResponseAndThinking(poem.poem_id);
            
            console.log('Completed processing poem ID:', poem.poem_id);
            
        }} catch (error) {{
            console.error('Error processing poem ID', poem.poem_id, ':', error);
        }}
    }}
    
    // Main scraping function
    async function scrapeGemini() {{
        try {{
            console.log('Starting Gemini scraper...');
            console.log('Total prompts to process:', promptsList.length);
            
            // Process each poem in sequence
            for (let i = 0; i < promptsList.length; i++) {{
                const poem = promptsList[i];
                console.log(`Processing poem ${{i + 1}} of ${{promptsList.length}}: ID ${{poem.poem_id}}`);
                
                await processPoem(poem);
                
                // Wait a bit between poems (except for the last one)
                if (i < promptsList.length - 1) {{
                    console.log('Waiting 3 seconds before processing next poem...');
                    await new Promise(resolve => setTimeout(resolve, 3000));
                }}
            }}
            
            console.log('All poems processed successfully!');
            
        }} catch (error) {{
            console.error('Error in scraper:', error);
        }}
    }}
    
    // Run the scraper
    scrapeGemini();
}})();
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(scraper_script)
    
    print(f"Scraper script created: {output_file}")

def get_prompts_only(data: List[Dict[str, Any]]) -> List[str]:
    """
    Extract only the prompt texts from the data.
    
    Args:
        data: List of prepared data dictionaries
    
    Returns:
        List of prompt strings
    """
    return [item['prompt_text'] for item in data]

if __name__ == "__main__":
    # Example usage
    input_file = "./diwan.json"
    output_file = "gemini_prompts.js"
    n_samples = 200
    
    try:
        prepared_data = load_and_process_data(input_file, n_samples)
        save_to_js_file(prepared_data, output_file)
        create_scraper_js(prepared_data, "scraper.js")
        
        print(f"Successfully processed {len(prepared_data)} poems")
        print(f"Data saved to {output_file}")
        print("Scraper script created: scraper.js")
        
        # Print sample of prepared data
        if prepared_data:
            print("\nSample prepared data:")
            print(json.dumps(prepared_data[0], ensure_ascii=False, indent=2))
            
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except Exception as e:
        print(f"Error: {e}")
