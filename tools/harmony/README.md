# Harmony Tools

This directory contains tools for working with the Harmony system - a reasoning capture and analysis tool for Arabic poetry generation.

## Files

### `capture_fixture.py`
Captures execution data from the poetry generation pipeline and saves it as a fixture for testing.

**Usage:**
```bash
cd poet/tools/harmony
python capture_fixture.py
```

This will:
1. Run a sample poetry generation workflow
2. Capture all LLM calls, inputs, outputs, and metadata
3. Save the execution data to `tests/fixtures/harmony_test.json`

### `generate_harmony_reasoning.py`
Generates structured harmony data and converts it to conversation format using real LLMs based on captured fixture data.

**Usage:**
```bash
cd poet/tools/harmony

# Use Anthropic Claude (default)
python generate_harmony_reasoning.py

# Use OpenAI
python generate_harmony_reasoning.py --provider openai

# Customize parameters
python generate_harmony_reasoning.py --provider anthropic --max-tokens 8000 --temperature 0.5
```

**Options:**
- `--provider`: Choose between `anthropic` or `openai` (default: anthropic)
- `--max-tokens`: Maximum tokens for response (default: 4000)
- `--temperature`: Generation temperature (default: 0.7)

## Setup

### 1. API Keys
Ensure you have API keys configured in `tests/fixtures/llms.json`:

```json
{
    "openai": {
        "name": "openai",
        "model": "o3-mini-2025-01-31",
        "provider": "openai",
        "api_key": "your-openai-api-key",
        "api_base": "https://api.openai.com/v1"
    },
    "anthropic": {
        "name": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "api_key": "your-anthropic-api-key",
        "api_base": "https://api.anthropic.com/v1"
    }
}
```

### 2. Environment Variables
Set these environment variables to enable real LLM testing:

```bash
export TEST_REAL_LLMS=1
export REAL_LLM_PROVIDER=anthropic  # or openai
```

## Workflow

1. **Capture Fixture**: Run `capture_fixture.py` to create a test fixture with real execution data
2. **Generate Reasoning**: Use `generate_harmony_reasoning.py` to analyze the captured data with real LLMs
3. **Review Output**: Check the generated harmony reasoning files in the `temp/` directory

## Output Files

The tools generate several output files:

- `{execution_id}_harmony_{provider}.txt`: The generated harmony conversation
- `{execution_id}_harmony_{provider}.json`: The structured harmony data
- `{execution_id}_harmony_{provider}.metadata.json`: Metadata about the generation

## Example Output

The harmony conversation shows the complete thought process of the Arabic poetry generation system, including:

- System prompts and developer instructions
- User requests and constraints
- Step-by-step analysis and reasoning
- Function calls and tool usage
- Technical decisions about meter, rhyme, and poetic form

This provides valuable insights into how the system works and can be used for debugging, optimization, and understanding the AI's decision-making process. 