# Integration Tests

This directory contains integration tests that test the system with real LLM providers.

## Setup

### 1. Install Dependencies

Make sure you have the required LLM provider packages installed:

```bash
# For OpenAI
pip install openai
```

### 2. Configure LLM Credentials

Copy the example configuration and add your API keys:

```bash
cp tests/fixtures/llms_example.json tests/fixtures/llms.json
```

Edit `tests/fixtures/llms.json` and replace placeholder values with your actual API keys:

```json
{
    "openai": {
        "name": "openai",
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-your-actual-openai-key-here",
        "api_base": "https://api.openai.com/v1",
        "api_version": "2023-05-19"
    }
}
```

**⚠️ Important**: Never commit real API keys to version control. The `llms.json` file is gitignored.

## Running Tests

### Run All Integration Tests

```bash
# Set environment variable and run integration tests
TEST_REAL_LLMS=1 python -m pytest tests/integration/ -v
```

### Run Only Fast Tests (Skip Slow Performance Tests)

```bash
TEST_REAL_LLMS=1 python -m pytest tests/integration/ -v -m "not slow"
```

### Run Specific Test Classes

```bash
# Test only OpenAI adapter
TEST_REAL_LLMS=1 python -m pytest tests/integration/test_openai.py::TestOpenAIAdapter -v

# Test constraint parser with real LLM (parametrized tests)
TEST_REAL_LLMS=1 python -m pytest tests/unit/test_constraint_parser.py -k "parametrized" -v
```

## Test Categories

### TestOpenAIAdapter (tests/integration/test_openai.py)
- Basic text generation
- Generation with metadata
- Availability checking
- Model information retrieval

### TestOpenAIErrorHandling (tests/integration/test_openai.py)
- Invalid API key handling
- Timeout handling
- Connection error handling

### TestOpenAIPerformance (tests/integration/test_openai.py, marked as `@pytest.mark.slow`)
- Response time testing
- Multiple request handling
- Performance benchmarks

### Parametrized Constraint Parser Tests (tests/unit/test_constraint_parser.py)
- Example 1 and Example 2 constraint parsing
- Runs with both mock and real LLMs automatically
- Real LLM constraint extraction validation

## Environment Variables

- `TEST_REAL_LLMS`: Set to any non-empty value to enable real LLM tests
- `REAL_LLM_PROVIDER`: Specify which provider to use (default: "openai")
- `OPENAI_API_KEY`: Alternative way to set OpenAI API key (overrides config file)

## Troubleshooting

### Tests are Skipped

If tests are being skipped, check:

1. `TEST_REAL_LLMS` environment variable is set
2. `llms.json` file exists and has valid configuration
3. API keys are not placeholder values (don't start with "sk-" for the example)
4. Required packages are installed (`pip install openai`)

### API Errors

- **Authentication errors**: Check your API key is valid and has sufficient permissions
- **Rate limit errors**: Wait a moment and try again, or use a different API key tier
- **Timeout errors**: Check your internet connection and try increasing timeout values

### Cost Considerations

These tests make real API calls which may incur costs:
- Basic tests use minimal tokens (usually < $0.01 per run)
- Performance tests may use more tokens
- Monitor your API usage if running frequently

## Adding New LLM Providers

To add support for a new LLM provider:

1. Create the adapter class in `poet/llm/your_provider_adapter.py`
2. Add configuration to `llms.json`
3. Update `llm_factory.py` to support the new provider
4. Add test fixtures and test classes in `test_your_provider.py`
5. Update this README with setup instructions 