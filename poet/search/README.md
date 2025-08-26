# Best-Of-N Search Implementation

This module implements Best-Of-N search strategy for the poet pipeline. It allows you to run any node (generation, evaluation, refinement) multiple times with different parameters and select the best result using LLM-based selection.

## Overview

The Best-Of-N search works by:
1. **Generating multiple candidates** by running the underlying node with different parameters (e.g., temperature)
2. **Selecting the best candidate** using LLM-based selection with task-specific prompts
3. **Returning the best result** with selection metadata

## Architecture

### BestOfNNode
- **Wrapper node** that applies Best-Of-N search to any underlying node
- **No modification** of existing nodes required
- **Configurable** search parameters (n_candidates, selection_prompt, etc.)

### Selection Prompts
- **Task-specific prompts** for different selection criteria:
  - `generation_selection`: Select best generated poem
  - `evaluation_selection`: Select most consistent evaluation
  - `refinement_selection`: Select best refinement path
- **Bilingual support**: Arabic and English prompts available

## Usage

### Configuration

Add Best-Of-N nodes to your pipeline configuration:

```yaml
pipeline:
  - constraints_parser
  - qafiya_selector
  - bahr_selector
  - data_enrichment:
      sources:
        - type: corpus
          local_knowledge_path: "dataset"
          top_k: 20
          search_criteria:
            - meter
            - qafiya
  - best_of_n_generation:
      n_candidates: 5
      selection_prompt: "generation_selection"
      selection_metric: "overall_score"
      temperature_range: [0.5, 0.7, 0.9, 1.1, 1.3]
  - best_of_n_evaluation:
      metrics:
        - "prosody"
        - "qafiya"
      n_candidates: 3
      selection_prompt: "evaluation_selection"
      selection_metric: "consistency_score"
  - best_of_n_refiner_chain:
      max_iterations: 3
      target_quality: 1
      refiners:
        - prosody_refiner
        - qafiya_refiner
      n_candidates: 4
      selection_prompt: "refinement_selection"
      selection_metric: "final_quality_score"
```

### Parameters

#### Common Parameters
- `n_candidates`: Number of candidates to generate (default: 5)
- `selection_prompt`: Prompt template to use for selection
- `selection_metric`: Metric to optimize for (e.g., "overall_score", "consistency_score")
- `temperature_range`: List of temperatures for diversity (default: [0.5, 0.7, 0.9, 1.1, 1.3])

#### Node-Specific Parameters
- **Generation**: Uses underlying `SimplePoemGenerator` parameters
- **Evaluation**: Uses underlying `PoemEvaluator` parameters (metrics, etc.)
- **Refinement**: Uses underlying `RefinerChain` parameters (max_iterations, refiners, etc.)

## Selection Prompts

### Generation Selection
Selects the best generated poem based on:
- Adherence to meter and rhyme
- Language quality and style
- Meaning and thematic coherence
- Creativity and aesthetic appeal

### Evaluation Selection
Selects the most consistent evaluation based on:
- Consistency in evaluation results
- Accuracy of analysis
- Comprehensiveness of coverage
- Clarity of recommendations

### Refinement Selection
Selects the best refinement path based on:
- Improvement in final quality
- Efficiency of refinement process
- Stability of results
- Comprehensiveness of improvements

## Output

The BestOfNNode returns the best candidate with additional metadata:
- `selection_metadata`: LLM selection reasoning and scores
- `selected_index`: Index of the selected candidate
- `_candidate_index`: Original candidate index
- `_candidate_temperature`: Temperature used for this candidate

## Example Output

```json
{
  "poem": {
    "verses": ["قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ", "..."],
    "quality": {...}
  },
  "selection_metadata": {
    "selected_candidate": 2,
    "reasoning": "This candidate shows the best adherence to meter...",
    "quality_scores": {
      "candidate_0": 0.85,
      "candidate_1": 0.72,
      "candidate_2": 0.91
    }
  },
  "selected_index": 2,
  "_candidate_index": 2,
  "_candidate_temperature": 0.9
}
```

## Testing

Run the unit tests:
```bash
pytest tests/unit/test_best_of_n_node.py
```

## Integration

The BestOfN nodes are automatically registered in the pipeline and work with:
- **Harmony capture**: Selection reasoning is captured
- **Existing evaluation**: Uses existing evaluation pipeline
- **Existing prompts**: Leverages existing prompt system
- **Pipeline validation**: Follows existing validation patterns
