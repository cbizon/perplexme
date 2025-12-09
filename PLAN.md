# PerplexMe Implementation Plan

## Overview
Build a system to test whether perplexity correlates with truth by comparing perplexity scores across:
1. **True statements**: Real edges from ROBOKOP (e.g., "Metformin treats type 2 diabetes")
2. **False-permuted statements**: Permuting subjects/objects that appear in real edges of same predicate (not checking types)
3. **False-random statements**: Random subject/object pairs NOT in any valid edge of that predicate
4. **Nonsense statements**: Incorrect entity type combinations (e.g., "skin treats aspirin")

## Implementation Steps

### 1. Project Setup
- Initialize uv-based Python project with pyproject.toml
- Create src/ and tests/ directories
- Set up dependencies:
  - torch
  - transformers (for HuggingFace models)
  - pytest, pytest-cov
  - pandas (for data handling)
  - matplotlib, seaborn (for visualizations)
  - scipy (for statistical tests)
  - tqdm (for progress bars)

### 2. Data Loading Module (`src/data_loader.py`)
- Load KGX nodes from JSONL
  - Parse node ID, name, category (first element)
  - Create lookup: ID -> {name, category}
- Load KGX edges from JSONL
  - Parse subject, predicate, object
  - Handle qualifiers (for future use, not for "treats")
  - Filter by predicate (e.g., "biolink:treats")
  - Create index structures:
    - edges by predicate
    - all subjects that appear in edges of this predicate
    - all objects that appear in edges of this predicate
    - set of valid (subject, object) pairs per predicate
- Tests: validate parsing, filtering, indexing

### 3. Sentence Generation Module (`src/sentence_generator.py`)
- Template-based generation:
  - Extract predicate label (remove "biolink:", replace "_" with " ")
  - Generate: "{subject_name} {predicate_label} {object_name}"
  - Example: "Metformin treats type 2 diabetes"
- Generate true statements:
  - Iterate through filtered edges
  - Create sentence for each edge
- Generate false-permuted statements:
  - For each true edge, randomly select different object from the pool of objects that appear in edges with this predicate
  - Ensure (subject, new_object) is NOT a valid edge
  - Don't worry about type compatibility
- Generate false-random statements:
  - For each true edge, randomly select subject and object from appropriate categories
  - Ensure pair is NOT in any valid edge of that predicate
  - Can sample from all nodes, not just those in edges
- Generate nonsense statements:
  - For each true edge, swap subject/object categories
  - Randomly select entities with wrong types
  - Example: pick SmallMolecule for object, Disease for subject (reversed)
- Return: List of (statement_text, category, edge_id) tuples
- Tests: validate sentence format, category correctness, no overlap between true/false

### 4. Perplexity Calculation Module (`src/perplexity.py`)
- Load model and tokenizer (parameterizable):
  - Default: "BioMistral/BioMistral-7B" (biomedical base model, recommended for perplexity)
  - Support any HuggingFace causal LM
- Calculate perplexity for a single statement:
  - Tokenize statement
  - Get model outputs with labels
  - Extract cross-entropy loss
  - Compute perplexity: exp(loss)
- Batch processing:
  - Process statements in batches for efficiency
  - Handle different sequence lengths (padding)
  - Progress bar for long runs
- Return: List of (statement_text, category, perplexity_score)
- Tests: validate perplexity calculation, batch processing correctness

### 5. Statistical Analysis Module (`src/analysis.py`)
- Descriptive statistics:
  - Group by category (true, false-permuted, false-random, nonsense)
  - Calculate mean, median, std, min, max for each group
  - Generate summary table
- Statistical tests:
  - Pairwise t-tests between groups
  - ANOVA/Kruskal-Wallis for overall difference
  - Effect sizes (Cohen's d)
- Visualizations:
  - Box plots comparing groups
  - Histograms with overlaid distributions
  - Violin plots
  - Save to output directory
- Export results:
  - CSV with all statements and perplexity scores
  - JSON with statistical summary
  - PNG/PDF plots
- Tests: validate statistical calculations, ensure plots are generated

### 6. Main Pipeline (`src/main.py`)
- CLI interface with arguments:
  - `--predicate`: Which predicate to analyze (default: "biolink:treats")
  - `--model`: Model name (default: "BioMistral/BioMistral-7B")
  - `--num-samples`: Number of samples (default: all edges, or limit)
  - `--output-dir`: Where to save results
  - `--device`: CPU or CUDA
  - `--batch-size`: For perplexity calculation
- Pipeline:
  1. Load nodes and edges
  2. Filter by predicate
  3. Generate sentences (4 categories)
  4. Calculate perplexity scores
  5. Perform statistical analysis
  6. Save results
- Progress reporting at each stage
- Tests: integration test with small sample

### 7. Configuration & Utilities
- `src/config.py`: Constants and paths
  - Input data path
  - Default model settings
  - Template strings
- `src/utils.py`: Helper functions
  - Random sampling utilities
  - Category matching functions
  - File I/O helpers

## Project Structure
```
perplexme/
├── pyproject.toml
├── CLAUDE.md
├── kgx.md
├── PLAN.md
├── README.md (usage instructions)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── sentence_generator.py
│   ├── perplexity.py
│   ├── analysis.py
│   ├── utils.py
│   └── main.py
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_sentence_generator.py
    ├── test_perplexity.py
    ├── test_analysis.py
    └── fixtures/
        └── sample_data/ (small test files)
```

## Key Design Decisions

1. **No caching initially**: Keep it simple, run fresh each time. Add caching later if needed.

2. **Single predicate focus**: Start with "biolink:treats", make easy to extend.

3. **Four statement categories**:
   - True (real edges)
   - False-permuted (swap subjects/objects from real edge pool, ignore types)
   - False-random (random pairs not in edges, same types)
   - Nonsense (wrong entity types)

4. **Equal sample sizes**: 1:1:1:1 ratio for statistical validity.

5. **HuggingFace transformers**: Direct control over model, good for research.

6. **Parameterizable model**: Easy to swap models for comparison.

7. **Simple templates now**: "{subject} {predicate} {object}", extensible for qualifiers later.

8. **Comprehensive testing**: Test each module independently, no mocks.

9. **Statistical rigor**: Multiple tests, visualizations, effect sizes.

10. **Clean data handling**: Stream large files, don't load everything into memory at once.

## Testing Strategy

- Unit tests for each module
- Integration test for full pipeline with small dataset
- Test with sample data (create fixtures)
- Validate statistical calculations against known values
- No mocks - test real functionality

## Extensibility Points

- Template system can be extended for qualifiers
- Model loading is parameterized
- Predicate filtering is configurable
- Sample generation strategies are modular
- Analysis can be extended with more tests/plots

## Risks & Mitigations

1. **Large data files**: Stream processing, limit samples for testing
2. **Model memory**: Support batching, device selection (CPU/GPU)
3. **Long runtimes**: Progress bars, ability to limit samples
4. **False sample collision**: Verify generated false samples aren't accidentally true
5. **Category matching**: Careful handling of hierarchical biolink categories

## Next Steps

1. Set up project structure and dependencies
2. Implement data loading with tests
3. Implement sentence generation with tests
4. Implement perplexity calculation with tests
5. Implement analysis with tests
6. Build main pipeline
7. Run on full "treats" dataset
8. Analyze results
