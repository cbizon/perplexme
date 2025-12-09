# PerplexMe

Determine whether perplexity correlates with truth in biomedical statements.

## Overview

PerplexMe tests whether language model perplexity can distinguish between:
1. **True statements**: Real edges from ROBOKOP knowledge graph
2. **False-permuted statements**: Permuted subjects/objects from real edges
3. **False-random statements**: Random entity pairs with correct types
4. **Nonsense statements**: Entity pairs with incorrect types

## Installation

This project uses `uv` for package management. Make sure you have `uv` installed.

```bash
# Install dependencies
uv pip install -e ".[dev]"
```

## Usage

PerplexMe provides four main tools:

### 1. Edge Perplexity Pipeline (`src.main`)

Analyze perplexity across true/false/nonsense statements from knowledge graph edges.

**Basic usage:**
```bash
uv run python -m src.main
```

**Advanced usage:**
```bash
uv run python -m src.main \
  --predicate "biolink:treats" \
  --model "gpt2-xl" \
  --device mps \
  --batch-size 8 \
  --num-samples 100 \
  --output-dir outputs/my_experiment
```

### 2. Entity Perplexity Analysis (`src.main_entity_perplexity`)

Calculate entity-level perplexity in three contexts (isolated, neutral, typed).

**Basic usage:**
```bash
uv run python -m src.main_entity_perplexity --predicate biolink:treats
```

**With custom model:**
```bash
uv run python -m src.main_entity_perplexity \
  --predicate biolink:treats \
  --model gpt2-xl \
  --output-dir outputs/entity_analysis
```

### 3. Single Sentence Calculator (`src.calculate_sentence_perplexity`)

Calculate normalized perplexity for a specific sentence with two entities.

**Usage:**
```bash
uv run python -m src.calculate_sentence_perplexity \
  "Metformin" \
  "type 2 diabetes" \
  "Metformin treats type 2 diabetes"
```

**Example output:**
```
Sentence: "Metformin treats type 2 diabetes"
  Entity 1 (Metformin):        77,974.52
  Entity 2 (type 2 diabetes):        47,848.41
  Entity product:                3,730,956,300.77
  Sentence perplexity:              90,043.38
  Normalized (sent/product):         0.000024134
```

**With different model:**
```bash
uv run python -m src.calculate_sentence_perplexity \
  "Albuterol" \
  "asthma" \
  "Albuterol treats asthma" \
  --model gpt2
```

### 4. Entity-Edge Comparison (`src.compare_entity_edge_perplexity`)

Compare entity-level and edge-level perplexities, with normalization analysis.

**Usage:**
```bash
uv run python -m src.compare_entity_edge_perplexity \
  --entity-file outputs/entity_perp_treats_YYYYMMDD_HHMMSS/entity_perplexities.tsv \
  --edge-file outputs/treats_YYYYMMDD_HHMMSS/results.tsv \
  --output-dir outputs/comparison
```

### Common Options

**All tools support:**
- `--model`: HuggingFace model name (default: `BioMistral/BioMistral-7B` for pipelines, `gpt2-xl` for calculator)
- `--device`: Device - `cuda` (NVIDIA), `mps` (Apple Silicon), or `cpu` (default: auto-detect)
- `--batch-size`: Batch size for perplexity calculation (default: 16)

**Pipeline tools (main, main_entity_perplexity):**
- `--predicate`: Predicate to analyze (default: `biolink:treats`)
- `--output-dir`: Output directory (default: auto-generated with timestamp)
- `--nodes-file`: Path to nodes file
- `--edges-file`: Path to edges file

**Edge pipeline only:**
- `--num-samples`: Limit number of edges to process (default: all)
- `--random-seed`: Random seed for reproducibility (default: 42)

## Output Files

### Edge Perplexity Pipeline (`src.main`)

Generates in `outputs/<predicate>_<timestamp>/`:
- **results.tsv**: All statements with perplexity scores
- **descriptive_stats.json**: Statistics by category (mean, median, std)
- **statistical_tests.json**: ANOVA, t-tests, Cohen's d
- **boxplot.png**, **violinplot.png**, **histogram.png**: Visualizations

### Entity Perplexity Analysis (`src.main_entity_perplexity`)

Generates in `outputs/entity_perp_<predicate>_<timestamp>/`:
- **entity_perplexities.tsv**: Entity perplexities in 3 contexts + KG degree
- **entity_context_comparison.png**: Comparison of 3 contexts
- **entity_perp_vs_degree.png**: Correlation with knowledge graph degree
- **entity_context_pairwise.png**: Pairwise context comparisons

### Comparison Analysis (`src.compare_entity_edge_perplexity`)

Generates in specified output directory:
- **entity_edge_comparison.tsv**: Individual entity datapoints
- **entity_product_comparison.tsv**: Product datapoints with normalized perplexity
- **entity_vs_edge_scatter.png**: Entity vs edge perplexity by category
- **product_vs_edge_scatter.png**: Product vs edge perplexity
- **normalized_distributions.png**: Normalized perplexity histograms (true vs false)
- **normalized_boxplots.png**, **normalized_violinplots.png**, **normalized_cdf.png**: Distribution comparisons

## Testing

Run tests:

```bash
uv run pytest tests/ -v
```

Run tests with coverage:

```bash
uv run pytest tests/ -v --cov=src --cov-report=html
```

## Key Findings

**Perplexity measures surprise** - higher values mean more unexpected/surprising.

1. **Raw perplexity doesn't distinguish true from false well**
   - True edges: ~84K perplexity
   - False-permuted: ~83K (very similar!)
   - This suggests the model hasn't learned these specific relationships

2. **Entity familiarity dominates raw perplexity**
   - Entity product correlates r=0.75 with edge perplexity
   - "Nonsense" statements have the LOWEST raw perplexity (~21K) because entities like "skin" and "aspirin" are more common

3. **Normalized perplexity helps but effect is small**
   - After dividing by entity product, true vs false differences emerge
   - Cohen's d = 0.06-0.13 (small but statistically significant)
   - Highly significant (p < 1e-10) but distributions overlap substantially

4. **Neutral context provides best normalization**
   - "This is about {entity}" works better than isolated or typed contexts
   - Captures entity familiarity without adding confounding information

## Project Structure

```
perplexme/
├── src/
│   ├── config.py                          # Configuration constants
│   ├── data_loader.py                     # Load KGX format data
│   ├── sentence_generator.py              # Generate true/false/nonsense statements
│   ├── perplexity.py                      # Perplexity calculation (MPS optimized)
│   ├── analysis.py                        # Statistical analysis
│   ├── entity_perplexity.py               # Entity-level perplexity
│   ├── cli_common.py                      # Shared CLI utilities
│   ├── main.py                            # Edge perplexity pipeline
│   ├── main_entity_perplexity.py          # Entity perplexity pipeline
│   ├── calculate_sentence_perplexity.py   # Single sentence calculator
│   └── compare_entity_edge_perplexity.py  # Comparison analysis
├── tests/
│   ├── fixtures/                          # Test data
│   └── test_*.py                          # Test files
├── pyproject.toml
├── README.md
└── CLAUDE.md                              # Implementation notes
```

## Data Format

Input data should be in KGX format:
- **nodes.jsonl**: One JSON object per line with `id`, `name`, and `category`
- **edges.jsonl**: One JSON object per line with `subject`, `predicate`, and `object`

See `tests/fixtures/sample_data/` for examples.

## Recommended Models

Any HuggingFace causal language model works. Recommended options:

**For biomedical analysis:**
- `BioMistral/BioMistral-7B` (default) - Base model trained on PubMed Central
- `gpt2-xl` (1.5B params) - Good balance of speed and quality for Apple Silicon

**For general testing:**
- `gpt2` - Fast, small (117M params), good for development
- `gpt2-medium` - 355M params
- `gpt2-large` - 774M params

**Performance notes:**
- On Apple M4: `gpt2-xl` recommended (good speed with MPS)
- On NVIDIA GPU: `BioMistral/BioMistral-7B` with batch_size=16+
- On CPU: `gpt2` or `gpt2-medium` for reasonable speed

**Important:** Use base/pretrained models, not instruction-tuned (`-it`) models. Perplexity calculation requires models trained with standard language modeling objectives.

## Development

- Don't use mocks in tests
- Maintain high code coverage
- Use `uv run` for all commands (never install to system)
- Follow the rules in CLAUDE.md
