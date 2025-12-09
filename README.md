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

### Basic Usage

Run the pipeline with default settings (biolink:treats predicate):

```bash
uv run python -m src.main
```

### Advanced Usage

```bash
uv run python -m src.main \
  --predicate "biolink:treats" \
  --model "BioMistral/BioMistral-7B" \
  --device cuda \
  --batch-size 8 \
  --output-dir outputs/my_experiment
```

### Options

- `--predicate`: Predicate to analyze (default: `biolink:treats`)
- `--model`: HuggingFace model name (default: `BioMistral/BioMistral-7B`)
- `--num-samples`: Limit number of edges to process (default: all)
- `--output-dir`: Output directory (default: `outputs/<predicate>_<timestamp>`)
- `--device`: Device to use - `cuda` (NVIDIA), `mps` (Apple Silicon), or `cpu` (default: auto-detect)
- `--batch-size`: Batch size for perplexity calculation (default: 8)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--nodes-file`: Path to nodes file (default: input data path)
- `--edges-file`: Path to edges file (default: input data path)

### Example: Testing with a Small Model

For quick testing with a smaller model:

```bash
uv run python -m src.main \
  --model "gpt2" \
  --device cpu \
  --num-samples 10 \
  --output-dir outputs/test_run
```

## Output

The pipeline generates:

1. **results.tsv**: Raw data with all statements and perplexity scores (tab-separated)
2. **descriptive_stats.json**: Descriptive statistics by category (mean, median, std, etc.)
3. **statistical_tests.json**: Statistical test results (ANOVA, t-tests, effect sizes)
4. **boxplot.png**: Box plot comparison of categories
5. **violinplot.png**: Violin plot comparison
6. **histogram.png**: Overlaid histogram comparison

## Testing

Run tests:

```bash
uv run pytest tests/ -v
```

Run tests with coverage:

```bash
uv run pytest tests/ -v --cov=src --cov-report=html
```

## Project Structure

```
perplexme/
├── src/
│   ├── config.py           # Configuration and constants
│   ├── data_loader.py      # Load KGX format data
│   ├── sentence_generator.py  # Generate statements
│   ├── perplexity.py       # Calculate perplexity
│   ├── analysis.py         # Statistical analysis
│   └── main.py             # Main pipeline
├── tests/
│   ├── fixtures/           # Test data
│   └── test_*.py           # Test files
├── pyproject.toml
└── README.md
```

## Data Format

Input data should be in KGX format:
- **nodes.jsonl**: One JSON object per line with `id`, `name`, and `category`
- **edges.jsonl**: One JSON object per line with `subject`, `predicate`, and `object`

See `tests/fixtures/sample_data/` for examples.

## Models

The default model is `BioMistral/BioMistral-7B` (base model pretrained on PubMed, ideal for biomedical perplexity), but you can use any HuggingFace causal language model:
- `BioMistral/BioMistral-7B` (recommended: biomedical base model, 7B params)
- `mistralai/Mistral-7B-v0.1` (general purpose base model, 7B params)
- `gpt2` (small, fast, for testing)
- `meta-llama/Llama-2-7b-hf` (may be gated)
- `google/medgemma-27b-text-it` (medical domain, instruction-tuned, may be gated)

**Note:** For perplexity calculation, base/pretrained models are preferred over instruction-tuned (`-it`) models. BioMistral is specifically trained on biomedical text (PubMed Central) and is fully open without access restrictions.

## Development

- Don't use mocks in tests
- Maintain high code coverage
- Use `uv run` for all commands (never install to system)
- Follow the rules in CLAUDE.md
