# PerplexMe

## Goal

Determine whether perplexity correlates with truth.

## Basic Setup

* uv: we are using uv for package and environment management and an isolated environment
* tests: we are using pytest, and want to maintain high code coverage

### Environment Management - CRITICAL
**NEVER EVER INSTALL ANYTHING INTO SYSTEM LIBRARIES OR ANACONDA BASE ENVIRONMENT**
- ALWAYS use the isolated virtual environment at `.venv/`
- ALWAYS use `uv run` to execute commands, which automatically uses the isolated environment
- The virtual environment is sacred. System packages are not your garbage dump.

## Key Dependencies

torch

## Basic Workflow

Read nodes and edges from ROBOKOP.  Given a particular predicate (e.g. treats), generate three sets of sentences:

1. True sentences corresponding to a treats edge (Metformin treats type 2 diabetes)
2. False sentences made by permuting entities of the correct type (Albuterol treats type 2 diabetes)
3. Nonsense sentences made by permuting entities of the incorrect type (skin treats aspirin)

Generate the perplexity of these statements and compare them statistically.

## Input

The input data may never be changed. 

The input data is found in /Users/bizon/Projects/experiments/SimplePredictions/input_graphs/rbn_6f3

The nodes and edges are in the kgx format
@kgx.md

## Project structure

```
src/
├── config.py                          # Configuration constants
├── data_loader.py                     # KGX format data loading
├── sentence_generator.py              # Generate true/false/nonsense statements
├── perplexity.py                      # Perplexity calculation with MPS optimization
├── analysis.py                        # Statistical analysis and visualization
├── entity_perplexity.py               # Entity-level perplexity in 3 contexts
├── cli_common.py                      # Shared CLI utilities (argument parsing, etc.)
├── main.py                            # Main edge perplexity pipeline
├── main_entity_perplexity.py          # Entity perplexity analysis pipeline
├── calculate_sentence_perplexity.py   # Single sentence perplexity calculator
└── compare_entity_edge_perplexity.py  # Compare entity vs edge perplexity
tests/
└── test_*.py
```

## Key Ideas

Basic Approach: Perplexity
The most straightforward method uses the model's predicted probabilities for each token in a statement:
```
pythonimport torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

statement = "The cat sat on the mat"
inputs = tokenizer(statement, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    
# Lower loss = less surprising
loss = outputs.loss  # Cross-entropy loss
perplexity = torch.exp(loss)  # Common metric
```

Interpretation: Perplexity is the standard metric here - it represents roughly how many equally-likely choices the model thinks it had at each step. Lower perplexity means the statement was more predictable (less surprising).

## Implementation Details

### Statement Generation Categories

The system generates 4 types of statements from KG edges:

1. **True**: Direct from real edges (e.g., "Metformin treats type 2 diabetes")
2. **False-permuted**: Swap subjects/objects from edge pool (must be valid entities that appeared in real edges, but not as this pair)
3. **False-random**: Random entity pairs with correct types (drug + disease)
4. **Nonsense**: Wrong entity types (e.g., "skin treats aspirin")

**Critical Implementation Note**: False-permuted does NOT check entity types - it only verifies the pair doesn't exist in valid edges. This was a deliberate design decision to avoid introducing type information as a confound.

### Entity Perplexity Normalization

Entity perplexity is calculated in 3 contexts:
- **Isolated**: Just the entity name ("Metformin")
- **Neutral**: Generic context ("This is about Metformin")
- **Typed**: With type information ("The drug Metformin")

The neutral context empirically provides the best normalization baseline.

### Normalized Perplexity

Normalized perplexity = `edge_perplexity / (entity1_perplexity × entity2_perplexity)`

This normalization controls for entity familiarity, revealing whether edge perplexity differences are due to the relationship plausibility vs. entity frequency.

### MPS (Apple Silicon) Optimizations

**Critical for M-series Macs**: MPS requires consistent tensor shapes to avoid expensive recompilation.

Key optimizations in `perplexity.py`:
- Force `padding="max_length"` in tokenizer calls
- Use float32 (not float16) on MPS
- Auto-detect device and apply appropriate dtype
- Process one-at-a-time on MPS/CPU, batched on CUDA

Without these, MPS performance degrades ~80x due to constant shader recompilation.

### Code Refactoring Patterns

`cli_common.py` contains shared utilities to avoid duplication:
- `create_base_argument_parser()`: Common CLI arguments
- `load_knowledge_graph()`: KG loading with validation
- `create_perplexity_calculator()`: Model initialization
- `setup_output_directory()`: Output path handling with timestamps

This eliminates ~50 lines of duplication between main programs.

### Key Findings

**Perplexity = Surprise** (higher = more unexpected)

Empirical results show:
1. **True edges have HIGHER perplexity than false-permuted** (counterintuitive!)
   - True: ~84K, False-permuted: ~83K
   - This suggests the model hasn't learned these specific relationships

2. **Entity product correlates strongly with edge perplexity** (r=0.75)
   - Individual entities: r=0.48-0.50
   - Product of both entities: r=0.75
   - This confirms entity familiarity dominates raw perplexity

3. **Normalized perplexity provides small but significant discrimination**
   - True vs false-permuted: Cohen's d=0.06-0.13 (small effect)
   - Highly significant (p<1e-10) due to large sample size
   - But distributions overlap substantially

4. **Nonsense has LOWEST raw perplexity** (very counterintuitive!)
   - Because entities like "skin" and "aspirin" are more common/familiar
   - Demonstrates critical importance of entity normalization

## ***RULES OF THE ROAD***

- Don't use mocks. They obscure problems

- Ask clarifying questions

- Don't make classes just to group code. It is non-pythonic and hard to test.

- Do not implement bandaids - treat the root cause of problems

- Don't use try/except as a way to hide problems.  It is often good just to let something fail and figure out why.

- Once we have a test, do not delete it without explicit permission.  

- Do not return made up results if an API fails.  Let it fail.

- When changing code, don't make duplicate functions - just change the function. We can always roll back changes if needed.

- Keep the directories clean, don't leave a bunch of junk laying around.

- When making pull requests, NEVER ever mention a `co-authored-by` or similar aspects. In particular, never mention the tool used to create the commit message or PR.

- Check git status before commits

