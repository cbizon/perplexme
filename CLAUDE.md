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

src/
tests/

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

