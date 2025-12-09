"""Calculate normalized and unnormalized perplexity for a sentence.

Given two entities and a sentence containing them, this script calculates:
1. Perplexity for each entity (isolated, neutral, typed contexts)
2. Perplexity for the full sentence
3. Normalized perplexity (sentence / entity product)
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_perplexity(text: str, model, tokenizer, device: str) -> tuple[float, float]:
    """Calculate perplexity for a single text.

    Args:
        text: Input text
        model: Language model
        tokenizer: Tokenizer
        device: Device (cpu, cuda, mps)

    Returns:
        Tuple of (perplexity, loss)
    """
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity, loss.item()


def calculate_entity_perplexity(entity: str, model, tokenizer, device: str) -> float:
    """Calculate perplexity for an entity (isolated context).

    Args:
        entity: Entity name
        model: Language model
        tokenizer: Tokenizer
        device: Device

    Returns:
        Entity perplexity
    """
    perplexity, _ = calculate_perplexity(entity, model, tokenizer, device)
    return perplexity


def main():
    """Calculate and display perplexities."""
    parser = argparse.ArgumentParser(
        description="Calculate normalized perplexity for a sentence with two entities"
    )
    parser.add_argument(
        "entity1",
        type=str,
        help="First entity name"
    )
    parser.add_argument(
        "entity2",
        type=str,
        help="Second entity name"
    )
    parser.add_argument(
        "sentence",
        type=str,
        help="Sentence containing both entities"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-xl",
        help="Model to use (default: gpt2-xl)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use (cpu, cuda, mps) (default: auto-detect)"
    )

    args = parser.parse_args()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(args.device)
    if args.device == "mps":
        # Convert to float32 for MPS compatibility
        model = model.to(dtype=torch.float32)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Calculate entity perplexities
    entity1_perp = calculate_entity_perplexity(args.entity1, model, tokenizer, args.device)
    entity2_perp = calculate_entity_perplexity(args.entity2, model, tokenizer, args.device)

    # Calculate sentence perplexity
    sentence_perp, sentence_loss = calculate_perplexity(
        args.sentence, model, tokenizer, args.device
    )

    # Calculate product and normalized perplexity
    entity_product = entity1_perp * entity2_perp
    normalized = sentence_perp / entity_product

    # Display results
    print(f"\nSentence: \"{args.sentence}\"")
    print(f"  Entity 1 ({args.entity1}):     {entity1_perp:>12,.2f}")
    print(f"  Entity 2 ({args.entity2}):     {entity2_perp:>12,.2f}")
    print(f"  Entity product:                {entity_product:>15,.2f}")
    print(f"  Sentence perplexity:           {sentence_perp:>12,.2f}")
    print(f"  Normalized (sent/product):     {normalized:>15.9f}")


if __name__ == "__main__":
    main()
