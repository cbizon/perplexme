"""Calculate perplexity scores for statements using language models."""

import torch
from typing import List, Dict
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.sentence_generator import Statement


@dataclass
class PerplexityResult:
    """Result of perplexity calculation."""
    statement: Statement
    perplexity: float
    loss: float


class PerplexityCalculator:
    """Calculate perplexity using a language model."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        """Initialize the perplexity calculator.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cuda" or "cpu")
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Determine the best available device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32  # MPS works better with float32
        elif torch.backends.mps.is_available():
            # Auto-detect MPS if available (Apple Silicon)
            self.device = "mps"
            self.dtype = torch.float32
        elif torch.cuda.is_available():
            # Auto-detect CUDA if available
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        print(f"Loading model {model_name}...")
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully")

    def calculate_perplexity_single(self, text: str) -> tuple[float, float]:
        """Calculate perplexity for a single text.

        Args:
            text: The text to calculate perplexity for

        Returns:
            Tuple of (perplexity, loss)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"  # Force consistent length for MPS compilation
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item(), loss.item()

    def calculate_perplexity_batch(self, texts: List[str]) -> List[tuple[float, float]]:
        """Calculate perplexity for a batch of texts.

        Args:
            texts: List of texts

        Returns:
            List of (perplexity, loss) tuples
        """
        # For MPS and CPU, process individually for better performance
        # The manual per-sample loss calculation is very slow on these devices
        if self.device in ["mps", "cpu"]:
            results = []
            for text in texts:
                ppl, loss = self.calculate_perplexity_single(text)
                results.append((ppl, loss))
            return results

        # For CUDA, use true batching with manual loss calculation
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # Calculate per-sample loss
            # outputs.loss is already averaged, so we need to calculate individually
            logits = outputs.logits
            labels = inputs["input_ids"]

            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate loss per sample
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Reshape and average per sample
            losses = losses.view(shift_labels.size())

            # Only average over non-padding tokens
            attention_mask = inputs["attention_mask"][..., 1:].contiguous()
            losses = (losses * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

            perplexities = torch.exp(losses)

        results = []
        for ppl, loss in zip(perplexities.cpu().numpy(), losses.cpu().numpy()):
            results.append((float(ppl), float(loss)))

        return results

    def calculate_for_statements(
        self, statements: List[Statement]
    ) -> List[PerplexityResult]:
        """Calculate perplexity for a list of statements.

        Args:
            statements: List of Statement objects

        Returns:
            List of PerplexityResult objects
        """
        results = []
        texts = [s.text for s in statements]

        print(f"Calculating perplexity for {len(statements)} statements...")
        print(f"Batch size: {self.batch_size}")

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
            batch_texts = texts[i : i + self.batch_size]
            batch_statements = statements[i : i + self.batch_size]

            batch_results = self.calculate_perplexity_batch(batch_texts)

            for statement, (ppl, loss) in zip(batch_statements, batch_results):
                result = PerplexityResult(
                    statement=statement,
                    perplexity=ppl,
                    loss=loss,
                )
                results.append(result)

        return results

    def calculate_for_statements_dict(
        self, statements: List[Statement]
    ) -> List[Dict]:
        """Calculate perplexity and return as list of dicts (for easy DataFrame creation).

        Args:
            statements: List of Statement objects

        Returns:
            List of dictionaries with statement info and perplexity
        """
        results = self.calculate_for_statements(statements)

        dict_results = []
        for result in results:
            dict_results.append({
                "text": result.statement.text,
                "category": result.statement.category,
                "edge_id": result.statement.edge_id,
                "subject_id": result.statement.subject_id,
                "object_id": result.statement.object_id,
                "perplexity": result.perplexity,
                "loss": result.loss,
            })

        return dict_results
