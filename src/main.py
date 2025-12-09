"""Main pipeline for PerplexMe."""

from pathlib import Path

from src.cli_common import (
    create_base_argument_parser,
    setup_output_directory,
    load_knowledge_graph,
    create_perplexity_calculator,
    print_banner,
)
from src.sentence_generator import SentenceGenerator
from src.analysis import analyze_results


def main():
    """Run the PerplexMe pipeline."""
    parser = create_base_argument_parser(
        description="Determine whether perplexity correlates with truth"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of edges to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/<predicate>_<timestamp>)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = setup_output_directory(args.output_dir, args.predicate)

    print_banner("PERPLEXME PIPELINE")
    print(f"Predicate: {args.predicate}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 80)

    # Step 1: Load knowledge graph
    print("\n[1/5] Loading knowledge graph...")
    kg = load_knowledge_graph(args.nodes_file, args.edges_file, args.predicate)
    edges = kg.get_edges(args.predicate)

    # Limit number of edges if requested
    if args.num_samples is not None and args.num_samples < len(edges):
        print(f"Limiting to {args.num_samples} edges")
        # Modify the KG to only keep the first N edges
        kg.edges_by_predicate[args.predicate] = edges[:args.num_samples]

        # Update the index structures
        kg.subjects_by_predicate[args.predicate] = set()
        kg.objects_by_predicate[args.predicate] = set()
        kg.valid_pairs_by_predicate[args.predicate] = set()

        for edge in kg.edges_by_predicate[args.predicate]:
            kg.subjects_by_predicate[args.predicate].add(edge.subject)
            kg.objects_by_predicate[args.predicate].add(edge.object)
            kg.valid_pairs_by_predicate[args.predicate].add((edge.subject, edge.object))

    # Step 2: Generate sentences
    print("\n[2/5] Generating sentences...")
    generator = SentenceGenerator(kg, args.predicate, random_seed=args.random_seed)
    statements = generator.generate_all_statements()

    if len(statements) == 0:
        print("ERROR: No statements generated")
        return

    # Step 3: Calculate perplexity
    print("\n[3/4] Calculating perplexity...")
    calculator = create_perplexity_calculator(args.model, args.device, args.batch_size)
    results = calculator.calculate_for_statements_dict(statements)

    # Step 4: Analyze results
    print("\n[4/4] Analyzing results...")
    analyzer = analyze_results(results, output_dir)

    # Done
    print("\nDone!")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - results.tsv: Raw data with all statements and perplexity scores")
    print(f"  - descriptive_stats.json: Descriptive statistics by category")
    print(f"  - statistical_tests.json: Statistical test results")
    print(f"  - boxplot.png: Box plot comparison")
    print(f"  - violinplot.png: Violin plot comparison")
    print(f"  - histogram.png: Histogram comparison")


if __name__ == "__main__":
    main()
