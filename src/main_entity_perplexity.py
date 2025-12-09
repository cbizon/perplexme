"""Calculate and analyze entity perplexities in different contexts."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.cli_common import (
    create_base_argument_parser,
    setup_output_directory,
    load_knowledge_graph,
    create_perplexity_calculator,
    print_banner,
)
from src.entity_perplexity import calculate_entity_perplexities


def analyze_entity_perplexities(results, output_dir):
    """Analyze and visualize entity perplexity results.

    Args:
        results: Dictionary of entity_id -> EntityPerplexityResult
        output_dir: Path to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    data = []
    for entity_id, result in results.items():
        data.append({
            'entity_id': result.entity_id,
            'entity_name': result.entity_name,
            'isolated_perp': result.isolated_perplexity,
            'neutral_perp': result.neutral_perplexity,
            'typed_perp': result.typed_perplexity,
            'kg_degree': result.kg_degree,
        })

    df = pd.DataFrame(data)

    # Save raw data
    tsv_path = output_dir / "entity_perplexities.tsv"
    df.to_csv(tsv_path, sep='\t', index=False)
    print(f"\nSaved entity perplexities to {tsv_path}")

    # Calculate correlations
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    corr_matrix = df[['isolated_perp', 'neutral_perp', 'typed_perp', 'kg_degree']].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Correlation with KG degree
    print("\n" + "-"*80)
    print("Correlation with KG Degree (# edges):")
    print("-"*80)
    for context in ['isolated_perp', 'neutral_perp', 'typed_perp']:
        corr, pval = stats.pearsonr(df[context], df['kg_degree'])
        print(f"{context:20s}: r={corr:7.4f}, p={pval:.4e}")

    # Pairwise comparisons between contexts
    print("\n" + "-"*80)
    print("Pairwise Comparisons Between Contexts:")
    print("-"*80)

    contexts = ['isolated_perp', 'neutral_perp', 'typed_perp']
    for i, ctx1 in enumerate(contexts):
        for ctx2 in contexts[i+1:]:
            corr, pval = stats.pearsonr(df[ctx1], df[ctx2])
            print(f"{ctx1} vs {ctx2}: r={corr:.4f}, p={pval:.4e}")

    # Descriptive statistics
    print("\n" + "-"*80)
    print("Descriptive Statistics:")
    print("-"*80)
    print(df[['isolated_perp', 'neutral_perp', 'typed_perp', 'kg_degree']].describe())

    # Create visualizations
    sns.set_style("whitegrid")

    # Plot 1: Comparison of three context methods
    plt.figure(figsize=(12, 6))
    data_for_plot = df[['isolated_perp', 'neutral_perp', 'typed_perp']].melt(
        var_name='context', value_name='perplexity'
    )
    sns.boxplot(data=data_for_plot, x='context', y='perplexity')
    plt.title('Entity Perplexity by Context Method')
    plt.xlabel('Context Type')
    plt.ylabel('Perplexity')
    plt.xticks([0, 1, 2], ['Isolated', 'Neutral', 'Typed'])
    plt.tight_layout()
    plot1_path = output_dir / "context_comparison.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"\nSaved context comparison plot to {plot1_path}")

    # Plot 2: Perplexity vs KG degree for each context
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, context in enumerate(['isolated_perp', 'neutral_perp', 'typed_perp']):
        axes[idx].scatter(df['kg_degree'], df[context], alpha=0.5)
        axes[idx].set_xlabel('KG Degree (# edges)')
        axes[idx].set_ylabel('Perplexity')
        axes[idx].set_title(context.replace('_perp', '').capitalize())
        axes[idx].set_xscale('log')
        axes[idx].set_yscale('log')
    plt.tight_layout()
    plot2_path = output_dir / "perplexity_vs_degree.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved perplexity vs degree plot to {plot2_path}")

    # Plot 3: Pairwise scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ('isolated_perp', 'neutral_perp'),
        ('isolated_perp', 'typed_perp'),
        ('neutral_perp', 'typed_perp'),
    ]
    for idx, (x, y) in enumerate(pairs):
        axes[idx].scatter(df[x], df[y], alpha=0.5)
        axes[idx].set_xlabel(x.replace('_perp', '').capitalize())
        axes[idx].set_ylabel(y.replace('_perp', '').capitalize())
        axes[idx].plot([df[x].min(), df[x].max()], [df[y].min(), df[y].max()],
                      'r--', alpha=0.5, label='y=x')
        axes[idx].legend()
    plt.tight_layout()
    plot3_path = output_dir / "context_pairwise.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved pairwise comparison plot to {plot3_path}")

    print("\n" + "="*80)


def main():
    """Run entity perplexity analysis."""
    parser = create_base_argument_parser(
        description="Calculate entity perplexities in different contexts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/entity_perp_<predicate>_<timestamp>)",
    )

    args = parser.parse_args()

    # Set up output directory with "entity_perp" prefix
    output_dir = setup_output_directory(args.output_dir, args.predicate, prefix="entity_perp")

    print_banner("ENTITY PERPLEXITY ANALYSIS")
    print(f"Predicate: {args.predicate}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Step 1: Load knowledge graph
    print("\n[1/3] Loading knowledge graph...")
    kg = load_knowledge_graph(args.nodes_file, args.edges_file, args.predicate)

    # Step 2: Calculate entity perplexities
    print("\n[2/3] Calculating entity perplexities...")
    calculator = create_perplexity_calculator(args.model, args.device, args.batch_size)
    entity_perplexities = calculate_entity_perplexities(kg, calculator, args.predicate)

    # Step 3: Analyze and visualize
    print("\n[3/3] Analyzing results...")
    analyze_entity_perplexities(entity_perplexities, output_dir)

    print("\nDone!")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - entity_perplexities.tsv: Raw data for all entities")
    print(f"  - context_comparison.png: Box plots comparing three context methods")
    print(f"  - perplexity_vs_degree.png: Entity perplexity vs KG degree")
    print(f"  - context_pairwise.png: Pairwise comparison of context methods")


if __name__ == "__main__":
    main()
