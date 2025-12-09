"""Compare entity-level and edge-level perplexities.

This script analyzes whether entity perplexity predicts edge perplexity.
For each edge, we create two datapoints (one for subject, one for object)
and correlate entity perplexity with the edge perplexity.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(entity_file: Path, edge_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load entity and edge perplexity data.

    Args:
        entity_file: Path to entity_perplexities.tsv
        edge_file: Path to results.tsv (edge perplexities)

    Returns:
        Tuple of (entity_df, edge_df)
    """
    entity_df = pd.read_csv(entity_file, sep='\t')
    edge_df = pd.read_csv(edge_file, sep='\t')

    print(f"Loaded {len(entity_df)} entity perplexities")
    print(f"Loaded {len(edge_df)} edge perplexities")

    return entity_df, edge_df


def create_entity_edge_comparison(
    entity_df: pd.DataFrame,
    edge_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create comparison datasets with entity and edge perplexities.

    Creates two datasets:
    1. Individual entity datapoints (one for subject, one for object)
    2. Product datapoints (one per edge with product of both entities)

    Args:
        entity_df: DataFrame with entity perplexities
        edge_df: DataFrame with edge perplexities

    Returns:
        Tuple of (individual_df, product_df)
    """
    # Create entity lookup dictionary
    entity_lookup = entity_df.set_index('entity_id').to_dict('index')

    individual_rows = []
    product_rows = []

    for _, edge in edge_df.iterrows():
        subject_id = edge['subject_id']
        object_id = edge['object_id']
        edge_perp = edge['perplexity']
        edge_category = edge['category']
        edge_text = edge['text']

        # Get entity data
        subject_data = entity_lookup.get(subject_id)
        object_data = entity_lookup.get(object_id)

        # Add individual subject datapoint
        if subject_data:
            individual_rows.append({
                'entity_id': subject_id,
                'entity_name': subject_data['entity_name'],
                'entity_role': 'subject',
                'isolated_perp': subject_data['isolated_perp'],
                'neutral_perp': subject_data['neutral_perp'],
                'typed_perp': subject_data['typed_perp'],
                'kg_degree': subject_data['kg_degree'],
                'edge_perplexity': edge_perp,
                'edge_category': edge_category,
                'edge_text': edge_text,
            })

        # Add individual object datapoint
        if object_data:
            individual_rows.append({
                'entity_id': object_id,
                'entity_name': object_data['entity_name'],
                'entity_role': 'object',
                'isolated_perp': object_data['isolated_perp'],
                'neutral_perp': object_data['neutral_perp'],
                'typed_perp': object_data['typed_perp'],
                'kg_degree': object_data['kg_degree'],
                'edge_perplexity': edge_perp,
                'edge_category': edge_category,
                'edge_text': edge_text,
            })

        # Add product datapoint if both entities exist
        if subject_data and object_data:
            isolated_prod = subject_data['isolated_perp'] * object_data['isolated_perp']
            neutral_prod = subject_data['neutral_perp'] * object_data['neutral_perp']
            typed_prod = subject_data['typed_perp'] * object_data['typed_perp']

            product_rows.append({
                'subject_id': subject_id,
                'object_id': object_id,
                'subject_name': subject_data['entity_name'],
                'object_name': object_data['entity_name'],
                'isolated_product': isolated_prod,
                'neutral_product': neutral_prod,
                'typed_product': typed_prod,
                'edge_perplexity': edge_perp,
                'edge_category': edge_category,
                'edge_text': edge_text,
                # Add normalized perplexities
                'isolated_normalized': edge_perp / isolated_prod if isolated_prod > 0 else None,
                'neutral_normalized': edge_perp / neutral_prod if neutral_prod > 0 else None,
                'typed_normalized': edge_perp / typed_prod if typed_prod > 0 else None,
            })

    individual_df = pd.DataFrame(individual_rows)
    product_df = pd.DataFrame(product_rows)

    print(f"\nCreated {len(individual_df)} individual entity-edge datapoints")
    print(f"  From {len(individual_df) // 2} edges (2 entities per edge)")
    print(f"Created {len(product_df)} product datapoints (1 per edge)")

    return individual_df, product_df


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate correlations between entity and edge perplexities.

    Args:
        df: Comparison DataFrame

    Returns:
        Dictionary of correlation results
    """
    results = {}

    entity_contexts = ['isolated_perp', 'neutral_perp', 'typed_perp']

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS: Entity Perplexity vs Edge Perplexity")
    print("=" * 80)

    # Overall correlations (all categories combined)
    print("\nOverall Correlations (all edge categories):")
    print("-" * 80)

    for context in entity_contexts:
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(df[context], df['edge_perplexity'])

        # Spearman correlation (rank-based, more robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(df[context], df['edge_perplexity'])

        results[f'{context}_overall'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        }

        print(f"\n{context}:")
        print(f"  Pearson:  r={pearson_r:7.4f}, p={pearson_p:.4e}")
        print(f"  Spearman: r={spearman_r:7.4f}, p={spearman_p:.4e}")

    # By edge category
    print("\n" + "-" * 80)
    print("Correlations by Edge Category:")
    print("-" * 80)

    for category in df['edge_category'].unique():
        cat_df = df[df['edge_category'] == category]

        if len(cat_df) < 3:
            continue

        print(f"\n{category} (n={len(cat_df)}):")

        for context in entity_contexts:
            pearson_r, pearson_p = stats.pearsonr(cat_df[context], cat_df['edge_perplexity'])
            spearman_r, spearman_p = stats.spearmanr(cat_df[context], cat_df['edge_perplexity'])

            results[f'{context}_{category}'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
            }

            print(f"  {context:15s}: Pearson r={pearson_r:7.4f}, Spearman r={spearman_r:7.4f}")

    # By entity role
    print("\n" + "-" * 80)
    print("Correlations by Entity Role:")
    print("-" * 80)

    for role in ['subject', 'object']:
        role_df = df[df['entity_role'] == role]

        print(f"\n{role.capitalize()} (n={len(role_df)}):")

        for context in entity_contexts:
            pearson_r, pearson_p = stats.pearsonr(role_df[context], role_df['edge_perplexity'])
            spearman_r, spearman_p = stats.spearmanr(role_df[context], role_df['edge_perplexity'])

            results[f'{context}_{role}'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
            }

            print(f"  {context:15s}: Pearson r={pearson_r:7.4f}, Spearman r={spearman_r:7.4f}")

    print("\n" + "=" * 80)

    return results


def calculate_product_correlations(product_df: pd.DataFrame) -> dict:
    """Calculate correlations between entity product and edge perplexities.

    Args:
        product_df: Product DataFrame

    Returns:
        Dictionary of correlation results
    """
    results = {}

    product_contexts = ['isolated_product', 'neutral_product', 'typed_product']

    print("\n" + "=" * 80)
    print("PRODUCT CORRELATION ANALYSIS: Entity Product vs Edge Perplexity")
    print("=" * 80)

    # Overall correlations
    print("\nOverall Correlations (all edge categories):")
    print("-" * 80)

    for context in product_contexts:
        pearson_r, pearson_p = stats.pearsonr(product_df[context], product_df['edge_perplexity'])
        spearman_r, spearman_p = stats.spearmanr(product_df[context], product_df['edge_perplexity'])

        results[f'{context}_overall'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        }

        print(f"\n{context}:")
        print(f"  Pearson:  r={pearson_r:7.4f}, p={pearson_p:.4e}")
        print(f"  Spearman: r={spearman_r:7.4f}, p={spearman_p:.4e}")

    # By edge category
    print("\n" + "-" * 80)
    print("Correlations by Edge Category:")
    print("-" * 80)

    for category in product_df['edge_category'].unique():
        cat_df = product_df[product_df['edge_category'] == category]

        if len(cat_df) < 3:
            continue

        print(f"\n{category} (n={len(cat_df)}):")

        for context in product_contexts:
            pearson_r, pearson_p = stats.pearsonr(cat_df[context], cat_df['edge_perplexity'])
            spearman_r, spearman_p = stats.spearmanr(cat_df[context], cat_df['edge_perplexity'])

            results[f'{context}_{category}'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
            }

            print(f"  {context:20s}: Pearson r={pearson_r:7.4f}, Spearman r={spearman_r:7.4f}")

    print("\n" + "=" * 80)

    return results


def analyze_normalized_perplexity(product_df: pd.DataFrame) -> dict:
    """Compare normalized perplexity distributions between true and false_permuted.

    Args:
        product_df: Product DataFrame with normalized perplexities

    Returns:
        Dictionary of statistical test results
    """
    results = {}

    # Filter for true and false_permuted only (they have the most data)
    true_df = product_df[product_df['edge_category'] == 'true'].copy()
    false_df = product_df[product_df['edge_category'] == 'false_permuted'].copy()

    # Remove any None values
    true_df = true_df.dropna(subset=['isolated_normalized', 'neutral_normalized', 'typed_normalized'])
    false_df = false_df.dropna(subset=['isolated_normalized', 'neutral_normalized', 'typed_normalized'])

    print("\n" + "=" * 80)
    print("NORMALIZED PERPLEXITY ANALYSIS: True vs False-Permuted")
    print("=" * 80)
    print(f"\nTrue edges: n={len(true_df)}")
    print(f"False-permuted edges: n={len(false_df)}")

    normalized_contexts = ['isolated_normalized', 'neutral_normalized', 'typed_normalized']
    context_labels = ['Isolated', 'Neutral', 'Typed']

    print("\n" + "-" * 80)
    print("Descriptive Statistics:")
    print("-" * 80)

    for context, label in zip(normalized_contexts, context_labels):
        true_data = true_df[context]
        false_data = false_df[context]

        print(f"\n{label} Normalized:")
        print(f"  True:           mean={true_data.mean():.6f}, median={true_data.median():.6f}, std={true_data.std():.6f}")
        print(f"  False-permuted: mean={false_data.mean():.6f}, median={false_data.median():.6f}, std={false_data.std():.6f}")

        # Store for results
        results[f'{context}_true'] = {
            'mean': float(true_data.mean()),
            'median': float(true_data.median()),
            'std': float(true_data.std()),
            'min': float(true_data.min()),
            'max': float(true_data.max()),
        }
        results[f'{context}_false_permuted'] = {
            'mean': float(false_data.mean()),
            'median': float(false_data.median()),
            'std': float(false_data.std()),
            'min': float(false_data.min()),
            'max': float(false_data.max()),
        }

    print("\n" + "-" * 80)
    print("Statistical Tests (True vs False-Permuted):")
    print("-" * 80)

    for context, label in zip(normalized_contexts, context_labels):
        true_data = true_df[context]
        false_data = false_df[context]

        # T-test
        t_stat, t_p = stats.ttest_ind(true_data, false_data)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(true_data, false_data, alternative='two-sided')

        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.var(true_data) + np.var(false_data)) / 2)
        cohens_d = (np.mean(true_data) - np.mean(false_data)) / pooled_std if pooled_std > 0 else 0

        # Kolmogorov-Smirnov test (distribution difference)
        ks_stat, ks_p = stats.ks_2samp(true_data, false_data)

        results[f'{context}_comparison'] = {
            't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
            'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
            'cohens_d': float(cohens_d),
            'ks_test': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
            'mean_diff': float(np.mean(true_data) - np.mean(false_data)),
        }

        print(f"\n{label} Normalized:")
        print(f"  t-test:        t={t_stat:8.3f}, p={t_p:.4e}")
        print(f"  Mann-Whitney:  U={u_stat:8.0f}, p={u_p:.4e}")
        print(f"  Cohen's d:     {cohens_d:8.4f} {'(small)' if abs(cohens_d) < 0.5 else '(medium)' if abs(cohens_d) < 0.8 else '(large)'}")
        print(f"  KS test:       D={ks_stat:8.4f}, p={ks_p:.4e}")
        print(f"  Mean diff:     {np.mean(true_data) - np.mean(false_data):8.6f}")

    print("\n" + "=" * 80)

    return results, true_df, false_df


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualization plots.

    Args:
        df: Comparison DataFrame
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    entity_contexts = ['isolated_perp', 'neutral_perp', 'typed_perp']
    context_labels = ['Isolated', 'Neutral', 'Typed']

    # Plot 1: Scatter plots organized by category (rows) x context (columns)
    categories = sorted(df['edge_category'].unique())
    fig, axes = plt.subplots(len(categories), 3, figsize=(18, 5 * len(categories)))

    if len(categories) == 1:
        axes = axes.reshape(1, -1)

    for cat_idx, category in enumerate(categories):
        cat_df = df[df['edge_category'] == category]

        for ctx_idx, (context, label) in enumerate(zip(entity_contexts, context_labels)):
            ax = axes[cat_idx, ctx_idx]

            ax.scatter(cat_df[context], cat_df['edge_perplexity'],
                      alpha=0.4, s=15, c='blue')

            # Calculate and display correlation
            if len(cat_df) >= 3:
                r, p_val = stats.pearsonr(cat_df[context], cat_df['edge_perplexity'])
                ax.text(0.05, 0.95, f'r={r:.3f}\np={p_val:.2e}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(f'Entity Perplexity ({label})', fontsize=10)
            ax.set_ylabel('Edge Perplexity', fontsize=10)
            ax.set_title(f'{category} - {label}', fontsize=11)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = output_dir / "entity_vs_edge_scatter.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"\nSaved scatter plot to {plot1_path}")

    # Plot 2: Separate plots by edge category
    categories = df['edge_category'].unique()
    fig, axes = plt.subplots(len(categories), 3, figsize=(18, 5 * len(categories)))

    if len(categories) == 1:
        axes = axes.reshape(1, -1)

    for cat_idx, category in enumerate(categories):
        cat_df = df[df['edge_category'] == category]

        for ctx_idx, (context, label) in enumerate(zip(entity_contexts, context_labels)):
            ax = axes[cat_idx, ctx_idx]

            ax.scatter(cat_df[context], cat_df['edge_perplexity'],
                      alpha=0.5, s=20, c='blue')

            # Add regression line
            if len(cat_df) > 1:
                z = np.polyfit(np.log10(cat_df[context]), np.log10(cat_df['edge_perplexity']), 1)
                p = np.poly1d(z)
                x_line = np.logspace(np.log10(cat_df[context].min()),
                                    np.log10(cat_df[context].max()), 100)
                y_line = 10 ** p(np.log10(x_line))
                ax.plot(x_line, y_line, 'r--', alpha=0.7, linewidth=2)

            # Calculate and display correlation
            if len(cat_df) >= 3:
                r, p_val = stats.pearsonr(cat_df[context], cat_df['edge_perplexity'])
                ax.text(0.05, 0.95, f'r={r:.3f}\np={p_val:.2e}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(f'Entity Perplexity ({label})', fontsize=10)
            ax.set_ylabel('Edge Perplexity', fontsize=10)
            ax.set_title(f'{category} - {label}', fontsize=11)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot2_path = output_dir / "entity_vs_edge_by_category.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved category comparison plot to {plot2_path}")

    # Plot 3: Comparison of three contexts (correlation strength)
    pearson_corrs = []
    spearman_corrs = []

    for context in entity_contexts:
        pearson_r, _ = stats.pearsonr(df[context], df['edge_perplexity'])
        spearman_r, _ = stats.spearmanr(df[context], df['edge_perplexity'])
        pearson_corrs.append(pearson_r)
        spearman_corrs.append(spearman_r)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(context_labels))
    width = 0.35

    ax.bar(x - width/2, pearson_corrs, width, label='Pearson', alpha=0.8)
    ax.bar(x + width/2, spearman_corrs, width, label='Spearman', alpha=0.8)

    ax.set_xlabel('Entity Context', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Correlation Strength: Entity vs Edge Perplexity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(context_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plot3_path = output_dir / "correlation_comparison.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved correlation comparison plot to {plot3_path}")


def create_product_visualizations(product_df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualization plots for product analysis.

    Args:
        product_df: Product DataFrame
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    product_contexts = ['isolated_product', 'neutral_product', 'typed_product']
    context_labels = ['Isolated Product', 'Neutral Product', 'Typed Product']

    # Plot 1: Scatter plots organized by category (rows) x context (columns)
    categories = sorted(product_df['edge_category'].unique())
    fig, axes = plt.subplots(len(categories), 3, figsize=(18, 5 * len(categories)))

    if len(categories) == 1:
        axes = axes.reshape(1, -1)

    for cat_idx, category in enumerate(categories):
        cat_df = product_df[product_df['edge_category'] == category]

        for ctx_idx, (context, label) in enumerate(zip(product_contexts, context_labels)):
            ax = axes[cat_idx, ctx_idx]

            ax.scatter(cat_df[context], cat_df['edge_perplexity'],
                      alpha=0.4, s=15, c='green')

            # Calculate and display correlation
            if len(cat_df) >= 3:
                r, p_val = stats.pearsonr(cat_df[context], cat_df['edge_perplexity'])
                ax.text(0.05, 0.95, f'r={r:.3f}\np={p_val:.2e}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel('Edge Perplexity', fontsize=10)
            ax.set_title(f'{category} - {label}', fontsize=11)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "product_vs_edge_scatter.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved product scatter plot to {plot_path}")

    # Plot 2: Comparison of correlation strengths
    pearson_corrs = []
    spearman_corrs = []

    for context in product_contexts:
        pearson_r, _ = stats.pearsonr(product_df[context], product_df['edge_perplexity'])
        spearman_r, _ = stats.spearmanr(product_df[context], product_df['edge_perplexity'])
        pearson_corrs.append(pearson_r)
        spearman_corrs.append(spearman_r)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(context_labels))
    width = 0.35

    ax.bar(x - width/2, pearson_corrs, width, label='Pearson', alpha=0.8, color='green')
    ax.bar(x + width/2, spearman_corrs, width, label='Spearman', alpha=0.8, color='darkgreen')

    ax.set_xlabel('Entity Product Context', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Correlation Strength: Entity Product vs Edge Perplexity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(context_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plot2_path = output_dir / "product_correlation_comparison.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved product correlation comparison plot to {plot2_path}")


def create_normalized_visualizations(true_df: pd.DataFrame, false_df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualizations comparing normalized perplexity distributions.

    Args:
        true_df: DataFrame with true edges
        false_df: DataFrame with false_permuted edges
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    normalized_contexts = ['isolated_normalized', 'neutral_normalized', 'typed_normalized']
    context_labels = ['Isolated Normalized', 'Neutral Normalized', 'Typed Normalized']

    # Plot 1: Histograms comparing distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (context, label) in enumerate(zip(normalized_contexts, context_labels)):
        ax = axes[idx]

        true_data = true_df[context]
        false_data = false_df[context]

        # Plot histograms
        ax.hist(true_data, bins=50, alpha=0.6, label='True', color='blue', density=True)
        ax.hist(false_data, bins=50, alpha=0.6, label='False-Permuted', color='red', density=True)

        # Add vertical lines for means
        ax.axvline(true_data.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(false_data.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label}\n(dashed lines = means)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = output_dir / "normalized_distributions.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"\nSaved normalized distribution plot to {plot1_path}")

    # Plot 2: Box plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (context, label) in enumerate(zip(normalized_contexts, context_labels)):
        ax = axes[idx]

        data_to_plot = [
            true_df[context],
            false_df[context]
        ]

        bp = ax.boxplot(data_to_plot, labels=['True', 'False-Permuted'],
                       patch_artist=True, showmeans=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot2_path = output_dir / "normalized_boxplots.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved normalized boxplot to {plot2_path}")

    # Plot 3: Violin plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (context, label) in enumerate(zip(normalized_contexts, context_labels)):
        ax = axes[idx]

        # Prepare data for violin plot
        plot_df = pd.DataFrame({
            'value': pd.concat([true_df[context], false_df[context]]),
            'category': ['True'] * len(true_df) + ['False-Permuted'] * len(false_df)
        })

        sns.violinplot(data=plot_df, x='category', y='value', ax=ax,
                      palette={'True': 'lightblue', 'False-Permuted': 'lightcoral'})

        ax.set_xlabel('', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot3_path = output_dir / "normalized_violinplots.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved normalized violin plot to {plot3_path}")

    # Plot 4: Cumulative distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (context, label) in enumerate(zip(normalized_contexts, context_labels)):
        ax = axes[idx]

        true_data = np.sort(true_df[context])
        false_data = np.sort(false_df[context])

        true_cdf = np.arange(1, len(true_data) + 1) / len(true_data)
        false_cdf = np.arange(1, len(false_data) + 1) / len(false_data)

        ax.plot(true_data, true_cdf, label='True', color='blue', linewidth=2)
        ax.plot(false_data, false_cdf, label='False-Permuted', color='red', linewidth=2)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title(f'{label} - CDF', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot4_path = output_dir / "normalized_cdf.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved normalized CDF plot to {plot4_path}")


def main():
    """Run entity-edge perplexity comparison."""
    parser = argparse.ArgumentParser(
        description="Compare entity and edge perplexities"
    )
    parser.add_argument(
        "--entity-file",
        type=str,
        required=True,
        help="Path to entity_perplexities.tsv"
    )
    parser.add_argument(
        "--edge-file",
        type=str,
        required=True,
        help="Path to edge results.tsv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/entity_edge_comparison",
        help="Output directory for results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENTITY-EDGE PERPLEXITY COMPARISON")
    print("=" * 80)
    print(f"Entity file: {args.entity_file}")
    print(f"Edge file: {args.edge_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load data
    entity_df, edge_df = load_data(Path(args.entity_file), Path(args.edge_file))

    # Create comparison datasets
    individual_df, product_df = create_entity_edge_comparison(entity_df, edge_df)

    # Save comparison data
    individual_path = output_dir / "entity_edge_comparison.tsv"
    individual_df.to_csv(individual_path, sep='\t', index=False)
    print(f"\nSaved individual comparison data to {individual_path}")

    product_path = output_dir / "entity_product_comparison.tsv"
    product_df.to_csv(product_path, sep='\t', index=False)
    print(f"Saved product comparison data to {product_path}")

    # Calculate correlations for individual entities
    individual_correlations = calculate_correlations(individual_df)

    # Calculate correlations for products
    product_correlations = calculate_product_correlations(product_df)

    # Analyze normalized perplexity
    normalized_results, true_df_norm, false_df_norm = analyze_normalized_perplexity(product_df)

    # Create visualizations
    print("\nGenerating individual entity visualizations...")
    create_visualizations(individual_df, output_dir)

    print("\nGenerating product visualizations...")
    create_product_visualizations(product_df, output_dir)

    print("\nGenerating normalized perplexity visualizations...")
    create_normalized_visualizations(true_df_norm, false_df_norm, output_dir)

    print("\n" + "=" * 80)
    print("Done!")
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - entity_edge_comparison.tsv: Individual entity comparison dataset")
    print("  - entity_product_comparison.tsv: Product comparison dataset (with normalized perplexities)")
    print("  - entity_vs_edge_scatter.png: Scatter plots for individual entities")
    print("  - entity_vs_edge_by_category.png: Detailed plots by category")
    print("  - correlation_comparison.png: Individual entity correlation strengths")
    print("  - product_vs_edge_scatter.png: Scatter plots for entity products")
    print("  - product_correlation_comparison.png: Product correlation strengths")
    print("  - normalized_distributions.png: Histograms of normalized perplexity (true vs false)")
    print("  - normalized_boxplots.png: Box plots of normalized perplexity")
    print("  - normalized_violinplots.png: Violin plots of normalized perplexity")
    print("  - normalized_cdf.png: Cumulative distribution functions")
    print("=" * 80)


if __name__ == "__main__":
    main()
