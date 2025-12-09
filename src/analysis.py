"""Statistical analysis and visualization of perplexity results."""

import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import CATEGORIES


class PerplexityAnalyzer:
    """Analyze and visualize perplexity results."""

    def __init__(self, results: List[Dict], output_dir: Path):
        """Initialize the analyzer.

        Args:
            results: List of result dictionaries with perplexity scores
            output_dir: Directory to save outputs
        """
        self.df = pd.DataFrame(results)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def compute_descriptive_stats(self) -> Dict:
        """Compute descriptive statistics for each category.

        Returns:
            Dictionary with statistics per category
        """
        stats_dict = {}

        for category in CATEGORIES:
            cat_data = self.df[self.df['category'] == category]['perplexity']

            if len(cat_data) > 0:
                stats_dict[category] = {
                    'count': len(cat_data),
                    'mean': float(cat_data.mean()),
                    'median': float(cat_data.median()),
                    'std': float(cat_data.std()),
                    'min': float(cat_data.min()),
                    'max': float(cat_data.max()),
                    'q25': float(cat_data.quantile(0.25)),
                    'q75': float(cat_data.quantile(0.75)),
                }
            else:
                stats_dict[category] = None

        return stats_dict

    def compute_statistical_tests(self) -> Dict:
        """Compute statistical tests comparing categories.

        Returns:
            Dictionary with test results
        """
        test_results = {}

        # Get data by category
        category_data = {}
        for category in CATEGORIES:
            cat_data = self.df[self.df['category'] == category]['perplexity']
            if len(cat_data) > 0:
                category_data[category] = cat_data.values

        if len(category_data) < 2:
            return {"error": "Need at least 2 categories for statistical tests"}

        # ANOVA / Kruskal-Wallis test
        groups = list(category_data.values())
        if len(groups) >= 2:
            # Check for normality (Shapiro-Wilk test)
            normality_tests = {}
            for cat, data in category_data.items():
                if len(data) >= 3:
                    stat, p = stats.shapiro(data)
                    normality_tests[cat] = {'statistic': float(stat), 'p_value': float(p)}

            test_results['normality_tests'] = normality_tests

            # Use ANOVA if data seems normal, otherwise Kruskal-Wallis
            f_stat, anova_p = stats.f_oneway(*groups)
            test_results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(anova_p)
            }

            h_stat, kw_p = stats.kruskal(*groups)
            test_results['kruskal_wallis'] = {
                'h_statistic': float(h_stat),
                'p_value': float(kw_p)
            }

        # Pairwise t-tests
        pairwise_tests = {}
        categories_list = list(category_data.keys())
        for i, cat1 in enumerate(categories_list):
            for cat2 in categories_list[i+1:]:
                data1 = category_data[cat1]
                data2 = category_data[cat2]

                # T-test
                t_stat, t_p = stats.ttest_ind(data1, data2)
                # Mann-Whitney U test (non-parametric alternative)
                u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')

                # Cohen's d (effect size)
                pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

                pairwise_tests[f"{cat1}_vs_{cat2}"] = {
                    't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
                    'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
                    'cohens_d': float(cohens_d),
                    'mean_diff': float(np.mean(data1) - np.mean(data2)),
                }

        test_results['pairwise'] = pairwise_tests

        return test_results

    def create_box_plot(self) -> Path:
        """Create box plot comparing categories.

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 8))

        # Order categories consistently
        order = [cat for cat in CATEGORIES if cat in self.df['category'].values]

        sns.boxplot(data=self.df, x='category', y='perplexity', order=order)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Perplexity', fontsize=12)
        plt.title('Perplexity Distribution by Statement Category', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / "boxplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def create_violin_plot(self) -> Path:
        """Create violin plot comparing categories.

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 8))

        order = [cat for cat in CATEGORIES if cat in self.df['category'].values]

        sns.violinplot(data=self.df, x='category', y='perplexity', order=order)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Perplexity', fontsize=12)
        plt.title('Perplexity Distribution by Statement Category (Violin Plot)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / "violinplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def create_histogram(self) -> Path:
        """Create overlaid histograms for each category.

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 8))

        for category in CATEGORIES:
            cat_data = self.df[self.df['category'] == category]['perplexity']
            if len(cat_data) > 0:
                plt.hist(cat_data, alpha=0.5, label=category, bins=30)

        plt.xlabel('Perplexity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Perplexity Distribution by Category', fontsize=14)
        plt.legend()
        plt.tight_layout()

        output_path = self.output_dir / "histogram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def save_results(self) -> None:
        """Save all results to files."""
        print(f"Saving results to {self.output_dir}")

        # Save raw data
        tsv_path = self.output_dir / "results.tsv"
        self.df.to_csv(tsv_path, sep='\t', index=False)
        print(f"  Saved raw data to {tsv_path}")

        # Compute and save statistics
        descriptive_stats = self.compute_descriptive_stats()
        stats_path = self.output_dir / "descriptive_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(descriptive_stats, f, indent=2)
        print(f"  Saved descriptive statistics to {stats_path}")

        # Compute and save statistical tests
        test_results = self.compute_statistical_tests()
        tests_path = self.output_dir / "statistical_tests.json"
        with open(tests_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"  Saved statistical tests to {tests_path}")

        # Create visualizations
        boxplot_path = self.create_box_plot()
        print(f"  Saved box plot to {boxplot_path}")

        violin_path = self.create_violin_plot()
        print(f"  Saved violin plot to {violin_path}")

        hist_path = self.create_histogram()
        print(f"  Saved histogram to {hist_path}")

    def print_summary(self) -> None:
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("PERPLEXITY ANALYSIS SUMMARY")
        print("="*80)

        descriptive_stats = self.compute_descriptive_stats()

        print("\nDescriptive Statistics:")
        print("-" * 80)
        for category, stats_data in descriptive_stats.items():
            if stats_data is not None:
                print(f"\n{category}:")
                print(f"  Count:  {stats_data['count']}")
                print(f"  Mean:   {stats_data['mean']:.2f}")
                print(f"  Median: {stats_data['median']:.2f}")
                print(f"  Std:    {stats_data['std']:.2f}")
                print(f"  Range:  [{stats_data['min']:.2f}, {stats_data['max']:.2f}]")

        test_results = self.compute_statistical_tests()

        print("\n" + "-" * 80)
        print("Statistical Tests:")
        print("-" * 80)

        if 'anova' in test_results:
            print(f"\nANOVA: F={test_results['anova']['f_statistic']:.2f}, p={test_results['anova']['p_value']:.4f}")
            print(f"Kruskal-Wallis: H={test_results['kruskal_wallis']['h_statistic']:.2f}, p={test_results['kruskal_wallis']['p_value']:.4f}")

        if 'pairwise' in test_results:
            print("\nPairwise Comparisons:")
            for comparison, results in test_results['pairwise'].items():
                print(f"\n  {comparison}:")
                print(f"    t-test p-value: {results['t_test']['p_value']:.4f}")
                print(f"    Cohen's d: {results['cohens_d']:.3f}")
                print(f"    Mean difference: {results['mean_diff']:.2f}")

        print("\n" + "="*80)


def analyze_results(results: List[Dict], output_dir: Path) -> PerplexityAnalyzer:
    """Convenience function to run complete analysis.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save outputs

    Returns:
        PerplexityAnalyzer instance
    """
    analyzer = PerplexityAnalyzer(results, output_dir)
    analyzer.print_summary()
    analyzer.save_results()
    return analyzer
