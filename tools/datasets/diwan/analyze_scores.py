#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Poetry Score Analysis and Visualization

This script analyzes the scored results and generates comprehensive plots
comparing different models and human performance across various metrics.

Usage:
    python analyze_scores.py --input scored.json --output plots/
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime

# Suppress font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Font setup - use fonts that support both Arabic and Latin characters
def setup_font():
    """Setup the best available font that supports both Arabic and Latin characters."""
    # Suppress all font-related warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    # Use DejaVu Sans as default - it's reliable for Latin characters
    plt.rcParams['font.family'] = 'DejaVu Sans'
    logging.info("Using DejaVu Sans font (supports Latin characters reliably)")

setup_font()

class ScoreAnalyzer:
    def __init__(self, input_files: List[str], output_dir: str):
        """Initialize the analyzer."""
        self.input_files = input_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define model name mapping for better readability
        self.model_name_mapping = {
            'gemini-2.5-pro': 'Gemini-2.5',
            'claude-sonnet-4-20250514': 'Sonnet-4',
            'gpt-5-2025-08-07': 'GPT-5',
            'human': 'Human'
        }
        
        # Define colors for different sources
        self.colors = {
            'gemini-2.5-pro': '#4285F4',
            'claude-sonnet-4-20250514': '#FF6B35',
            'gpt-5-2025-08-07': '#34A853',
            'human': '#EA4335'
        }
        
        # Define color mapping for display names (including comparative evaluations)
        self.display_colors = {
            'Gemini-2.5': '#4285F4',
            'Sonnet-4': '#FF6B35',
            'GPT-5': '#34A853',
            'Human': '#EA4335'
        }
        
        # Build analysis dataframe from one or more files
        self.df = self.prepare_dataframe()
        
        # Define metric names in English (to avoid font issues)
        self.metric_names = {
            'meter': 'Meter',
            'rhyme': 'Rhyme', 
            'meaning': 'Meaning',
            'beauty': 'Beauty',
            'creativity': 'Creativity',
            'consistency': 'Consistency',
            'vocab': 'Vocabulary',
            'total': 'Total'
        }
    
    def _aggregate_file(self, input_file: str) -> pd.DataFrame:
        """Load one scored JSON and aggregate to (poem_id, source) by averaging single + comparative sides."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} results from {input_file}")
        except Exception as e:
            logging.error(f"Error loading {input_file}: {e}")
            raise

        rows: List[Dict[str, Any]] = []
        for item in data:
            scores = item.get('scores', {})
            row = {
                'poem_id': item.get('poem_id'),
                'source': item.get('source'),
            }
            for metric, value in scores.items():
                try:
                    row[metric] = float(value)
                except Exception:
                    pass
            rows.append(row)

        df = pd.DataFrame(rows)

        # Derive base source to merge single and comparative for each source
        def get_source_base(s: str) -> str:
            if isinstance(s, str) and '_vs_' in s:
                left, right = s.split('_vs_', 1)
                if left == 'human':
                    return 'human'
                if right == 'human':
                    return left
                return left
            return s
        df['source_base'] = df['source'].apply(get_source_base)

        metrics = ['meter', 'rhyme', 'meaning', 'beauty', 'creativity', 'consistency', 'vocab', 'total']
        # Ensure metric columns exist
        for m in metrics:
            if m not in df.columns:
                df[m] = np.nan

        agg_df = df.groupby(['poem_id', 'source_base'])[metrics].mean().reset_index()
        agg_df.rename(columns={'source_base': 'source'}, inplace=True)
        return agg_df
    
    def prepare_dataframe(self) -> pd.DataFrame:
        """Load multiple files, aggregate within each, intersect keys, and average across files."""
        if not self.input_files:
            raise ValueError("No input files provided")

        per_file_dfs: List[pd.DataFrame] = []
        for path in self.input_files:
            per_file_dfs.append(self._aggregate_file(path))

        # Compute intersection of (poem_id, source) across all files
        key_sets = [set(zip(df['poem_id'].astype(str), df['source'].astype(str))) for df in per_file_dfs]
        common_keys = set.intersection(*key_sets) if key_sets else set()
        logging.info(f"Common (poem_id, source) pairs across all files: {len(common_keys)}")

        if not common_keys:
            logging.warning("No common items across all files; resulting DataFrame will be empty")
            return pd.DataFrame(columns=['poem_id', 'source', 'source_display', 'color', 'meter', 'rhyme', 'meaning', 'beauty', 'creativity', 'consistency', 'vocab', 'total'])

        # Filter each df to common keys
        filtered = []
        for df in per_file_dfs:
            mask = list(zip(df['poem_id'].astype(str), df['source'].astype(str)))
            df_filtered = df[[key in common_keys for key in mask]].copy()
            filtered.append(df_filtered)

        # Concatenate and average metrics across files
        all_concat = pd.concat(filtered, ignore_index=True)
        metrics = ['meter', 'rhyme', 'meaning', 'beauty', 'creativity', 'consistency', 'vocab', 'total']
        avg_df = all_concat.groupby(['poem_id', 'source'])[metrics].mean().reset_index()

        # Display mapping and colors
        avg_df['source_display'] = avg_df['source'].map(self.model_name_mapping).fillna(avg_df['source'])

        def get_color_for_display_name(display_name):
            if ' vs ' in display_name:
                first_model = display_name.split(' vs ')[0]
                return self.display_colors.get(first_model, '#666666')
            else:
                return self.display_colors.get(display_name, '#666666')
        avg_df['color'] = avg_df['source_display'].apply(get_color_for_display_name)

        logging.info(f"Created averaged DataFrame with {len(avg_df)} rows from {len(self.input_files)} files")
        return avg_df
    
    def create_overall_comparison_plot(self):
        """Create overall comparison plot showing total scores by source."""
        plt.figure(figsize=(12, 8))
        
        # Calculate statistics using display names
        stats = self.df.groupby('source_display')['total'].agg(['mean', 'std', 'count']).reset_index()
        
        # Sort by mean score for better visualization
        stats = stats.sort_values('mean', ascending=False)
        
        # Create bar plot
        bars = plt.bar(range(len(stats)), stats['mean'], 
                      yerr=stats['std'], 
                      capsize=5, 
                      color=[self.display_colors.get(src.split(' vs ')[0] if ' vs ' in src else src, '#666666') for src in stats['source_display']],
                      alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, mean_val, count) in enumerate(zip(bars, stats['mean'], stats['count'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{mean_val:.2f}\n(n={count})', 
                    ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.title('Overall Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Source', fontsize=12)
        plt.ylabel('Average Total Score', fontsize=12)
        plt.ylim(0, 10)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels with better spacing
        plt.xticks(range(len(stats)), stats['source_display'], rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("Created overall comparison plot")
    
    def create_metric_comparison_plot(self):
        """Create detailed comparison plot for all metrics."""
        metrics = ['meter', 'rhyme', 'meaning', 'beauty', 'creativity', 'consistency', 'vocab']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get data for this metric using display names
            metric_data = self.df.groupby('source_display')[metric].agg(['mean', 'std']).reset_index()
            
            # Sort by mean score for better visualization
            metric_data = metric_data.sort_values('mean', ascending=False)
            
            # Create bar plot with better spacing
            x_pos = range(len(metric_data))
            bars = ax.bar(x_pos, metric_data['mean'],
                         yerr=metric_data['std'], capsize=3,
                         color=[self.display_colors.get(src.split(' vs ')[0] if ' vs ' in src else src, '#666666') for src in metric_data['source_display']],
                         alpha=0.8)
            
            ax.set_title(self.metric_names[metric], fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Score', fontsize=10)
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels with better spacing and rotation
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_data['source_display'], rotation=45, ha='right', fontsize=8)
            ax.tick_params(axis='y', labelsize=9)
        
        # Remove the last subplot (8th position)
        fig.delaxes(axes[7])
        
        plt.suptitle('Detailed Metric Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("Created metric comparison plot")
    
    def create_box_plot(self):
        """Create box plot showing score distributions."""
        plt.figure(figsize=(12, 8))
        
        # Create box plot for total scores using display names
        unique_sources = self.df['source_display'].unique()
        box_data = [self.df[self.df['source_display'] == src]['total'].values for src in unique_sources]
        
        bp = plt.boxplot(box_data, labels=unique_sources, patch_artist=True)
        
        # Color the boxes
        for patch, src in zip(bp['boxes'], unique_sources):
            patch.set_facecolor(self.display_colors.get(src.split(' vs ')[0] if ' vs ' in src else src, '#666666'))
            patch.set_alpha(0.7)
        
        plt.title('Total Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Source', fontsize=12)
        plt.ylabel('Total Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("Created score distribution box plot")
    
    def create_heatmap(self):
        """Create correlation heatmap between different metrics."""
        # Calculate correlation matrix for all metrics
        metrics = ['meter', 'rhyme', 'meaning', 'beauty', 'creativity', 'consistency', 'vocab', 'total']
        corr_matrix = self.df[metrics].corr()
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   xticklabels=[self.metric_names[m] for m in metrics],
                   yticklabels=[self.metric_names[m] for m in metrics],
                   annot_kws={'size': 8})
        
        plt.title('Metric Correlation Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("Created correlation heatmap")
    
    def create_radar_chart(self):
        """Create radar chart comparing all metrics across sources."""
        metrics = ['meter', 'rhyme', 'meaning', 'beauty', 'creativity', 'consistency', 'vocab']
        
        # Calculate mean scores for each source using display names
        source_means = {}
        for source in self.df['source_display'].unique():
            source_data = self.df[self.df['source_display'] == source]
            source_means[source] = [source_data[metric].mean() for metric in metrics]
        
        # Number of variables
        N = len(metrics)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot each source
        for source, means in source_means.items():
            values = means + means[:1]  # Complete the circle
            color = self.display_colors.get(source.split(' vs ')[0] if ' vs ' in source else source, '#666666')
            ax.plot(angles, values, 'o-', linewidth=2, label=source, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.metric_names[m] for m in metrics], fontsize=9)
        
        # Set the y-axis limits
        ax.set_ylim(0, 10)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        
        plt.title('Comprehensive Metric Comparison', fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("Created radar chart")
    
    def create_performance_ranking(self):
        """Create performance ranking table and visualization."""
        # Calculate overall performance metrics using display names
        performance = self.df.groupby('source_display').agg({
            'total': ['mean', 'std', 'count'],
            'meter': 'mean',
            'rhyme': 'mean',
            'meaning': 'mean',
            'beauty': 'mean',
            'creativity': 'mean',
            'consistency': 'mean',
            'vocab': 'mean'
        }).round(3)
        
        # Flatten column names
        performance.columns = ['_'.join(col).strip() for col in performance.columns]
        performance = performance.reset_index()
        
        # Sort by total mean
        performance = performance.sort_values('total_mean', ascending=False)
        
        # Save ranking table
        ranking_file = self.output_dir / 'performance_ranking.csv'
        performance.to_csv(ranking_file, index=False, encoding='utf-8')
        logging.info(f"Saved performance ranking to {ranking_file}")
        
        # Create ranking visualization
        plt.figure(figsize=(8, 6))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(performance))
        bars = plt.barh(y_pos, performance['total_mean'],
                       color=[self.display_colors.get(src.split(' vs ')[0] if ' vs ' in src else src, '#666666') for src in performance['source_display']],
                       alpha=0.8)
        
        # Add value labels
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, performance['total_mean'], performance['total_std'])):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{mean_val:.2f} ± {std_val:.2f}', 
                    va='center', fontweight='bold', fontsize=9)
        
        plt.yticks(y_pos, performance['source_display'], fontsize=9)
        plt.xlabel('Average Total Score', fontsize=10)
        plt.title('Performance Ranking by Total Score', fontsize=12, fontweight='bold')
        plt.xlim(0, 10)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_ranking.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("Created performance ranking visualization")
    
    def create_simplified_comparison(self):
        """Create simplified comparison plots separating single and comparative evaluations."""
        # Separate single and comparative evaluations
        single_eval = self.df[~self.df['source_display'].str.contains(' vs ', na=False)]
        comparative_eval = self.df[self.df['source_display'].str.contains(' vs ', na=False)]
        
        # Create single evaluation plot
        if len(single_eval) > 0:
            plt.figure(figsize=(10, 6))
            
            metrics = ['meter', 'rhyme', 'meaning', 'beauty', 'creativity', 'consistency', 'vocab']
            x = np.arange(len(metrics))
            width = 0.2
            
            for i, source in enumerate(single_eval['source_display'].unique()):
                source_data = single_eval[single_eval['source_display'] == source]
                means = [source_data[metric].mean() for metric in metrics]
                color = self.display_colors.get(source, '#666666')
                
                plt.bar(x + i * width, means, width, label=source, color=color, alpha=0.8)
            
            plt.xlabel('Metrics', fontsize=12)
            plt.ylabel('Average Score', fontsize=12)
            plt.title('Single Evaluation Comparison', fontsize=14, fontweight='bold')
            plt.xticks(x + width * 1.5, [self.metric_names[m] for m in metrics], rotation=45, ha='right')
            plt.legend()
            plt.ylim(0, 10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'single_evaluation_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            logging.info("Created single evaluation comparison plot")
        
        # Create comparative evaluation plot
        if len(comparative_eval) > 0:
            plt.figure(figsize=(12, 8))
            
            # Group by the AI model being evaluated
            comparative_summary = []
            for source in comparative_eval['source_display'].unique():
                if ' vs ' in source:
                    parts = source.split(' vs ')
                    ai_model = parts[0] if parts[0] != 'Human' else parts[1]
                    source_data = comparative_eval[comparative_eval['source_display'] == source]
                    comparative_summary.append({
                        'model': ai_model,
                        'evaluation_type': 'Comparative',
                        'total_mean': source_data['total'].mean(),
                        'total_std': source_data['total'].std()
                    })
            
            if comparative_summary:
                comp_df = pd.DataFrame(comparative_summary)
                comp_df = comp_df.sort_values('total_mean', ascending=False)
                
                bars = plt.bar(range(len(comp_df)), comp_df['total_mean'],
                              yerr=comp_df['total_std'], capsize=5,
                              color=['#666666'] * len(comp_df), alpha=0.8)
                
                plt.title('Comparative Evaluation Results', fontsize=14, fontweight='bold')
                plt.xlabel('AI Model', fontsize=12)
                plt.ylabel('Average Total Score', fontsize=12)
                plt.xticks(range(len(comp_df)), comp_df['model'], rotation=45, ha='right')
                plt.ylim(0, 10)
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, mean_val in zip(bars, comp_df['total_mean']):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'comparative_evaluation_results.png', dpi=150, bbox_inches='tight')
                plt.close()
                logging.info("Created comparative evaluation results plot")
    
    def create_statistical_summary(self):
        """Create comprehensive statistical summary."""
        summary_stats = {}
        
        # Overall statistics
        summary_stats['total_samples'] = len(self.df)
        summary_stats['unique_poems'] = self.df['poem_id'].nunique()
        summary_stats['sources'] = list(self.df['source'].unique())
        
        # Statistics by source
        source_stats = {}
        for source in self.df['source'].unique():
            source_data = self.df[self.df['source'] == source]
            source_stats[source] = {
                'count': len(source_data),
                'total_mean': source_data['total'].mean(),
                'total_std': source_data['total'].std(),
                'total_min': source_data['total'].min(),
                'total_max': source_data['total'].max()
            }
        
        summary_stats['by_source'] = source_stats
        
        # Correlation with human scores
        human_scores = self.df[self.df['source'] == 'human']['total']
        correlations = {}
        for source in self.df['source'].unique():
            if source != 'human':
                source_scores = self.df[self.df['source'] == source]['total']
                if len(human_scores) == len(source_scores):
                    corr = np.corrcoef(human_scores, source_scores)[0, 1]
                    correlations[source] = corr
        
        summary_stats['human_correlations'] = correlations
        
        # Save summary
        summary_file = self.output_dir / 'statistical_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved statistical summary to {summary_file}")
        
        # Print key findings
        print("\n" + "="*50)
        print("KEY FINDINGS")
        print("="*50)
        print(f"Total samples analyzed: {summary_stats['total_samples']}")
        print(f"Unique poems: {summary_stats['unique_poems']}")
        print(f"Sources: {', '.join(summary_stats['sources'])}")
        
        print("\nPerformance by Source:")
        for source, stats in source_stats.items():
            print(f"  {source}: {stats['total_mean']:.2f} ± {stats['total_std']:.2f} (n={stats['count']})")
        
        print("\nCorrelation with Human Scores:")
        for source, corr in correlations.items():
            print(f"  {source}: {corr:.3f}")
        
        # Find best performing model
        best_model = max(source_stats.items(), key=lambda x: x[1]['total_mean'])
        print(f"\nBest performing model: {best_model[0]} ({best_model[1]['total_mean']:.2f})")
        
        # Find model closest to human performance
        human_mean = source_stats['human']['total_mean']
        closest_model = min([(src, stats) for src, stats in source_stats.items() if src != 'human'], 
                           key=lambda x: abs(x[1]['total_mean'] - human_mean))
        print(f"Model closest to human performance: {closest_model[0]} ({closest_model[1]['total_mean']:.2f})")
    
    def run_analysis(self):
        """Run complete analysis and generate all plots."""
        logging.info("Starting comprehensive score analysis...")
        
        # Generate all plots
        self.create_overall_comparison_plot()
        self.create_metric_comparison_plot()
        self.create_box_plot()
        self.create_heatmap()
        self.create_radar_chart()
        self.create_performance_ranking()
        self.create_simplified_comparison()
        self.create_statistical_summary()
        
        logging.info(f"Analysis complete! All plots saved to {self.output_dir}")
        
        # Create a summary HTML file
        self.create_html_summary()
    
    def create_html_summary(self):
        """Create an HTML summary of all results."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic Poetry Evaluation Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .plot-section {{ margin: 30px 0; }}
        .plot-section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .plot-section img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Arabic Poetry Evaluation Analysis</h1>
        <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="plot-section">
        <h2>Overall Comparison</h2>
        <img src="overall_comparison.png" alt="Overall Comparison">
    </div>
    
    <div class="plot-section">
        <h2>Detailed Metric Comparison</h2>
        <img src="metric_comparison.png" alt="Metric Comparison">
    </div>
    
    <div class="plot-section">
        <h2>Score Distribution</h2>
        <img src="score_distribution.png" alt="Score Distribution">
    </div>
    
    <div class="plot-section">
        <h2>Correlation Matrix</h2>
        <img src="correlation_heatmap.png" alt="Correlation Heatmap">
    </div>
    
    <div class="plot-section">
        <h2>Comprehensive Radar Chart</h2>
        <img src="radar_chart.png" alt="Radar Chart">
    </div>
    
    <div class="plot-section">
        <h2>Performance Ranking</h2>
        <img src="performance_ranking.png" alt="Performance Ranking">
    </div>
    
    <div class="footer">
        <p>Generated by Arabic Poetry Analysis Tool</p>
    </div>
</body>
</html>
        """
        
        html_file = self.output_dir / 'analysis_summary.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"Created HTML summary: {html_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze Arabic poetry scores and generate plots")
    parser.add_argument("--input", required=True, nargs='+', help="One or more input JSON files with scored results")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Validate input file
    for p in args.input:
        if not Path(p).exists():
            logging.error(f"Input file {p} does not exist")
            return
    
    # Create analyzer and run analysis
    analyzer = ScoreAnalyzer(args.input, args.output)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()