"""
filepath: d:\Project Magang Bagas\Eval_Spider\generate_all_visualizations.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ============== KONFIGURASI ==============
RESULTS_DIR = "results"
OUTPUT_DIR = "results/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============== LOAD DATA ==============

def load_spider_results():
    """Load Spider evaluation results"""
    files = list(Path(RESULTS_DIR).glob("spider_evaluation_all_models_*.csv"))
    if files:
        latest = max(files, key=os.path.getctime)
        df = pd.read_csv(latest, sep=';')
        
        # ✅ Convert numeric columns
        numeric_cols = ['exact_match_all', 'exec_match_all', 'exact_match_easy', 
                       'exact_match_medium', 'exact_match_hard', 'exact_match_extra']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Loaded Spider results: {latest.name}")
        print(f"  Columns: {df.columns.tolist()}")
        return df
    return None

def load_syntax_validation():
    """Load syntax validation results"""
    files = list(Path(RESULTS_DIR).glob("syntax_validation_summary_*.csv"))
    if files:
        latest = max(files, key=os.path.getctime)
        df = pd.read_csv(latest, sep=';')
        
        # ✅ Convert numeric columns
        df['validity_rate'] = pd.to_numeric(df['validity_rate'], errors='coerce')
        df['valid_queries'] = pd.to_numeric(df['valid_queries'], errors='coerce')
        df['total_queries'] = pd.to_numeric(df['total_queries'], errors='coerce')
        
        print(f"✓ Loaded Syntax validation: {latest.name}")
        return df
    return None

def load_latency_stats():
    """Load latency statistics"""
    files = list(Path(RESULTS_DIR + "/latency_analysis").glob("latency_statistics_*.csv"))
    if files:
        latest = max(files, key=os.path.getctime)
        df = pd.read_csv(latest)
        print(f"✓ Loaded Latency stats: {latest.name}")
        return df
    return None

def load_metrics_details():
    """Load detailed metrics for complexity analysis"""
    metrics_dirs = [
        "output/Few Shot Output",
        "output/Few Shot + Llamaindex Output"
    ]
    
    all_data = []
    for metrics_dir in metrics_dirs:
        if os.path.exists(metrics_dir):
            for file in Path(metrics_dir).glob("metric_*.csv"):
                try:
                    df = pd.read_csv(file, sep=';')
                    
                    # Extract info from filename
                    filename = file.stem
                    if 'fewshot' in filename.lower():
                        approach = 'LlamaIndex'
                        model = filename.replace('metric_fewshot_llamaindex_', '').replace('metric_fewshot_', '')
                    else:
                        approach = 'Manual'
                        model = filename.replace('metric_', '')
                    
                    model = '_'.join(model.split('_')[:-2])
                    df['model'] = model
                    df['approach'] = approach
                    all_data.append(df)
                except Exception as e:
                    print(f"  Warning: Failed to load {file.name}: {e}")
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        print(f"✓ Loaded detailed metrics: {len(df)} queries")
        return df
    return None

# ============== FIGURE 1: Compare EX per Model (Grouped Bar Chart) ==============

def plot_figure1_execution_accuracy(df_spider, output_dir):
    """Figure 1A: Grouped bar chart comparing Execution Accuracy per model"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data - pivot to get LlamaIndex vs Manual side by side
    # Get unique models
    models = df_spider['model_name'].unique()
    
    # Create data structure for grouped bars
    llamaindex_scores = []
    manual_scores = []
    model_labels = []
    
    for model in models:
        df_model = df_spider[df_spider['model_name'] == model]
        
        # Get LlamaIndex score
        llamaindex_row = df_model[df_model['approach'] == 'Few-Shot LlamaIndex']
        if len(llamaindex_row) > 0:
            llamaindex_score = llamaindex_row['exec_match_all'].iloc[0]
        else:
            llamaindex_score = 0
        
        # Get Manual score
        manual_row = df_model[df_model['approach'] == 'Few-Shot Manual']
        if len(manual_row) > 0:
            manual_score = manual_row['exec_match_all'].iloc[0]
        else:
            manual_score = 0
        
        # Only add if at least one approach has data
        if llamaindex_score > 0 or manual_score > 0:
            model_labels.append(model)
            llamaindex_scores.append(llamaindex_score)
            manual_scores.append(manual_score)
    
    # Sort by average score (descending)
    avg_scores = [(l + m) / 2 for l, m in zip(llamaindex_scores, manual_scores)]
    sorted_indices = sorted(range(len(avg_scores)), key=lambda i: avg_scores[i], reverse=True)
    
    model_labels = [model_labels[i] for i in sorted_indices]
    llamaindex_scores = [llamaindex_scores[i] for i in sorted_indices]
    manual_scores = [manual_scores[i] for i in sorted_indices]
    
    # Set position of bars
    x = np.arange(len(model_labels))
    width = 0.35  # Width of bars
    
    # Create bars
    bars1 = ax.bar(x - width/2, llamaindex_scores, width, 
                   label='Few-Shot LlamaIndex', color='#5DADE2', alpha=0.9)
    bars2 = ax.bar(x + width/2, manual_scores, width, 
                   label='Few-Shot Manual', color='#EC7063', alpha=0.9)
    
    # Customize chart
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('EX Score', fontsize=13, fontweight='bold')
    ax.set_title('Exec Accuracy by Model', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 110)  # Set y-axis limit
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add value labels on bars (optional)
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=7, alpha=0.8)
    
    # Uncomment to add value labels
    # autolabel(bars1)
    # autolabel(bars2)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure1_execution_accuracy.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== FIGURE 2: ESM-EX Correlation (Scatter Plot) ==============

def plot_figure2_esm_ex_correlation(df_spider, output_dir):
    """Figure 2A: Scatter plot showing Exact Match vs Execution Accuracy correlation"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with different colors for approach
    for approach in df_spider['approach'].unique():
        df_app = df_spider[df_spider['approach'] == approach]
        marker = 'o' if 'LlamaIndex' in approach else '^'
        ax.scatter(df_app['exact_match_all'], df_app['exec_match_all'], 
                  s=150, alpha=0.7, label=approach, marker=marker)
    
    # Add model labels - ✅ Use model_name
    for _, row in df_spider.iterrows():
        ax.annotate(row['model_name'], 
                   (row['exact_match_all'], row['exec_match_all']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    # Diagonal line (perfect correlation)
    max_val = max(df_spider['exact_match_all'].max(), df_spider['exec_match_all'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect Correlation')
    
    # Calculate correlation
    correlation = df_spider['exact_match_all'].corr(df_spider['exec_match_all'])
    
    ax.set_xlabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 2A: Exact Match vs Execution Accuracy Correlation\n(r = {correlation:.3f})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure2_esm_ex_correlation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== FIGURE 3: Complexity Impact (Line/Bar Chart) ==============

def plot_figure3_complexity_impact(df_spider, output_dir):
    """Figure 3B: Show impact of query complexity on accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate average performance per difficulty
    difficulty_cols = ['exact_match_easy', 'exact_match_medium', 
                       'exact_match_hard', 'exact_match_extra']
    
    # Prepare data for each approach
    approaches = df_spider['approach'].unique()
    
    # Bar chart - Average by difficulty
    x_pos = np.arange(4)
    width = 0.35
    
    for i, approach in enumerate(approaches):
        df_app = df_spider[df_spider['approach'] == approach]
        means = [df_app[col].mean() for col in difficulty_cols]
        
        offset = width * (i - 0.5)
        ax1.bar(x_pos + offset, means, width, label=approach, alpha=0.8)
    
    ax1.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Exact Match (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 3B(i): Performance by Query Complexity', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Easy', 'Medium', 'Hard', 'Extra'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Line chart - Individual models - ✅ Use model_name
    for _, row in df_spider.iterrows():
        values = [row[col] for col in difficulty_cols]
        linestyle = '-' if 'LlamaIndex' in row['approach'] else '--'
        ax2.plot(['Easy', 'Medium', 'Hard', 'Extra'], values, 
                marker='o', label=f"{row['model_name']} ({row['approach'][:3]})",
                linestyle=linestyle, alpha=0.6)
    
    ax2.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 3B(ii): Model Performance Across Complexity', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure3_complexity_impact.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== FIGURE 4: Syntax Validity Comparison (Bar Chart) ==============

def plot_figure4_syntax_validity(df_syntax, output_dir):
    """Figure 4C: Compare syntax validity rates"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # ✅ Use model_name
    df_plot = df_syntax.sort_values('validity_rate', ascending=True)
    
    # Create bars
    colors = ['#2ecc71' if rate > 70 else '#f39c12' if rate > 50 else '#e74c3c' 
              for rate in df_plot['validity_rate']]
    
    bars = ax.barh(range(len(df_plot)), df_plot['validity_rate'], color=colors)
    
    # Labels
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{row['model_name']}\n({row['approach']})" 
                        for _, row in df_plot.iterrows()], fontsize=9)
    ax.set_xlabel('Syntax Validity Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4C: SQL Syntax Validity Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels and valid/invalid counts
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        ax.text(row['validity_rate'] + 1, i, 
               f"{row['validity_rate']:.1f}% ({int(row['valid_queries'])}/{int(row['total_queries'])})", 
               va='center', fontsize=8)
    
    # Add reference lines
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='50% threshold')
    ax.axvline(x=70, color='orange', linestyle='--', alpha=0.3, label='70% threshold')
    ax.axvline(x=90, color='green', linestyle='--', alpha=0.3, label='90% threshold')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure4_syntax_validity.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== FIGURE 5: Error Pattern Heatmap ==============

def plot_figure5_error_heatmap(df_syntax, output_dir):
    """Figure 5C: Heatmap showing error patterns"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Parse error_categories (it's a string representation of dict)
    error_types = []
    for _, row in df_syntax.iterrows():
        try:
            import ast
            errors = ast.literal_eval(row['error_categories'])
            error_types.extend(errors.keys())
        except:
            pass
    
    error_types = list(set(error_types))
    
    if not error_types:
        print("⚠️  No error types found, skipping heatmap")
        return
    
    # Create matrix
    matrix = []
    models = []
    
    # ✅ Use model_name
    for _, row in df_syntax.iterrows():
        models.append(f"{row['model_name']}\n({row['approach'][:3]})")
        error_counts = []
        
        try:
            import ast
            errors = ast.literal_eval(row['error_categories'])
            for error_type in error_types:
                count = errors.get(error_type, 0)
                # Normalize by total queries
                percentage = (count / row['total_queries']) * 100
                error_counts.append(percentage)
        except:
            error_counts = [0] * len(error_types)
        
        matrix.append(error_counts)
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                xticklabels=error_types, yticklabels=models,
                cbar_kws={'label': 'Error Rate (%)'}, ax=ax)
    
    ax.set_title('Figure 5C: Error Pattern Distribution Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Error Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure5_error_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== FIGURE 6: Latency Comparison (Bar Chart) ==============

def plot_figure6_latency_comparison(df_latency, output_dir):
    """Figure 6D: Compare average latency per model"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by mean latency
    df_plot = df_latency.sort_values('mean', ascending=True)
    
    # Create bars with color coding
    colors = ['#3498db' if app == 'LlamaIndex' else '#e74c3c' 
              for app in df_plot['approach']]
    
    bars = ax.barh(range(len(df_plot)), df_plot['mean'], color=colors)
    
    # Labels
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{row['model']}\n({row['approach']})" 
                        for _, row in df_plot.iterrows()], fontsize=9)
    ax.set_xlabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 6D: Average Query Latency Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        ax.text(row['mean'] + 0.1, i, f"{row['mean']:.3f}s", 
                va='center', fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='LlamaIndex'),
        Patch(facecolor='#e74c3c', label='Manual')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure6_latency_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== FIGURE 7: Latency Variability (Boxplot) ==============

def plot_figure7_latency_boxplot(df_metrics, output_dir):
    """Figure 7D: Boxplot showing latency variability"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort models by median latency
    model_order = df_metrics.groupby('model')['latency_seconds'].median().sort_values().index
    
    # Create boxplot
    bp = ax.boxplot([df_metrics[df_metrics['model'] == model]['latency_seconds'].values 
                      for model in model_order],
                     labels=model_order, vert=False, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color boxes
    for patch, model in zip(bp['boxes'], model_order):
        approach = df_metrics[df_metrics['model'] == model]['approach'].iloc[0]
        color = '#3498db' if approach == 'LlamaIndex' else '#e74c3c'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Figure 7D: Latency Distribution and Variability per Model', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure7_latency_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== FIGURE 8: Slow Queries Identification (Bar Chart) ==============

def plot_figure8_slow_queries(df_metrics, output_dir, top_n=20):
    """Figure 8D: Identify slowest queries across all models"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Find top N slowest queries (by average across models)
    slow_queries = df_metrics.groupby('question_id')['latency_seconds'].agg(['mean', 'max']).reset_index()
    slow_queries = slow_queries.sort_values('mean', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    y_pos = range(len(slow_queries))
    bars = ax.barh(y_pos, slow_queries['mean'], color='#e74c3c', alpha=0.7, label='Mean')
    
    # Add max latency markers
    ax.scatter(slow_queries['max'], y_pos, color='#c0392b', s=100, 
              marker='D', label='Max', zorder=5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Q{int(qid)}" for qid in slow_queries['question_id']], 
                       fontsize=9)
    ax.set_xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Question ID', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 8D: Top {top_n} Slowest Queries (Across All Models)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Add value labels
    for i, (idx, row) in enumerate(slow_queries.iterrows()):
        ax.text(row['mean'] + 0.2, i, f"{row['mean']:.2f}s", 
                va='center', fontsize=8)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'figure8_slow_queries.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ============== MAIN FUNCTION ==============

def main():
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Load all data
    print("Loading data...")
    df_spider = load_spider_results()
    df_syntax = load_syntax_validation()
    df_latency = load_latency_stats()
    df_metrics = load_metrics_details()
    
    print("\n" + "="*80)
    print("Generating figures...")
    print("="*80 + "\n")
    
    # Generate all figures
    if df_spider is not None:
        print("[1/8] Generating Figure 1: Execution Accuracy Comparison...")
        plot_figure1_execution_accuracy(df_spider, OUTPUT_DIR)
        
        print("[2/8] Generating Figure 2: ESM-EX Correlation...")
        plot_figure2_esm_ex_correlation(df_spider, OUTPUT_DIR)
        
        print("[3/8] Generating Figure 3: Complexity Impact...")
        plot_figure3_complexity_impact(df_spider, OUTPUT_DIR)
    else:
        print("⚠️  Skipping Figures 1-3: Spider results not found")
    
    if df_syntax is not None:
        print("[4/8] Generating Figure 4: Syntax Validity...")
        plot_figure4_syntax_validity(df_syntax, OUTPUT_DIR)
        
        print("[5/8] Generating Figure 5: Error Pattern Heatmap...")
        plot_figure5_error_heatmap(df_syntax, OUTPUT_DIR)
    else:
        print("⚠️  Skipping Figures 4-5: Syntax validation results not found")
    
    if df_latency is not None:
        print("[6/8] Generating Figure 6: Latency Comparison...")
        plot_figure6_latency_comparison(df_latency, OUTPUT_DIR)
    else:
        print("⚠️  Skipping Figure 6: Latency stats not found")
    
    if df_metrics is not None:
        print("[7/8] Generating Figure 7: Latency Boxplot...")
        plot_figure7_latency_boxplot(df_metrics, OUTPUT_DIR)
        
        print("[8/8] Generating Figure 8: Slow Queries...")
        plot_figure8_slow_queries(df_metrics, OUTPUT_DIR)
    else:
        print("⚠️  Skipping Figures 7-8: Detailed metrics not found")
    
    print("\n" + "="*80)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Print summary
    print("Generated Files:")
    for file in sorted(Path(OUTPUT_DIR).glob("figure*.png")):
        print(f"  ✓ {file.name}")
    print()

if __name__ == "__main__":
    main()