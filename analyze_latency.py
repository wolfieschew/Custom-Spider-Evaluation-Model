"""
filepath: d:\Project Magang Bagas\Eval_Spider\analyze_latency.py
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

METRICS_DIR_FEWSHOT = "output/Few Shot + Llamaindex Output"
METRICS_DIR_MANUAL = "output/Few Shot Output"
OUTPUT_DIR = "results/latency_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_metrics(directory):
    """Load semua file metric CSV dari directory"""
    metrics_files = list(Path(directory).glob("metric_*.csv"))
    all_data = []
    
    for file in metrics_files:
        try:
            try:
                df = pd.read_csv(file, sep=';')
            except:
                df = pd.read_csv(file, sep=',')
            
            # Extract model name dari filename
            filename = file.stem
            if 'fewshot_llamaindex' in filename:
                model_name = filename.replace('metric_fewshot_llamaindex_', '').replace('metric_', '')
                approach = 'LlamaIndex'
            elif 'fewshot' in filename:
                model_name = filename.replace('metric_fewshot_', '').replace('metric_', '')
                approach = 'LlamaIndex'
            else:
                model_name = filename.replace('metric_', '')
                approach = 'Manual'
            
            # Remove timestamp dari model name
            model_name = '_'.join(model_name.split('_')[:-2])
            
            df['model'] = model_name
            df['approach'] = approach
            df['source_file'] = file.name
            
            all_data.append(df)
            print(f"✓ Loaded: {file.name} ({len(df)} queries, model: {model_name})")
            
        except Exception as e:
            print(f"✗ Error loading {file.name}: {e}")
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def calculate_latency_stats(df):
    """Calculate latency statistics per model"""
    stats = df.groupby(['model', 'approach'])['latency_seconds'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('p25', lambda x: np.percentile(x, 25)),
        ('p75', lambda x: np.percentile(x, 75)),
        ('p95', lambda x: np.percentile(x, 95))
    ]).reset_index()
    
    stats = stats.round(4)
    return stats

def plot_latency_boxplot(df, output_dir):
    """Create boxplot comparison"""
    plt.figure(figsize=(14, 8))
    
    # Sort by median latency
    model_order = df.groupby('model')['latency_seconds'].median().sort_values().index
    
    sns.boxplot(data=df, x='model', y='latency_seconds', order=model_order)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.title('Latency Distribution per Model', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'latency_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_latency_comparison(stats, output_dir):
    """Create bar chart comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Sort by mean latency
    stats_sorted = stats.sort_values('mean')
    
    # Mean latency
    ax1 = axes[0, 0]
    bars = ax1.barh(stats_sorted['model'], stats_sorted['mean'])
    ax1.set_xlabel('Mean Latency (seconds)', fontsize=11)
    ax1.set_title('Average Latency per Model', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Color bars by approach
    for i, approach in enumerate(stats_sorted['approach']):
        if approach == 'LlamaIndex':
            bars[i].set_color('#3498db')
        else:
            bars[i].set_color('#e74c3c')
    
    # Median latency
    ax2 = axes[0, 1]
    bars = ax2.barh(stats_sorted['model'], stats_sorted['median'])
    ax2.set_xlabel('Median Latency (seconds)', fontsize=11)
    ax2.set_title('Median Latency per Model', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, approach in enumerate(stats_sorted['approach']):
        if approach == 'LlamaIndex':
            bars[i].set_color('#3498db')
        else:
            bars[i].set_color('#e74c3c')
    
    # Standard deviation
    ax3 = axes[1, 0]
    bars = ax3.barh(stats_sorted['model'], stats_sorted['std'])
    ax3.set_xlabel('Standard Deviation (seconds)', fontsize=11)
    ax3.set_title('Latency Variability per Model', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    for i, approach in enumerate(stats_sorted['approach']):
        if approach == 'LlamaIndex':
            bars[i].set_color('#3498db')
        else:
            bars[i].set_color('#e74c3c')
    
    # Query count
    ax4 = axes[1, 1]
    bars = ax4.barh(stats_sorted['model'], stats_sorted['count'])
    ax4.set_xlabel('Number of Queries', fontsize=11)
    ax4.set_title('Query Count per Model', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    for i, approach in enumerate(stats_sorted['approach']):
        if approach == 'LlamaIndex':
            bars[i].set_color('#3498db')
        else:
            bars[i].set_color('#e74c3c')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='LlamaIndex'),
        Patch(facecolor='#e74c3c', label='Manual')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_latency_by_question(df, output_dir, top_n=20):
    """Plot latency distribution by question"""
    # Get average latency per question across all models
    question_latency = df.groupby('question_id')['latency_seconds'].agg(['mean', 'std']).reset_index()
    question_latency = question_latency.sort_values('mean', ascending=False).head(top_n)
    
    plt.figure(figsize=(14, 8))
    plt.barh(range(len(question_latency)), question_latency['mean'])
    plt.yticks(range(len(question_latency)), [f"Q{int(qid)}" for qid in question_latency['question_id']])
    plt.xlabel('Average Latency (seconds)', fontsize=12)
    plt.ylabel('Question ID', fontsize=12)
    plt.title(f'Top {top_n} Questions by Average Latency', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'latency_by_question.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def generate_latency_report(stats, df, output_dir):
    """Generate detailed latency report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'latency_report_{timestamp}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LATENCY ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models: {len(stats)}\n")
        f.write(f"Total Queries: {len(df)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write(f"Overall Average Latency: {df['latency_seconds'].mean():.4f} seconds\n")
        f.write(f"Overall Median Latency:  {df['latency_seconds'].median():.4f} seconds\n")
        f.write(f"Overall Std Dev:         {df['latency_seconds'].std():.4f} seconds\n")
        f.write(f"Overall Min Latency:     {df['latency_seconds'].min():.4f} seconds\n")
        f.write(f"Overall Max Latency:     {df['latency_seconds'].max():.4f} seconds\n\n")
        
        f.write("="*80 + "\n")
        f.write("PER-MODEL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        # Sort by mean latency
        stats_sorted = stats.sort_values('mean')
        
        for idx, row in stats_sorted.iterrows():
            f.write(f"Model: {row['model']} ({row['approach']})\n")
            f.write(f"{'─'*80}\n")
            f.write(f"  Queries:      {int(row['count'])}\n")
            f.write(f"  Mean:         {row['mean']:.4f} seconds\n")
            f.write(f"  Median:       {row['median']:.4f} seconds\n")
            f.write(f"  Std Dev:      {row['std']:.4f} seconds\n")
            f.write(f"  Min:          {row['min']:.4f} seconds\n")
            f.write(f"  Max:          {row['max']:.4f} seconds\n")
            f.write(f"  25th %ile:    {row['p25']:.4f} seconds\n")
            f.write(f"  75th %ile:    {row['p75']:.4f} seconds\n")
            f.write(f"  95th %ile:    {row['p95']:.4f} seconds\n\n")
        
        f.write("="*80 + "\n")
        f.write("RANKINGS\n")
        f.write("="*80 + "\n\n")
        
        f.write("FASTEST MODELS (by mean latency):\n")
        for idx, (i, row) in enumerate(stats_sorted.head(5).iterrows(), 1):
            f.write(f"  {idx}. {row['model']:30s} - {row['mean']:8.4f}s ({row['approach']})\n")
        
        f.write("\nSLOWEST MODELS (by mean latency):\n")
        for idx, (i, row) in enumerate(stats_sorted.tail(5).iloc[::-1].iterrows(), 1):
            f.write(f"  {idx}. {row['model']:30s} - {row['mean']:8.4f}s ({row['approach']})\n")
        
        f.write("\nMOST CONSISTENT (lowest std dev):\n")
        stats_by_std = stats.sort_values('std')
        for idx, (i, row) in enumerate(stats_by_std.head(5).iterrows(), 1):
            f.write(f"  {idx}. {row['model']:30s} - σ={row['std']:8.4f}s ({row['approach']})\n")
        
    print(f"✓ Saved: {report_file}")
    return report_file

def main():
    print("\n" + "="*80)
    print("LATENCY ANALYSIS")
    print("="*80 + "\n")
    
    # Load data from both directories
    print("Loading metrics from Few Shot + LlamaIndex...")
    df_llamaindex = load_all_metrics(METRICS_DIR_FEWSHOT)
    
    print("\nLoading metrics from Few Shot Manual...")
    df_manual = load_all_metrics(METRICS_DIR_MANUAL)
    
    # Combine data
    if df_llamaindex is not None and df_manual is not None:
        df = pd.concat([df_llamaindex, df_manual], ignore_index=True)
    elif df_llamaindex is not None:
        df = df_llamaindex
    elif df_manual is not None:
        df = df_manual
    else:
        print("\nNo data found!")
        return
    
    print(f"\n✓ Total loaded: {len(df)} queries from {df['model'].nunique()} models\n")
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_latency_stats(df)
    
    # Save statistics to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stats_file = os.path.join(OUTPUT_DIR, f'latency_statistics_{timestamp}.csv')
    stats.to_csv(stats_file, index=False)
    print(f"✓ Saved: {stats_file}\n")
    
    # Print summary to console
    print("="*80)
    print("LATENCY STATISTICS SUMMARY")
    print("="*80)
    print(stats.to_string(index=False))
    print("="*80 + "\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_latency_boxplot(df, OUTPUT_DIR)
    plot_latency_comparison(stats, OUTPUT_DIR)
    plot_latency_by_question(df, OUTPUT_DIR)
    
    # Generate report
    print("\nGenerating detailed report...")
    report_file = generate_latency_report(stats, df, OUTPUT_DIR)
    
    # Print fastest and slowest models
    print("\n" + "="*80)
    print("TOP 5 FASTEST MODELS")
    print("="*80)
    top5 = stats.nsmallest(5, 'mean')
    for idx, row in top5.iterrows():
        print(f"  {row['model']:30s} - {row['mean']:8.4f}s (median: {row['median']:7.4f}s) [{row['approach']}]")
    
    print("\n" + "="*80)
    print("TOP 5 SLOWEST MODELS")
    print("="*80)
    bottom5 = stats.nlargest(5, 'mean')
    for idx, row in bottom5.iterrows():
        print(f"  {row['model']:30s} - {row['mean']:8.4f}s (median: {row['median']:7.4f}s) [{row['approach']}]")
    
    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()