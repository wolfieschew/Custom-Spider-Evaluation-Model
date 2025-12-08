""""
filepath: d:\Project Magang Bagas\Eval_Spider\figure1_execution_accuracy.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============== KONFIGURASI ==============
RESULTS_DIR = "results"
OUTPUT_DIR = "results/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# ============== LOAD DATA ==============
def load_spider_results():
    """Load Spider evaluation results"""
    files = list(Path(RESULTS_DIR).glob("spider_evaluation_all_models_*.csv"))
    if files:
        latest = max(files, key=os.path.getctime)
        df = pd.read_csv(latest, sep=';')
        
        # ✅ Convert numeric columns - handle malformed numbers
        numeric_cols = ['exact_match_all', 'exec_match_all', 'exact_match_easy', 
                       'exact_match_medium', 'exact_match_hard', 'exact_match_extra']
        
        for col in numeric_cols:
            if col in df.columns:
                # Remove dots used as thousand separators (keep only last dot as decimal)
                df[col] = df[col].astype(str).apply(lambda x: x.replace('.', '', x.count('.')-1) if x.count('.') > 1 else x)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Loaded: {latest.name}")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique models: {df['model_name'].nunique()}")
        print(f"  Approaches: {df['approach'].unique()}")
        
        # ✅ Debug: Show sample data
        print("\n📊 Sample Data:")
        print(df[['model_name', 'approach', 'exec_match_all']].head(10))
        
        return df
    print("❌ Spider results not found!")
    return None

# ============== PLOT FUNCTION ==============
def plot_execution_accuracy(df_spider):
    """Grouped bar chart comparing Execution Accuracy per model"""
    
    # ✅ Check for NaN values
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    print(f"Total rows: {len(df_spider)}")
    print(f"NaN in exec_match_all: {df_spider['exec_match_all'].isna().sum()}")
    print(f"Valid exec_match_all values: {df_spider['exec_match_all'].notna().sum()}")
    
    # ✅ Drop rows with NaN exec_match_all
    df_spider = df_spider.dropna(subset=['exec_match_all'])
    
    if len(df_spider) == 0:
        print("❌ No valid data after removing NaN!")
        return
    
    # Debug: Print data structure
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    for model in df_spider['model_name'].unique():
        df_model = df_spider[df_spider['model_name'] == model]
        print(f"\n{model}:")
        for approach in df_model['approach'].unique():
            df_app = df_model[df_model['approach'] == approach]
            if len(df_app) > 0:
                score = df_app['exec_match_all'].iloc[0]
                print(f"  - {approach}: {score:.2f}")
    print("="*60 + "\n")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get ALL unique models from BOTH approaches
    all_models = set()
    all_models.update(df_spider[df_spider['approach'] == 'Few-Shot LlamaIndex']['model_name'].unique())
    all_models.update(df_spider[df_spider['approach'] == 'Few-Shot Manual']['model_name'].unique())
    all_models = list(all_models)
    
    print(f"Found {len(all_models)} unique models:")
    for m in sorted(all_models):
        print(f"  - {m}")
    print()
    
    # Create data structure for grouped bars
    llamaindex_scores = []
    manual_scores = []
    model_labels = []
    
    for model in all_models:
        df_model = df_spider[df_spider['model_name'] == model]
        
        # Get LlamaIndex score
        llamaindex_row = df_model[df_model['approach'] == 'Few-Shot LlamaIndex']
        if len(llamaindex_row) > 0:
            llamaindex_score = llamaindex_row['exec_match_all'].iloc[0]
        else:
            llamaindex_score = np.nan
        
        # Get Manual score
        manual_row = df_model[df_model['approach'] == 'Few-Shot Manual']
        if len(manual_row) > 0:
            manual_score = manual_row['exec_match_all'].iloc[0]
        else:
            manual_score = np.nan
        
        model_labels.append(model)
        llamaindex_scores.append(llamaindex_score)
        manual_scores.append(manual_score)
    
    # Sort by average score (descending), handling NaN values
    avg_scores = []
    for l, m in zip(llamaindex_scores, manual_scores):
        if pd.isna(l) and pd.isna(m):
            avg_scores.append(0)
        elif pd.isna(l):
            avg_scores.append(m)
        elif pd.isna(m):
            avg_scores.append(l)
        else:
            avg_scores.append((l + m) / 2)
    
    sorted_indices = sorted(range(len(avg_scores)), key=lambda i: avg_scores[i], reverse=True)
    
    model_labels = [model_labels[i] for i in sorted_indices]
    llamaindex_scores = [llamaindex_scores[i] for i in sorted_indices]
    manual_scores = [manual_scores[i] for i in sorted_indices]
    
    # Convert NaN to 0 for plotting
    llamaindex_scores_plot = [0 if pd.isna(x) else x for x in llamaindex_scores]
    manual_scores_plot = [0 if pd.isna(x) else x for x in manual_scores]
    
    # Set position of bars
    x = np.arange(len(model_labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, llamaindex_scores_plot, width, 
                   label='Few-Shot LI', color='#5DADE2', alpha=0.9)
    bars2 = ax.bar(x + width/2, manual_scores_plot, width, 
                   label='Few-Shot Man', color='#EC7063', alpha=0.9)
    
    # Add hatching for missing data
    for i, (lli, man) in enumerate(zip(llamaindex_scores, manual_scores)):
        if pd.isna(lli):
            ax.bar(x[i] - width/2, 2, width, color='lightgray', alpha=0.3, hatch='///')
        if pd.isna(man):
            ax.bar(x[i] + width/2, 2, width, color='lightgray', alpha=0.3, hatch='///')
    
    # Customize chart
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('EX Score', fontsize=13, fontweight='bold')
    ax.set_title('Exec Accuracy by Model', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 110)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5DADE2', label='Few-Shot LI', alpha=0.9),
        Patch(facecolor='#EC7063', label='Few-Shot Man', alpha=0.9),
        Patch(facecolor='lightgray', label='No Data', alpha=0.3, hatch='///')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add value labels
    def autolabel(bars, scores):
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            if height > 0 and not pd.isna(score):
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=7, alpha=0.8)
    
    autolabel(bars1, llamaindex_scores)
    autolabel(bars2, manual_scores)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'figure1_execution_accuracy.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total models plotted: {len(model_labels)}")
    print(f"Models with LlamaIndex data: {sum(1 for x in llamaindex_scores if not pd.isna(x))}")
    print(f"Models with Manual data: {sum(1 for x in manual_scores if not pd.isna(x))}")
    print(f"Models with both: {sum(1 for l, m in zip(llamaindex_scores, manual_scores) if not pd.isna(l) and not pd.isna(m))}")
    print("="*60 + "\n")

# ============== MAIN ==============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FIGURE 1: EXECUTION ACCURACY COMPARISON")
    print("="*60 + "\n")
    
    df = load_spider_results()
    if df is not None:
        plot_execution_accuracy(df)
        print("\n✅ Figure 1 generated successfully!\n")
    else:
        print("\n❌ Failed to generate Figure 1\n")