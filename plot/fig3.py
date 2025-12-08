import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============== LOAD DATA ==============
file_path = '../results/spider_evaluation_all_models_20251207_232503.csv'
df = pd.read_csv(file_path, delimiter=';')

# ============== CLEAN DATA ==============
def clean_score(val):
    """Clean malformed numeric values"""
    try:
        if isinstance(val, (int, float)):
            return float(val)
        
        s = str(val).replace('.', '')
        if not s: 
            return 0.0
        
        # Take first 4 digits and convert to percentage
        num = float(s[:4])
        
        # Adjust scale
        while num > 100:
            num /= 10
            
        return num
    except:
        return 0.0

# Clean columns
complexity_cols = ['exact_match_easy', 'exact_match_medium', 'exact_match_hard', 'exact_match_extra']
for col in complexity_cols:
    df[col] = df[col].apply(clean_score)

# Map approach names
approach_map = {
    'Few-Shot LlamaIndex': 'LlamaIndex',
    'Few-Shot Manual': 'Manual'
}
df['Method'] = df['approach'].map(approach_map)

# ============== CALCULATE AVERAGES ==============
# Group by approach and calculate mean for each complexity level
avg_by_complexity = df.groupby('Method')[complexity_cols].mean()

# Prepare data for plotting
complexity_labels = ['Easy', 'Medium', 'Hard', 'Extra']
llamaindex_scores = avg_by_complexity.loc['LlamaIndex'].values
manual_scores = avg_by_complexity.loc['Manual'].values

# ============== CREATE PLOT ==============
fig, ax = plt.subplots(figsize=(12, 8))

# Set bar positions
x = np.arange(len(complexity_labels))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, llamaindex_scores, width, 
               label='LlamaIndex', color='#5DADE2', alpha=0.9)
bars2 = ax.bar(x + width/2, manual_scores, width, 
               label='Manual', color='#EC7063', alpha=0.9)

# ============== STYLING ==============
ax.set_xlabel('Complexity', fontsize=13, fontweight='bold')
ax.set_ylabel('ESM Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Performance by Query Complexity', fontsize=16, fontweight='bold', pad=20)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels(complexity_labels, fontsize=11)

# Set y-axis limits
ax.set_ylim(0, 100)

# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, alpha=0.8)

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()

# ============== SAVE ==============
output_path = '../results/visualizations/figure3_complexity_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_path}")

# ============== PRINT STATISTICS ==============
print("\n" + "="*60)
print("PERFORMANCE BY QUERY COMPLEXITY (Average ESM)")
print("="*60)
print(f"{'Complexity':<12} {'LlamaIndex':>12} {'Manual':>12} {'Difference':>12}")
print("-"*60)
for i, label in enumerate(complexity_labels):
    diff = llamaindex_scores[i] - manual_scores[i]
    print(f"{label:<12} {llamaindex_scores[i]:>12.2f}% {manual_scores[i]:>12.2f}% {diff:>+12.2f}%")
print("="*60)

# Calculate overall trends
print(f"\nTrend Analysis:")
print(f"LlamaIndex: Easy→Extra drop = {llamaindex_scores[0] - llamaindex_scores[-1]:.1f}%")
print(f"Manual:     Easy→Extra drop = {manual_scores[0] - manual_scores[-1]:.1f}%")
print(f"\nAverage across all complexities:")
print(f"LlamaIndex: {np.mean(llamaindex_scores):.2f}%")
print(f"Manual:     {np.mean(manual_scores):.2f}%")

plt.show()