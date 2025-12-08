import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
        
        num = float(s[:4])
        
        while num > 100:
            num /= 10
            
        return num
    except:
        return 0.0

df['esm'] = df['exact_match_all'].apply(clean_score)
df['ex'] = df['exec_match_all'].apply(clean_score)

approach_map = {
    'Few-Shot LlamaIndex': 'LlamaIndex',
    'Few-Shot Manual': 'Manual'
}
df['Method'] = df['approach'].map(approach_map)

# ============== CALCULATE STATISTICS ==============
correlation = df['esm'].corr(df['ex'])
slope, intercept, r_value, p_value, std_err = stats.linregress(df['esm'], df['ex'])

# ============== CREATE PLOT ==============
fig, ax = plt.subplots(figsize=(12, 8))

colors = {'LlamaIndex': '#5DADE2', 'Manual': '#EC7063'}

df['bubble_size'] = ((df['esm'] + df['ex']) / 2) * 10

# Plot scatter for each method
for method in df['Method'].unique():
    df_method = df[df['Method'] == method]
    ax.scatter(
        df_method['esm'], 
        df_method['ex'],
        s=df_method['bubble_size'] * 10,
        c=colors[method],
        alpha=0.6,
        edgecolors='white',
        linewidth=2,
        label=method
    )

# ✅ FIX: Create continuous trend line with more points
x_trend = np.linspace(0, 75, 100)  # 100 points from 0 to 75
y_trend = slope * x_trend + intercept
ax.plot(x_trend, y_trend, 'k--', linewidth=2, alpha=0.5, label='Trend')

# ============== STYLING ==============
ax.set_xlabel('ESM (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('EX (%)', fontsize=13, fontweight='bold')
ax.set_title('ESM vs EX Accuracy', fontsize=16, fontweight='bold', pad=20)

ax.set_xlim(0, 75)
ax.set_ylim(0, 75)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Correlation text
ax.text(0.98, 0.02, f'Correlation: r = {correlation:.3f}', 
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# ============== SAVE ==============
output_path = '../results/visualizations/figure2_esm_ex_correlation.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_path}")

plt.show()