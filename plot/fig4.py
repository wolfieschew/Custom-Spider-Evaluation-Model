import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '../results/syntax_validation_summary_20251207_233213.csv'
df = pd.read_csv(file_path, delimiter=';')

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

# Clean validity_rate and convert to actual numbers
df['validity_rate'] = df['validity_rate'].apply(clean_score)
df['valid_queries'] = pd.to_numeric(df['valid_queries'], errors='coerce')
df['total_queries'] = pd.to_numeric(df['total_queries'], errors='coerce')
df['invalid_queries'] = df['total_queries'] - df['valid_queries']

# Map approach names
approach_map = {
    'Few-Shot LlamaIndex': 'LI',
    'Few-Shot Manual': 'Man'
}
df['Method'] = df['approach'].map(approach_map)

# ============== PREPARE DATA ==============
# Get unique models
models = df['model_name'].unique()
models_sorted = sorted(models, key=lambda x: df[df['model_name'] == x]['validity_rate'].mean(), reverse=True)

# Prepare data for plotting
data_for_plot = []

for model in models_sorted:
    df_model = df[df['model_name'] == model]
    
    # LlamaIndex
    li_row = df_model[df_model['Method'] == 'LI']
    if len(li_row) > 0:
        data_for_plot.append({
            'model': model,
            'method': 'LI',
            'valid': int(li_row['valid_queries'].iloc[0]),
            'invalid': int(li_row['invalid_queries'].iloc[0])
        })
    else:
        data_for_plot.append({
            'model': model,
            'method': 'LI',
            'valid': 0,
            'invalid': 0
        })
    
    # Manual
    man_row = df_model[df_model['Method'] == 'Man']
    if len(man_row) > 0:
        data_for_plot.append({
            'model': model,
            'method': 'Man',
            'valid': int(man_row['valid_queries'].iloc[0]),
            'invalid': int(man_row['invalid_queries'].iloc[0])
        })
    else:
        data_for_plot.append({
            'model': model,
            'method': 'Man',
            'valid': 0,
            'invalid': 0
        })

df_plot = pd.DataFrame(data_for_plot)

# ============== CREATE PLOT ==============
fig, ax = plt.subplots(figsize=(14, 8))

# Set up bar positions
models_display = [m.replace('_', '.').replace('.', ' ').title() for m in models_sorted]
x = np.arange(len(models_sorted))
width = 0.35

# Colors
colors_valid = {'LI': '#2d7f47', 'Man': '#6db087'}  # Dark & light green
colors_invalid = {'LI': '#c94c4c', 'Man': '#e89999'}  # Dark & light red

# Plot stacked bars for each model
for i, model in enumerate(models_sorted):
    df_model_plot = df_plot[df_plot['model'] == model]
    
    # LlamaIndex (left bar)
    li_data = df_model_plot[df_model_plot['method'] == 'LI']
    if len(li_data) > 0:
        valid_li = li_data['valid'].iloc[0]
        invalid_li = li_data['invalid'].iloc[0]
        
        # Valid part
        ax.bar(x[i] - width/2, valid_li, width, 
               color=colors_valid['LI'], label='Valid (LI)' if i == 0 else '')
        # Invalid part (stacked on top)
        ax.bar(x[i] - width/2, invalid_li, width, bottom=valid_li,
               color=colors_invalid['LI'], label='Invalid (LI)' if i == 0 else '')
        
        # Add text labels
        ax.text(x[i] - width/2, valid_li/2, str(valid_li), 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        if invalid_li > 5:  # Only show if visible
            ax.text(x[i] - width/2, valid_li + invalid_li/2, str(invalid_li), 
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Manual (right bar)
    man_data = df_model_plot[df_model_plot['method'] == 'Man']
    if len(man_data) > 0:
        valid_man = man_data['valid'].iloc[0]
        invalid_man = man_data['invalid'].iloc[0]
        
        # Valid part
        ax.bar(x[i] + width/2, valid_man, width,
               color=colors_valid['Man'], label='Valid (Man)' if i == 0 else '')
        # Invalid part (stacked on top)
        ax.bar(x[i] + width/2, invalid_man, width, bottom=valid_man,
               color=colors_invalid['Man'], label='Invalid (Man)' if i == 0 else '')
        
        # Add text labels
        ax.text(x[i] + width/2, valid_man/2, str(valid_man), 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        if invalid_man > 5:  # Only show if visible
            ax.text(x[i] + width/2, valid_man + invalid_man/2, str(invalid_man), 
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# ============== STYLING ==============
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Valid Queries', fontsize=13, fontweight='bold')
ax.set_title('SQL Syntax Validity by Model', fontsize=16, fontweight='bold', pad=20)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels(models_display, rotation=0, ha='center', fontsize=10)

# Set y-axis
ax.set_ylim(0, 200)
ax.set_yticks(range(0, 201, 20))

# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)

# Add reference line at 195
ax.axhline(y=195, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(len(models_sorted)-0.5, 196, 'Total: 195 queries', 
        fontsize=9, color='gray', ha='right')

plt.tight_layout()

# ============== SAVE ==============
output_path = '../results/visualizations/figure4_syntax_validity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_path}")

# ============== PRINT STATISTICS ==============
print("\n" + "="*70)
print("SQL SYNTAX VALIDITY COMPARISON")
print("="*70)
print(f"{'Model':<15} {'LI Valid':>10} {'LI Invalid':>12} {'Man Valid':>11} {'Man Invalid':>13}")
print("-"*70)

for model in models_sorted:
    df_model_plot = df_plot[df_plot['model'] == model]
    
    li_data = df_model_plot[df_model_plot['method'] == 'LI']
    man_data = df_model_plot[df_model_plot['method'] == 'Man']
    
    li_valid = int(li_data['valid'].iloc[0]) if len(li_data) > 0 else 0
    li_invalid = int(li_data['invalid'].iloc[0]) if len(li_data) > 0 else 0
    man_valid = int(man_data['valid'].iloc[0]) if len(man_data) > 0 else 0
    man_invalid = int(man_data['invalid'].iloc[0]) if len(man_data) > 0 else 0
    
    print(f"{model:<15} {li_valid:>10} {li_invalid:>12} {man_valid:>11} {man_invalid:>13}")

print("="*70)

# Overall statistics
total_li_valid = df[df['Method'] == 'LI']['valid_queries'].sum()
total_li_invalid = df[df['Method'] == 'LI']['invalid_queries'].sum()
total_man_valid = df[df['Method'] == 'Man']['valid_queries'].sum()
total_man_invalid = df[df['Method'] == 'Man']['invalid_queries'].sum()

print(f"\nOverall:")
print(f"LlamaIndex:  {int(total_li_valid)} valid / {int(total_li_invalid)} invalid ({total_li_valid/(total_li_valid+total_li_invalid)*100:.1f}% valid)")
print(f"Manual:      {int(total_man_valid)} valid / {int(total_man_invalid)} invalid ({total_man_valid/(total_man_valid+total_man_invalid)*100:.1f}% valid)")

plt.show()