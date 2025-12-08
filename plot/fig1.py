import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '../results/spider_evaluation_all_models_20251207_232503.csv'
df = pd.read_csv(file_path, delimiter=';')

def clean_score(val):
    try:
        if isinstance(val, (int, float)):
            return float(val)
        
        s = str(val).replace('.', '')
        if not s: return 0.0
    
        num = float(s[:4])
        
        while num > 80:
            num /= 10
            
        return num
    except:
        return 0.0

df['score'] = df['exec_match_all'].apply(clean_score)

approach_map = {
    'Few-Shot LlamaIndex': 'Few-Shot Llamaindex',
    'Few-Shot Manual': 'Few-Shot Manual'
}
df['Method'] = df['approach'].map(approach_map)

df_pivot = df.pivot(index='model_name', columns='Method', values='score')

df_pivot = df_pivot.sort_values('Few-Shot Llamaindex', ascending=False)

df_plot = df_pivot.reset_index().melt(id_vars='model_name', var_name='Method', value_name='Score')

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

custom_colors = {"Few-Shot Llamaindex": "#17becf", "Few-Shot Manual": "#d62728"}

ax = sns.barplot(
    data=df_plot,
    x='model_name',
    y='Score',
    hue='Method',
    palette=custom_colors,
    edgecolor="white",
    width=0.8
)

plt.title('Exec Accuracy by Model', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('EX Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0, 100) 

plt.xticks(rotation=-30, ha='left', fontsize=10)

plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)

plt.figtext(0.1, -0.05, 
            "Figure 1: Execution Accuracy (EX) Comparison between Few-Shot LlamaIndex and Few-Shot Manual\napproaches across 10 language models", 
            ha="left", fontsize=10, wrap=True)

plt.tight_layout()
plt.show()