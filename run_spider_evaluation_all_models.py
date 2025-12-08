import os
import subprocess
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

EVALUATION_SCRIPT = "scripts/spider/evaluation.py"
GOLD_FILE = "database/gold_195_formatted.sql"
DATABASE_DIR = "data"
TABLE_FILE = "scripts/spider/evaluation_examples/examples/tables.json"

PREDICTION_DIRS = {
    "Few-Shot Manual": "output/Few Shot Output",
    "Few-Shot LlamaIndex": "output/Few Shot + Llamaindex Output"
}

RESULTS_DIR = "results"
ETYPE = "all"

os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_model_name(filename):
    """Extract model name from prediction filename"""
    if filename.startswith('prediksi_fewshot_'):
        match = re.search(r'prediksi_fewshot_(.+?)_\d{8}_\d{6}\.txt', filename)
    else:
        match = re.search(r'prediksi_(.+?)_\d{8}_\d{6}\.txt', filename)
    
    if match:
        return match.group(1)
    return filename.replace('.txt', '')

def extract_evaluation_metrics(output_text):
    """Extract metrics dari output evaluasi Spider - dengan partial results handling"""
    metrics = {
        'exact_match_easy': None,
        'exact_match_medium': None,
        'exact_match_hard': None,
        'exact_match_extra': None,
        'exact_match_all': None,
        'exec_match_easy': None,
        'exec_match_medium': None,
        'exec_match_hard': None,
        'exec_match_extra': None,
        'exec_match_all': None,
        'count_easy': None,
        'count_medium': None,
        'count_hard': None,
        'count_extra': None,
        'count_all': None,
        'evaluation_status': 'success'
    }
    
    # Check if evaluation crashed
    if 'Traceback' in output_text or 'IndexError' in output_text:
        metrics['evaluation_status'] = 'partial_crash'
    
    lines = output_text.split('\n')
    in_execution_section = False
    in_exact_match_section = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Detect sections
        if 'EXECUTION ACCURACY' in line.upper():
            in_execution_section = True
            in_exact_match_section = False
            continue
        elif 'EXACT MATCHING ACCURACY' in line.upper():
            in_execution_section = False
            in_exact_match_section = True
            continue
        elif 'PARTIAL MATCHING' in line.upper():
            in_execution_section = False
            in_exact_match_section = False
            continue
        
        # Parse count line
        if line.startswith('count'):
            parts = line.split()
            try:
                if len(parts) >= 6:
                    metrics['count_easy'] = int(parts[1])
                    metrics['count_medium'] = int(parts[2])
                    metrics['count_hard'] = int(parts[3])
                    metrics['count_extra'] = int(parts[4])
                    metrics['count_all'] = int(parts[5])
            except (ValueError, IndexError):
                pass
        
        # Parse execution accuracy
        if in_execution_section and line.startswith('execution'):
            parts = line.split()
            try:
                if len(parts) >= 6:
                    metrics['exec_match_easy'] = float(parts[1])
                    metrics['exec_match_medium'] = float(parts[2])
                    metrics['exec_match_hard'] = float(parts[3])
                    metrics['exec_match_extra'] = float(parts[4])
                    metrics['exec_match_all'] = float(parts[5])
            except (ValueError, IndexError):
                pass
        
        # Parse exact match
        if in_exact_match_section and 'exact' in line.lower() and 'match' in line.lower():
            parts = line.split()
            try:
                numbers = []
                for part in parts:
                    try:
                        numbers.append(float(part))
                    except ValueError:
                        continue
                
                if len(numbers) >= 5:
                    metrics['exact_match_easy'] = numbers[0]
                    metrics['exact_match_medium'] = numbers[1]
                    metrics['exact_match_hard'] = numbers[2]
                    metrics['exact_match_extra'] = numbers[3]
                    metrics['exact_match_all'] = numbers[4]
            except (ValueError, IndexError):
                pass
    
    # Convert to percentage (0-100 scale) if values are in 0-1 scale
    for key in metrics:
        if metrics[key] is not None and 'match' in key:
            if metrics[key] <= 1.0:
                metrics[key] = metrics[key] * 100
    
    return metrics

def run_evaluation(pred_file, model_name, approach):
    """Run Spider evaluation untuk satu model - dengan error recovery"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {approach} - {model_name}")
    print(f"File: {os.path.basename(pred_file)}")
    print(f"{'='*80}")
    
    cmd = [
        "python",
        EVALUATION_SCRIPT,
        "--gold", GOLD_FILE,
        "--pred", pred_file,
        "--db", DATABASE_DIR,
        "--table", TABLE_FILE,
        "--etype", ETYPE
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            errors='ignore' 
        )
        
        output_text = result.stdout + "\n\n" + result.stderr
        
        # Extract metrics (even if partially failed)
        metrics = extract_evaluation_metrics(output_text)
        
        # Add model info
        metrics['model_name'] = model_name
        metrics['approach'] = approach
        metrics['prediction_file'] = os.path.basename(pred_file)
        metrics['evaluation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Print summary
        print(f"\n[RESULTS] {approach} - {model_name}")
        
        if metrics['evaluation_status'] == 'partial_crash':
            print(f"  [WARNING] Evaluation partially crashed (some queries failed)")
        
        if metrics['exact_match_all'] is not None:
            print(f"  Exact Match (All): {metrics['exact_match_all']:.2f}%")
            print(f"  Exec Match (All):  {metrics['exec_match_all']:.2f}%")
            print(f"  Easy:   {metrics['exact_match_easy']:.2f}%")
            print(f"  Medium: {metrics['exact_match_medium']:.2f}%")
            print(f"  Hard:   {metrics['exact_match_hard']:.2f}%")
            print(f"  Extra:  {metrics['exact_match_extra']:.2f}%")
            print(f"  Total Questions: {metrics['count_all']}")
        else:
            print(f"  [ERROR] Failed to parse metrics")
            # Try to extract partial error info
            if 'eval_err_num:' in output_text:
                error_count = output_text.count('eval_err_num:')
                print(f"  Detected {error_count} evaluation errors before crash")
        
        # Save full output
        safe_model_name = model_name.replace(":", "_").replace("/", "_")
        safe_approach = approach.replace(" ", "_").replace("-", "_")
        log_file = os.path.join(RESULTS_DIR, f"evaluation_log_{safe_approach}_{safe_model_name}.txt")
        with open(log_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Approach: {approach}\n")
            f.write(f"Status: {metrics['evaluation_status']}\n")
            f.write(f"Timestamp: {metrics['evaluation_timestamp']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(output_text)
        print(f"  [SAVED] Log: {log_file}")
        
        return metrics
    
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Evaluation timeout for {model_name}")
        return None
    except Exception as e:
        print(f"  [ERROR] Failed to evaluate {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print(f"\n{'='*80}")
    print("SPIDER EVALUATION - ALL MODELS (MANUAL + LLAMAINDEX)")
    print(f"{'='*80}")
    
    for approach, pred_dir in PREDICTION_DIRS.items():
        print(f"\n{approach}: {pred_dir}")
    
    # Find all prediction files from both directories
    all_pred_files = []
    
    for approach, pred_dir in PREDICTION_DIRS.items():
        if not os.path.exists(pred_dir):
            print(f"\n[WARNING] Directory not found: {pred_dir}")
            continue
        
        pred_files = list(Path(pred_dir).glob("prediksi*.txt"))
        print(f"\nFound {len(pred_files)} files in {approach}:")
        for pf in pred_files:
            print(f"  - {pf.name}")
            all_pred_files.append({
                'file': str(pf),
                'approach': approach
            })
    
    if not all_pred_files:
        print(f"\n[ERROR] No prediction files found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Total files to evaluate: {len(all_pred_files)}")
    print(f"{'='*80}\n")
    
    # Run evaluation for each model
    all_metrics = []
    
    for item in all_pred_files:
        pred_file = item['file']
        approach = item['approach']
        model_name = extract_model_name(os.path.basename(pred_file))
        
        metrics = run_evaluation(pred_file, model_name, approach)
        
        if metrics:
            all_metrics.append(metrics)
    
    # Save to CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns
        column_order = [
            'approach',
            'model_name',
            'evaluation_status',
            'exact_match_all',
            'exec_match_all',
            'exact_match_easy',
            'exact_match_medium',
            'exact_match_hard',
            'exact_match_extra',
            'count_all',
            'count_easy',
            'count_medium',
            'count_hard',
            'count_extra',
            'prediction_file',
            'evaluation_timestamp'
        ]
        
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]
        
        # Sort by approach then exact_match_all
        df['exact_match_all'] = df['exact_match_all'].fillna(0)
        df = df.sort_values(['approach', 'exact_match_all'], ascending=[True, False])
        
        # Save CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = os.path.join(RESULTS_DIR, f"spider_evaluation_all_models_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*80}")
        print(f"[SUCCESS] Results saved to: {csv_file}")
        print(f"{'='*80}")
        
        # Print summary table by approach
        print("\n" + "="*120)
        print("EVALUATION SUMMARY BY APPROACH")
        print("="*120)
        
        for approach in PREDICTION_DIRS.keys():
            df_approach = df[df['approach'] == approach]
            if len(df_approach) > 0:
                print(f"\n### {approach} ###")
                display_cols = ['model_name', 'evaluation_status', 'exact_match_all', 'exec_match_all', 
                               'exact_match_easy', 'exact_match_medium', 
                               'exact_match_hard', 'exact_match_extra']
                display_cols = [col for col in display_cols if col in df_approach.columns]
                print(df_approach[display_cols].to_string(index=False))
        
        print("\n" + "="*120)
        
        # Print best model overall
        if df['exact_match_all'].max() > 0:
            best_model = df.loc[df['exact_match_all'].idxmax()]
            print(f"\nBEST MODEL OVERALL:")
            print(f"   Approach: {best_model['approach']}")
            print(f"   Model: {best_model['model_name']}")
            print(f"   Status: {best_model['evaluation_status']}")
            print(f"   Exact Match: {best_model['exact_match_all']:.2f}%")
            if best_model['exec_match_all'] is not None:
                print(f"   Exec Match:  {best_model['exec_match_all']:.2f}%")
        
        print("\n" + "="*120)
        
        # Comparison summary
        print("\nAPPROACH COMPARISON:")
        for approach in PREDICTION_DIRS.keys():
            df_approach = df[df['approach'] == approach]
            if len(df_approach) > 0:
                avg_exact = df_approach['exact_match_all'].mean()
                avg_exec = df_approach['exec_match_all'].mean()
                crashed = len(df_approach[df_approach['evaluation_status'] == 'partial_crash'])
                
                print(f"\n{approach}:")
                print(f"  Models evaluated: {len(df_approach)}")
                print(f"  Partial crashes: {crashed}")
                print(f"  Avg Exact Match: {avg_exact:.2f}%")
                print(f"  Avg Exec Match: {avg_exec:.2f}%")
                print(f"  Best: {df_approach['model_name'].iloc[0]} ({df_approach['exact_match_all'].iloc[0]:.2f}%)")
        
        print("\n" + "="*120 + "\n")
        
    else:
        print("\n[ERROR] No successful evaluations")

if __name__ == "__main__":
    main()