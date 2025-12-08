import os
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

# ============== KONFIGURASI ==============
DATABASE_DIR = "data"
OUTPUT_ROOT = "output"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============== SYNTAX VALIDATION FUNCTIONS ==============

def validate_sql_syntax(sql_query, db_path):
    """
    Validate SQL syntax by trying to parse it with SQLite
    Returns: (is_valid, error_category, error_message)
    """
    # Basic checks
    if not sql_query or sql_query.strip() == '':
        return False, "empty", "Empty query"
    
    sql_lower = sql_query.lower().strip()
    
    # Check for fallback query
    if sql_lower == 'select 1':
        return False, "fallback", "Fallback query (SELECT 1)"
    
    # Check for dangerous keywords (should not be in text-to-sql)
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'INSERT', 'UPDATE', 'CREATE']
    for keyword in dangerous_keywords:
        if keyword in sql_query.upper():
            return False, "dangerous", f"Dangerous keyword: {keyword}"
    
    # Check if starts with SELECT (basic requirement)
    if not sql_lower.startswith('select'):
        return False, "invalid_start", "Query does not start with SELECT"
    
    # Try to parse with SQLite
    if not os.path.exists(db_path):
        return False, "no_database", f"Database not found: {db_path}"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Use EXPLAIN to parse without executing
        cursor.execute(f"EXPLAIN {sql_query}")
        
        conn.close()
        return True, "valid", None
    
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        
        # Categorize errors
        if 'syntax error' in error_msg:
            return False, "syntax_error", str(e)
        elif 'no such table' in error_msg:
            return False, "table_error", str(e)
        elif 'no such column' in error_msg:
            return False, "column_error", str(e)
        elif 'ambiguous column' in error_msg:
            return False, "ambiguous_column", str(e)
        else:
            return False, "operational_error", str(e)
    
    except sqlite3.DatabaseError as e:
        return False, "database_error", str(e)
    
    except Exception as e:
        return False, "unknown_error", str(e)


def extract_model_name(filename):
    """Extract model name from prediction filename"""
    if filename.startswith('prediksi_fewshot_'):
        match = re.search(r'prediksi_fewshot_(.+?)_\d{8}_\d{6}\.txt', filename)
    else:
        match = re.search(r'prediksi_(.+?)_\d{8}_\d{6}\.txt', filename)
    
    if match:
        return match.group(1)
    return filename.replace('.txt', '')


def detect_approach_from_path(file_path):
    """Detect approach from file path"""
    path_lower = file_path.lower()
    filename_lower = Path(file_path).name.lower()
    
    if 'llamaindex' in path_lower or 'fewshot' in filename_lower:
        return "Few-Shot LlamaIndex"
    elif 'few shot output' in path_lower:
        return "Few-Shot Manual"
    else:
        return "Direct Output"


def validate_prediction_file(pred_file, approach, model_name):
    """
    Validate all predictions in a file
    Returns summary statistics
    """
    print(f"\n{'='*80}")
    print(f"Validating: {approach} - {model_name}")
    print(f"File: {os.path.basename(pred_file)}")
    print(f"{'='*80}")
    
    results = []
    total_queries = 0
    valid_queries = 0
    error_categories = {}
    
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                
                if len(parts) >= 2:
                    total_queries += 1
                    sql_query = parts[0]
                    db_id = parts[1]
                    
                    # Get database path
                    db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
                    
                    # Validate syntax
                    is_valid, error_category, error_msg = validate_sql_syntax(sql_query, db_path)
                    
                    if is_valid:
                        valid_queries += 1
                    else:
                        error_categories[error_category] = error_categories.get(error_category, 0) + 1
                    
                    results.append({
                        'approach': approach,
                        'model_name': model_name,
                        'line': line_num,
                        'db_id': db_id,
                        'sql_length': len(sql_query),
                        'is_valid': is_valid,
                        'error_category': error_category,
                        'error_message': error_msg if error_msg else 'OK'
                    })
        
        # ✅ Calculate validity rate properly
        if total_queries > 0:
            validity_rate = round((valid_queries / total_queries) * 100, 2)
        else:
            validity_rate = 0.0
        
        # Print summary
        print(f"\n[RESULTS]")
        print(f"  Total Queries: {total_queries}")
        print(f"  Valid: {valid_queries} ({validity_rate:.2f}%)")
        print(f"  Invalid: {total_queries - valid_queries} ({100-validity_rate:.2f}%)")
        
        if error_categories:
            print(f"\n  Error Breakdown:")
            for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_queries * 100)
                print(f"    - {category}: {count} ({percentage:.1f}%)")
        
        return results, {
            'approach': approach,
            'model_name': model_name,
            'total_queries': total_queries,
            'valid_queries': valid_queries,
            'validity_rate': validity_rate,  # ✅ Now properly formatted as float
            'error_categories': str(error_categories)  # ✅ Convert dict to string for CSV
        }
    
    except Exception as e:
        print(f"  [ERROR] Failed to process file: {e}")
        return [], None


def main():
    print(f"\n{'='*100}")
    print("SQL SYNTAX VALIDATION - ALL MODELS")
    print(f"{'='*100}")
    print(f"Scanning directory: {OUTPUT_ROOT}")
    
    # Find all prediction files
    all_pred_files = []
    
    for root, dirs, files in os.walk(OUTPUT_ROOT):
        for file in files:
            if file.startswith('prediksi') and file.endswith('.txt'):
                file_path = os.path.join(root, file)
                approach = detect_approach_from_path(file_path)
                model_name = extract_model_name(file)
                
                all_pred_files.append({
                    'file': file_path,
                    'approach': approach,
                    'model_name': model_name
                })
    
    if not all_pred_files:
        print(f"\n[ERROR] No prediction files found in {OUTPUT_ROOT}!")
        return
    
    print(f"\nFound {len(all_pred_files)} prediction files")
    print(f"{'='*100}\n")
    
    # Validate all files
    all_details = []
    all_summaries = []
    
    for item in all_pred_files:
        details, summary = validate_prediction_file(
            item['file'],
            item['approach'],
            item['model_name']
        )
        
        if details:
            all_details.extend(details)
        
        if summary:
            all_summaries.append(summary)
    
    # Save detailed results
    if all_details:
        df_details = pd.DataFrame(all_details)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed CSV
        detail_file = os.path.join(RESULTS_DIR, f"syntax_validation_details_{timestamp}.csv")
        df_details.to_csv(detail_file, index=False, encoding='utf-8-sig', sep=';')
        print(f"\n[SAVED] Detailed results: {detail_file}")
        
        # Save summary CSV
        df_summary = pd.DataFrame(all_summaries)
        summary_file = os.path.join(RESULTS_DIR, f"syntax_validation_summary_{timestamp}.csv")
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig', sep=';')
        print(f"[SAVED] Summary results: {summary_file}")
        
        # Generate comparison report
        print(f"\n{'='*100}")
        print("SYNTAX VALIDITY COMPARISON")
        print(f"{'='*100}\n")
        
        # Sort by validity rate
        df_summary = df_summary.sort_values('validity_rate', ascending=False)
        
        # Print table
        print(f"{'Approach':<25} {'Model':<20} {'Valid':<10} {'Invalid':<10} {'Validity %':<12}")
        print(f"{'-'*100}")
        
        for _, row in df_summary.iterrows():
            print(f"{row['approach']:<25} {row['model_name']:<20} "
                  f"{row['valid_queries']:<10} "
                  f"{row['total_queries'] - row['valid_queries']:<10} "
                  f"{row['validity_rate']:.2f}%")
        
        print(f"\n{'='*100}")
        
        # Best model
        best = df_summary.iloc[0]
        print(f"\n🏆 BEST SYNTAX VALIDITY:")
        print(f"   Approach: {best['approach']}")
        print(f"   Model: {best['model_name']}")
        print(f"   Validity Rate: {best['validity_rate']:.2f}%")
        
        # Average by approach
        print(f"\n📊 AVERAGE BY APPROACH:")
        for approach in df_summary['approach'].unique():
            df_approach = df_summary[df_summary['approach'] == approach]
            avg_validity = df_approach['validity_rate'].mean()
            print(f"   {approach}: {avg_validity:.2f}%")
        
        # Most common errors
        print(f"\n⚠️  MOST COMMON ERROR CATEGORIES:")
        error_counts = {}
        for summary in all_summaries:
            for category, count in summary['error_categories'].items():
                error_counts[category] = error_counts.get(category, 0) + count
        
        for category, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {category}: {count}")
        
        print(f"\n{'='*100}\n")
        
        # Merge with Spider evaluation results
        spider_file = os.path.join(RESULTS_DIR, "spider_evaluation_all_models_20251207_232503.csv")
        if os.path.exists(spider_file):
            print(f"[MERGING] Combining with Spider evaluation results...")
            
            try:
                # Read Spider CSV with proper delimiter
                df_spider = pd.read_csv(spider_file, sep=';')
                
                # ✅ Convert numeric columns to float (handle string values)
                numeric_cols = ['exact_match_all', 'exec_match_all', 'exact_match_easy', 
                               'exact_match_medium', 'exact_match_hard', 'exact_match_extra']
                
                for col in numeric_cols:
                    if col in df_spider.columns:
                        df_spider[col] = pd.to_numeric(df_spider[col], errors='coerce')
                
                # ✅ Fix validity_rate formatting in summary
                df_summary['validity_rate'] = pd.to_numeric(df_summary['validity_rate'], errors='coerce')
                
                # Merge on approach and model_name
                df_merged = df_spider.merge(
                    df_summary[['approach', 'model_name', 'validity_rate', 'valid_queries', 'total_queries']],
                    on=['approach', 'model_name'],
                    how='left'
                )
                
                # Save merged results
                merged_file = os.path.join(RESULTS_DIR, f"spider_with_syntax_validation_{timestamp}.csv")
                df_merged.to_csv(merged_file, index=False, encoding='utf-8-sig', sep=';')
                print(f"[SAVED] Merged results: {merged_file}")
                
                # Print comparison
                print(f"\n{'='*100}")
                print("SYNTAX VALIDITY vs EXECUTION ACCURACY")
                print(f"{'='*100}\n")
                print(f"{'Model':<20} {'Approach':<25} {'Syntax %':<12} {'Exec %':<12} {'Diff':<10}")
                print(f"{'-'*100}")
                
                # Sort and display
                df_display = df_merged.dropna(subset=['validity_rate', 'exec_match_all'])
                df_display = df_display.sort_values('validity_rate', ascending=False)
                
                for _, row in df_display.iterrows():
                    syntax_val = row['validity_rate']
                    exec_val = row['exec_match_all']
                    
                    # ✅ Safe calculation with proper type checking
                    if pd.notna(syntax_val) and pd.notna(exec_val):
                        diff = syntax_val - exec_val
                        
                        print(f"{row['model_name']:<20} {row['approach']:<25} "
                              f"{syntax_val:>10.2f}% "
                              f"{exec_val:>10.2f}% "
                              f"{diff:>+9.2f}%")
                    else:
                        print(f"{row['model_name']:<20} {row['approach']:<25} "
                              f"{'N/A':<12} {'N/A':<12} {'N/A':<10}")
                
                print(f"\n{'='*100}")
                
                # ✅ Statistical summary
                print(f"\n📊 STATISTICAL SUMMARY:")
                
                valid_rows = df_display.dropna(subset=['validity_rate', 'exec_match_all'])
                
                if len(valid_rows) > 0:
                    avg_syntax = valid_rows['validity_rate'].mean()
                    avg_exec = valid_rows['exec_match_all'].mean()
                    avg_diff = avg_syntax - avg_exec
                    
                    print(f"   Average Syntax Validity: {avg_syntax:.2f}%")
                    print(f"   Average Execution Accuracy: {avg_exec:.2f}%")
                    print(f"   Average Gap: {avg_diff:+.2f}%")
                    
                    # Correlation
                    correlation = valid_rows['validity_rate'].corr(valid_rows['exec_match_all'])
                    print(f"   Correlation (Syntax vs Exec): {correlation:.3f}")
                    
                    # Gap analysis
                    print(f"\n   Gap Distribution:")
                    valid_rows['gap'] = valid_rows['validity_rate'] - valid_rows['exec_match_all']
                    
                    high_gap = valid_rows[valid_rows['gap'] > 20]
                    medium_gap = valid_rows[(valid_rows['gap'] >= 10) & (valid_rows['gap'] <= 20)]
                    low_gap = valid_rows[valid_rows['gap'] < 10]
                    
                    print(f"     - High gap (>20%): {len(high_gap)} models")
                    print(f"     - Medium gap (10-20%): {len(medium_gap)} models")
                    print(f"     - Low gap (<10%): {len(low_gap)} models")
                    
                    if len(high_gap) > 0:
                        print(f"\n   Models with High Semantic Error (High Gap):")
                        for _, row in high_gap.nlargest(5, 'gap').iterrows():
                            print(f"     - {row['model_name']} ({row['approach']}): {row['gap']:.2f}% gap")
                
                print(f"\n{'='*100}\n")
                
            except Exception as e:
                print(f"[ERROR] Failed to merge results: {e}")
                import traceback
                traceback.print_exc()
    
    else:
        print("\n[ERROR] No validation results generated")


if __name__ == "__main__":
    main()