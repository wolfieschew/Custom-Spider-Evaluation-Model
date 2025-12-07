import ollama
import json
import os
import time
import csv
import sqlite3
from datetime import datetime


MODEL_NAME = "deepseek-r1:1.5b" 
DATA_DIR = "data"
DATABASE_DIR = "data"
GOLD_FILE = "database/gold_195.sql"
OUTPUT_DIR = "output"


def get_safe_filename(model_name):
    """Mengubah nama model menjadi nama file yang valid"""
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_")


def load_schema_from_sqlite(db_path):
    """Load schema dari database SQLite"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_str = ""
        for table in tables:
            table_name = table[0]
            if table_name.startswith('sqlite_'):
                continue
            cursor.execute(f"PRAGMA table_info(`{table_name}`);")
            columns = cursor.fetchall()
            cols = [f"{col[1]} ({col[2]})" for col in columns]
            schema_str += f"Table {table_name}: {', '.join(cols)}\n"
        
        conn.close()
        return schema_str
    except Exception as e:
        print(f"Error loading schema from SQLite: {e}")
        return ""


def load_schema_from_json(tables_json_path, db_id):
    """Load schema dari tables.json"""
    try:
        with open(tables_json_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        
        for db in tables_data:
            if db['db_id'] == db_id:
                schema_str = ""
                table_names = db['table_names_original']
                column_names = db['column_names_original']
                column_types = db['column_types']
                
                for i, table_name in enumerate(table_names):
                    cols = []
                    for j, (table_idx, col_name) in enumerate(column_names):
                        if table_idx == i:
                            col_type = column_types[j] if j < len(column_types) else "unknown"
                            cols.append(f"{col_name} ({col_type})")
                    if cols:
                        schema_str += f"Table {table_name}: {', '.join(cols)}\n"
                
                return schema_str
        return ""
    except Exception as e:
        print(f"Error loading schema from JSON: {e}")
        return ""


def parse_gold_file(gold_file_path):
    """Parse file gold_195.sql untuk mendapatkan questions dan db_id"""
    questions = []
    
    print(f"Reading file: {gold_file_path}")
    
    with open(gold_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    print(f"Total lines in file: {len(lines)}")
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        
        if not line or line.startswith('-- filepath:'):
            continue
            
        if line.startswith('Question'):
            try:
                if '|||' not in line:
                    continue
                    
                parts = line.split('|||')
                question_part = parts[0].strip()
                db_id = parts[1].strip()
                
                colon_idx = question_part.find(':')
                if colon_idx == -1:
                    continue
                
                q_num_part = question_part[:colon_idx]
                question_text = question_part[colon_idx + 1:].strip()
                q_num = int(q_num_part.replace('Question', '').strip())
                
                questions.append({
                    'id': q_num,
                    'question': question_text,
                    'db_id': db_id
                })
                
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Successfully parsed {len(questions)} questions")
    return questions


def get_schema(db_id, tables_json_path=None):
    """Get schema untuk database tertentu"""
    sqlite_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
    if os.path.exists(sqlite_path):
        schema = load_schema_from_sqlite(sqlite_path)
        if schema:
            return schema
    
    if tables_json_path and os.path.exists(tables_json_path):
        schema = load_schema_from_json(tables_json_path, db_id)
        if schema:
            return schema
    
    return f"Database: {db_id}"


def generate_sql_with_ollama(question, schema, model_name):
    """Generate SQL menggunakan Ollama - OPTIMIZED FOR DEEPSEEK-R1 AND OTHERS"""
    
    is_deepseek_r1 = 'deepseek-r1' in model_name.lower() or 'deepseek-reasoner' in model_name.lower()
    
    if is_deepseek_r1:
        prompt = f"""You are an expert SQL query generator.

Database Schema:
{schema}

Question: {question}

Think step by step, then generate the SQL query.
Your final answer must be a single-line SQL query without explanations or comments.

SQL Query:"""
    else:
        prompt = f"""You are an expert SQL query generator. Given the following database schema and question, generate ONLY the SQL query without any explanation.

Database Schema:
{schema}

Question: {question}

IMPORTANT: Generate the SQL query in a SINGLE LINE without line breaks. Do not add explanations, comments, or markdown formatting.

SQL Query:"""
    
    try:
        if is_deepseek_r1:
            print(f"  [DEBUG] Using DeepSeek-R1 optimized parameters")
            options = {
                'temperature': 0.3,       
                'top_p': 0.95,
                'top_k': 50,
                'num_predict': 2000,     
                'repeat_penalty': 1.1,
            }
        else:
            print(f"  [DEBUG] Using standard parameters")
            options = {
                'temperature': 0.3,
                'top_p': 0.9,
                'num_predict': 300,
            }
        
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options=options
        )

        sql = response['message']['content'].strip()
        
        print(f"  [DEBUG] Raw response length: {len(sql)} chars")
        if len(sql) > 150:
            print(f"  [DEBUG] Raw preview: {sql[:150]}...")
        else:
            print(f"  [DEBUG] Raw output: {sql}")
        
        if is_deepseek_r1:
            print(f"  [DEBUG] Applying DeepSeek-R1 specific cleaning")
            
            # Extract dari tag <answer>
            if '<answer>' in sql.lower():
                answer_start = sql.lower().find('<answer>')
                sql = sql[answer_start + 8:]
                
                if '</answer>' in sql.lower():
                    answer_end = sql.lower().find('</answer>')
                    sql = sql[:answer_end]
                
                print(f"  [DEBUG] Extracted from <answer> tag")
            
            lines = sql.split('\n')
            sql_candidates = []
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['select', 'insert', 'update', 'delete', 'with']):
                    sql_candidates.append(line)
            
            if sql_candidates:
                sql = sql_candidates[-1] 
                print(f"  [DEBUG] Found {len(sql_candidates)} SQL candidates, using last one")
        
        
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
       
        sql = sql.replace('<think>', '').replace('</think>', '').strip()
        sql = sql.replace('<answer>', '').replace('</answer>', '').strip()
        
        
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            
            if line.startswith('#') or line.startswith('//') or line.startswith('--'):
                continue
            
            
            if line.lower().startswith(('note:', 'explanation:', 'this query', 'the sql', 
                                       'therefore', 'so ', 'thus', 'here is', 'answer:')):
                continue
            
            
            if any(keyword in line.lower() for keyword in ['select', 'insert', 'update', 
                                                           'delete', 'with', 'from', 'where']):
                sql_lines.append(line)
        
        if sql_lines:
            sql = ' '.join(sql_lines)
    
        sql = ' '.join(sql.split())
        
        if '--' in sql:
            sql = sql.split('--')[0].strip()
        
        while '/*' in sql and '*/' in sql:
            start = sql.find('/*')
            end = sql.find('*/', start)
            if end != -1:
                sql = sql[:start] + sql[end+2:]
            else:
                break
        sql = sql.strip()
        
        if ';' in sql:
            sql = sql.split(';')[0].strip()
        
        sql_lower = sql.lower()
        if not any(keyword in sql_lower for keyword in ['select', 'insert', 'update', 'delete']):
            print(f"  [WARNING] No SQL keyword found, using fallback")
            return "SELECT 1"
        
        if sql.count('(') != sql.count(')'):
            print(f"  [WARNING] Unbalanced parentheses")
        
        sql = sql.replace('=', ' = ')
        sql = sql.replace('>', ' > ')
        sql = sql.replace('<', ' < ')
        sql = sql.replace('! =', '!=')
        sql = sql.replace('< =', '<=')
        sql = sql.replace('> =', '>=')
        sql = sql.replace('< >', '<>')
        
        sql = ' '.join(sql.split())
        sql = sql.strip()
        
        if len(sql) < 10:
            print(f"  [WARNING] SQL too short ({len(sql)} chars)")
            return "SELECT 1"
        
        print(f"  [DEBUG] Final SQL: {sql[:100]}..." if len(sql) > 100 else f"  [DEBUG] Final SQL: {sql}")
        
        return sql
        
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return "SELECT 1"


def validate_prediction_file(predictions_file):
    """Validasi file predictions sebelum evaluasi"""
    print(f"\n{'='*60}")
    print(f"VALIDATING PREDICTIONS FILE")
    print(f"{'='*60}")
    print(f"File: {predictions_file}\n")
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    valid_count = 0
    invalid_count = 0
    warning_count = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"  [X] Line {i}: Missing tab separator")
            invalid_count += 1
        else:
            sql, db_id = parts
            if not sql or not db_id:
                print(f"  [X] Line {i}: Empty SQL or db_id")
                invalid_count += 1
            else:
                sql_lower = sql.lower()
                if sql_lower == "select 1":
                    print(f"  [!] Line {i}: Fallback query")
                    warning_count += 1
                    valid_count += 1
                elif 'from' not in sql_lower:
                    print(f"  [!] Line {i}: Missing FROM clause")
                    warning_count += 1
                    valid_count += 1
                else:
                    valid_count += 1
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Valid predictions: {valid_count}")
    print(f"[!] Warnings: {warning_count}")
    print(f"[X] Invalid predictions: {invalid_count}")
    print(f"{'='*60}\n")
    
    return valid_count, invalid_count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = get_safe_filename(MODEL_NAME)
    
    predictions_file = os.path.join(OUTPUT_DIR, f"prediksi_{safe_model_name}_{timestamp}.txt")
    metrics_file = os.path.join(OUTPUT_DIR, f"metric_{safe_model_name}_{timestamp}.csv")
    
    print(f"\n{'='*60}")
    print(f"SQL GENERATION - OPTIMIZED FOR DEEPSEEK-R1")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    
    # Deteksi model type
    is_deepseek_r1 = 'deepseek-r1' in MODEL_NAME.lower()
    if is_deepseek_r1:
        print(f"Model Type: DeepSeek-R1 (Reasoning Model)")
        print(f"Special handling: Enabled")
    else:
        print(f"Model Type: Standard SQL Generation Model")
    
    print(f"\nOutput files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    print(f"{'='*60}\n")
    
    tables_json_path = os.path.join(DATA_DIR, "tables.json")
    if not os.path.exists(tables_json_path):
        tables_json_path = "scripts/spider/evaluation_examples/examples/tables.json"
    
    if not os.path.exists(tables_json_path):
        print(f"WARNING: tables.json not found at {tables_json_path}")
        print("Schema will be loaded from SQLite only\n")
    
    if not os.path.exists(GOLD_FILE):
        print(f"ERROR: File {GOLD_FILE} not found!")
        return
    
    questions = parse_gold_file(GOLD_FILE)
    
    if len(questions) == 0:
        print("ERROR: No questions found!")
        return
    
    metrics_data = []
    predictions = []
    
    print(f"\n{'='*60}")
    print(f"STARTING SQL GENERATION")
    print(f"Total questions: {len(questions)}")
    print(f"{'='*60}\n")
    
    total_start_time = time.time()
    error_count = 0
    fallback_count = 0
    
    for i, q in enumerate(questions):
        q_id = q['id']
        question = q['question']
        db_id = q['db_id']
        
        print(f"[{i+1}/{len(questions)}] Q{q_id} ({db_id})")
        print(f"  Question: {question[:70]}...")
        
        schema = get_schema(db_id, tables_json_path)
        
        start_time = time.time()
        try:
            pred_sql = generate_sql_with_ollama(question, schema, MODEL_NAME)
        except Exception as e:
            print(f"  [ERROR] {e}")
            pred_sql = "SELECT 1"
            error_count += 1
        
        end_time = time.time()
        latency = end_time - start_time
        
        if pred_sql.lower() == "select 1":
            fallback_count += 1
        
        if not pred_sql or pred_sql.strip() == "":
            pred_sql = "SELECT 1"
            error_count += 1
            print(f"  [WARNING] Empty SQL, using fallback")
        
        predictions.append(f"{pred_sql}\t{db_id}")
        metrics_data.append({
            'question_id': q_id,
            'db_id': db_id,
            'question': question,
            'predicted_sql': pred_sql,
            'latency_seconds': round(latency, 4),
            'model': MODEL_NAME,
            'timestamp': datetime.now().isoformat()
        })
        
        display_sql = pred_sql[:70] + "..." if len(pred_sql) > 73 else pred_sql
        print(f"  [OK] {latency:.2f}s: {display_sql}\n")
    
    total_time = time.time() - total_start_time
    avg_latency = total_time / len(questions)
    
    print(f"{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")
    print(f"Saving predictions to: {predictions_file}")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"  [OK] Saved {len(predictions)} predictions\n")
    
    print(f"Saving metrics to: {metrics_file}")
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'question', 'predicted_sql', 
                      'latency_seconds', 'model', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"  [OK] Saved {len(metrics_data)} rows\n")
    
    valid_count, invalid_count = validate_prediction_file(predictions_file)
    
    print(f"{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total Questions: {len(questions)}")
    print(f"Successful: {len(questions) - error_count}")
    print(f"Errors: {error_count}")
    print(f"Fallback queries (SELECT 1): {fallback_count}")
    print(f"Valid Predictions: {valid_count}")
    print(f"Invalid Predictions: {invalid_count}")
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average Latency: {avg_latency:.2f} seconds/query")
    
    success_rate = ((len(questions) - fallback_count) / len(questions)) * 100
    print(f"\nPerformance Metrics:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Throughput: {len(questions)/total_time:.2f} queries/second")
    
    print(f"\nOutput Files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    
    if invalid_count > 0:
        print(f"\nWARNING: {invalid_count} invalid predictions found!")
        print(f"Please review the predictions file before evaluation.")
    
    if fallback_count > len(questions) * 0.1:
        print(f"\nWARNING: High fallback rate ({fallback_count}/{len(questions)})")
        print(f"Consider switching to a different model:")
        print(f"  - qwen2.5-coder:7b (Recommended for SQL)")
        print(f"  - codellama:7b")
        print(f"  - deepseek-coder:6.7b (Not reasoning version)")
    
    print(f"\n{'='*60}")
    print("NEXT STEP - Run Evaluation:")
    print(f"{'='*60}")
    pred_path = predictions_file.replace('\\', '/')
    eval_command = f"python scripts/spider/evaluation.py --gold database/gold_195_formatted.sql --pred {pred_path} --db data --table scripts/spider/evaluation_examples/examples/tables.json --etype match"
    print(eval_command)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()