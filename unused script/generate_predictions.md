import ollama
import json
import os
import time
import csv
import sqlite3
from datetime import datetime

# ============== KONFIGURASI ==============
MODEL_NAME = "starcoder2:3b" 
DATA_DIR = "data"
DATABASE_DIR = "data"
GOLD_FILE = "database/gold_195.sql"
OUTPUT_DIR = "output"

# ============== FUNGSI HELPER ==============

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
        
        # Skip empty lines and filepath comment
        if not line or line.startswith('-- filepath:'):
            continue
            
        # Format: Question N:  <question> ||| <db_id>
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
    # Coba dari SQLite dulu
    sqlite_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
    if os.path.exists(sqlite_path):
        schema = load_schema_from_sqlite(sqlite_path)
        if schema:
            return schema
    
    # Jika tidak ada, coba dari tables.json
    if tables_json_path and os.path.exists(tables_json_path):
        schema = load_schema_from_json(tables_json_path, db_id)
        if schema:
            return schema
    
    return f"Database: {db_id}"


def generate_sql_with_ollama(question, schema, model_name):
    """Generate SQL menggunakan Ollama - VERSI PERBAIKAN"""
    prompt = f"""You are an expert SQL query generator. Given the following database schema and question, generate ONLY the SQL query without any explanation.

Database Schema:
{schema}

Question: {question}

IMPORTANT: Generate the SQL query in a SINGLE LINE without line breaks. Do not add explanations, comments, or markdown formatting.

SQL Query:"""
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        sql = response['message']['content'].strip()
        
        # PERBAIKAN 1: Hapus markdown code blocks
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        # PERBAIKAN 2: Hapus semua line breaks dan normalize whitespace
        # Ini adalah kunci utama perbaikan!
        sql = ' '.join(sql.split())
        
        # PERBAIKAN 3: Hapus komentar SQL (-- atau /* */)
        # Hapus single line comments
        if '--' in sql:
            sql = sql.split('--')[0].strip()
        
        # PERBAIKAN 4: Ambil hanya query pertama jika ada multiple queries
        if ';' in sql:
            sql = sql.split(';')[0].strip()
        
        # PERBAIKAN 5: Pastikan tidak ada trailing/leading spaces
        sql = sql.strip()
        
        # PERBAIKAN 6: Validasi minimal - pastikan ada kata kunci SQL
        sql_lower = sql.lower()
        if not any(keyword in sql_lower for keyword in ['select', 'insert', 'update', 'delete']):
            print(f"WARNING: Query tidak valid, menggunakan default: {sql}")
            return "SELECT 1"
        
        return sql
        
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return "SELECT 1"


def validate_prediction_file(predictions_file):
    """Validasi file predictions sebelum evaluasi"""
    print(f"\nValidating predictions file: {predictions_file}")
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    valid_count = 0
    invalid_count = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"  Line {i}: INVALID - Missing tab separator")
            invalid_count += 1
        else:
            sql, db_id = parts
            if not sql or not db_id:
                print(f"  Line {i}: INVALID - Empty SQL or db_id")
                invalid_count += 1
            else:
                valid_count += 1
    
    print(f"Validation result:")
    print(f"  Valid lines: {valid_count}")
    print(f"  Invalid lines: {invalid_count}")
    
    return valid_count, invalid_count


def main():
    # Buat output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Timestamp dan safe filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = get_safe_filename(MODEL_NAME)
    
    # Output file paths
    predictions_file = os.path.join(OUTPUT_DIR, f"prediksi_{safe_model_name}_{timestamp}.txt")
    metrics_file = os.path.join(OUTPUT_DIR, f"metric_{safe_model_name}_{timestamp}.csv")
    
    print(f"{'='*60}")
    print(f"SQL GENERATION - IMPROVED VERSION")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output files:")
    print(f"  Predictions: {predictions_file}")
    print(f"  Metrics: {metrics_file}")
    
    # Cek tables.json
    tables_json_path = os.path.join(DATA_DIR, "tables.json")
    if not os.path.exists(tables_json_path):
        tables_json_path = "scripts/spider/evaluation_examples/examples/tables.json"
    
    if not os.path.exists(tables_json_path):
        print(f"\nWARNING: tables.json not found at {tables_json_path}")
        print("Schema akan dimuat dari SQLite database saja")
    
    # Cek gold file
    if not os.path.exists(GOLD_FILE):
        print(f"\nERROR: File {GOLD_FILE} tidak ditemukan!")
        return
    
    # Parse questions
    questions = parse_gold_file(GOLD_FILE)
    
    if len(questions) == 0:
        print("\nERROR: Tidak ada pertanyaan yang ditemukan!")
        return
    
    # Prepare data storage
    metrics_data = []
    predictions = []
    
    print(f"\n{'='*60}")
    print(f"Starting SQL Generation")
    print(f"Total questions: {len(questions)}")
    print(f"{'='*60}\n")
    
    total_start_time = time.time()
    error_count = 0
    
    for i, q in enumerate(questions):
        q_id = q['id']
        question = q['question']
        db_id = q['db_id']
        
        print(f"[{i+1}/{len(questions)}] Q{q_id} ({db_id}): {question[:50]}...")
        
        # Get schema
        schema = get_schema(db_id, tables_json_path)
        
        # Generate SQL with timing
        start_time = time.time()
        try:
            pred_sql = generate_sql_with_ollama(question, schema, MODEL_NAME)
        except Exception as e:
            print(f"  ERROR: {e}")
            pred_sql = "SELECT 1"
            error_count += 1
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Validasi SQL tidak kosong
        if not pred_sql or pred_sql.strip() == "":
            pred_sql = "SELECT 1"
            error_count += 1
            print(f"  WARNING: Empty SQL generated, using default")
        
        # Store results (tanpa newline di dalam SQL)
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
        
        # Display preview (max 80 chars)
        display_sql = pred_sql[:77] + "..." if len(pred_sql) > 80 else pred_sql
        print(f"  ✓ {latency:.2f}s: {display_sql}")
    
    total_time = time.time() - total_start_time
    avg_latency = total_time / len(questions)
    
    # Save predictions (.txt)
    print(f"\n{'='*60}")
    print(f"Saving predictions to {predictions_file}...")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"  ✓ Saved {len(predictions)} predictions")
    
    # Save metrics (.csv)
    print(f"\nSaving metrics to {metrics_file}...")
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'question', 'predicted_sql', 
                      'latency_seconds', 'model', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"  ✓ Saved {len(metrics_data)} rows")
    
    # Validate predictions file
    valid_count, invalid_count = validate_prediction_file(predictions_file)
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total Questions: {len(questions)}")
    print(f"Successful: {len(questions) - error_count}")
    print(f"Errors: {error_count}")
    print(f"Valid Predictions: {valid_count}")
    print(f"Invalid Predictions: {invalid_count}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Latency: {avg_latency:.2f} seconds/query")
    print(f"\nOutput Files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    
    # Warning jika ada invalid predictions
    if invalid_count > 0:
        print(f"\n⚠️  WARNING: {invalid_count} invalid predictions found!")
        print(f"   Please check the predictions file before evaluation.")
    
    print(f"\n{'='*60}")
    print("NEXT STEP - Run Evaluation:")
    print(f"{'='*60}")
    # Convert path untuk command line
    pred_path = predictions_file.replace('\\', '/')
    eval_command = f"python scripts/spider/evaluation.py --gold database/gold_195_formatted.sql --pred output/prediksi_gemma3_4b_XXXXX.txt --db data --table scripts/spider/evaluation_examples/examples/tables.json --etype match"
    print(eval_command)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()