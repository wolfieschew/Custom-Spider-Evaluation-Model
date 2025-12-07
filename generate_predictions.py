import ollama
import json
import os
import time
import csv
import sqlite3
from datetime import datetime

MODEL_NAME = "gemma3:4b" 
DATA_DIR = "data"
DATABASE_DIR = "data"
GOLD_FILE = "database/gold_195.sql"
OUTPUT_DIR = "output"

FEW_SHOT_EXAMPLES = """
# EXAMPLE 1 - Basic SELECT with COUNT
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)
Question: How many singers do we have?
SQL: SELECT COUNT(*) FROM singer

# EXAMPLE 2 - SELECT with WHERE and ORDER BY
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)
Question: List the name, country and age for all singers ordered by age from the oldest to the youngest.
SQL: SELECT name, country, age FROM singer ORDER BY age DESC

# EXAMPLE 3 - Aggregation with WHERE
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)
Question: What is the average, minimum, and maximum age of all singers from France?
SQL: SELECT AVG(age), MIN(age), MAX(age) FROM singer WHERE country = 'France'

# EXAMPLE 4 - DISTINCT with WHERE
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)
Question: Show the distinct countries where singers above age 20 are from.
SQL: SELECT DISTINCT country FROM singer WHERE age > 20

# EXAMPLE 5 - GROUP BY with COUNT
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)
Question: How many singers do we have from each country?
SQL: SELECT country, COUNT(*) FROM singer GROUP BY country

# EXAMPLE 6 - Subquery with AVG
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER), song_name (TEXT)
Question: Show names of songs whose singer's age is older than the average age.
SQL: SELECT song_name FROM singer WHERE age > (SELECT AVG(age) FROM singer)

# EXAMPLE 7 - JOIN with multiple tables
Schema: Table concert: concert_id (INTEGER), concert_name (TEXT), stadium_id (INTEGER)
        Table stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER)
Question: Show the stadium name and the number of concerts in each stadium.
SQL: SELECT T2.name, COUNT(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id

# EXAMPLE 8 - BETWEEN operator
Schema: Table stadium: stadium_id (INTEGER), location (TEXT), name (TEXT), capacity (INTEGER)
Question: Show the location and name of stadiums which have some concerts with capacity between 5000 and 10000.
SQL: SELECT location, name FROM stadium WHERE capacity BETWEEN 5000 AND 10000

# EXAMPLE 9 - NOT IN subquery
Schema: Table stadium: stadium_id (INTEGER), name (TEXT)
        Table concert: concert_id (INTEGER), stadium_id (INTEGER)
Question: Show the stadium names without any concert.
SQL: SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)

# EXAMPLE 10 - INTERSECT (SET OPERATION)
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)
Question: What are the countries that have both singers above age 40 and singers below age 30?
SQL: SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30

# EXAMPLE 11 - UNION (SET OPERATION)
Schema: Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), birth_year (INTEGER)
Question: Show all birth years of singers from France or singers from USA.
SQL: SELECT birth_year FROM singer WHERE country = 'France' UNION SELECT birth_year FROM singer WHERE country = 'USA'

# EXAMPLE 12 - EXCEPT (SET OPERATION)
Schema: Table stadium: stadium_id (INTEGER), name (TEXT)
        Table concert: concert_id (INTEGER), stadium_id (INTEGER), year (INTEGER)
Question: Find names of stadiums that did not have a concert in 2014.
SQL: SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.year = 2014

# EXAMPLE 13 - LIKE operator
Schema: Table singer: singer_id (INTEGER), name (TEXT), song_name (TEXT)
Question: List all singers whose song name contains "Hey".
SQL: SELECT name FROM singer WHERE song_name LIKE '%Hey%'

# EXAMPLE 14 - HAVING clause
Schema: Table student: stuid (INTEGER), fname (TEXT), sex (TEXT)
        Table has_pet: stuid (INTEGER), petid (INTEGER)
Question: Find the first name and gender of students who have more than one pet.
SQL: SELECT T1.fname, T1.sex FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid GROUP BY T1.stuid HAVING COUNT(*) > 1

# EXAMPLE 15 - Multiple JOINs
Schema: Table singer: singer_id (INTEGER), name (TEXT)
        Table singer_in_concert: singer_id (INTEGER), concert_id (INTEGER)
        Table concert: concert_id (INTEGER), concert_name (TEXT), year (INTEGER)
Question: Show names of singers who performed in concerts in 2014.
SQL: SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id JOIN concert AS T3 ON T1.concert_id = T3.concert_id WHERE T3.year = 2014

# EXAMPLE 16 - MAX with ORDER BY and LIMIT
Schema: Table stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER), average (REAL)
Question: Show the stadium name and capacity with highest average attendance.
SQL: SELECT name, capacity FROM stadium ORDER BY average DESC LIMIT 1

# EXAMPLE 17 - OR condition
Schema: Table concert: concert_id (INTEGER), year (INTEGER)
Question: How many concerts are there in 2014 or 2015?
SQL: SELECT COUNT(*) FROM concert WHERE year = 2014 OR year = 2015

# EXAMPLE 18 - Complex JOIN with GROUP BY and ORDER BY
Schema: Table concert: concert_id (INTEGER), stadium_id (INTEGER), year (INTEGER)
        Table stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER)
Question: Show stadium name and capacity for the stadium with most concerts after 2013, and sort by the number of concerts in descending order.
SQL: SELECT T2.name, T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.year > 2013 GROUP BY T2.stadium_id ORDER BY COUNT(*) DESC LIMIT 1

# EXAMPLE 19 - Nested subquery with MAX
Schema: Table concert: concert_id (INTEGER), stadium_id (INTEGER)
        Table stadium: stadium_id (INTEGER), capacity (INTEGER)
Question: How many concerts occurred in the stadium with the largest capacity?
SQL: SELECT COUNT(*) FROM concert WHERE stadium_id = (SELECT stadium_id FROM stadium ORDER BY capacity DESC LIMIT 1)

# EXAMPLE 20 - Multiple aggregations
Schema: Table pets: petid (INTEGER), weight (REAL), pettype (TEXT)
Question: Find the maximum and minimum weight for each pet type.
SQL: SELECT MAX(weight), pettype FROM pets GROUP BY pettype
"""


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


def detect_query_pattern(question):
    """Deteksi pattern query untuk hints tambahan"""
    question_lower = question.lower()
    
    hints = []
    
    if any(word in question_lower for word in ['both', 'and also', 'as well as']):
        hints.append("HINT: This might need INTERSECT (items in BOTH conditions)")
    
    if any(word in question_lower for word in ['either', 'or']):
        hints.append("HINT: This might need UNION (items in EITHER condition)")
    
    if any(word in question_lower for word in ['not', 'without', 'except']):
        hints.append("HINT: This might need NOT IN or EXCEPT")
    
    if any(word in question_lower for word in ['how many', 'count', 'number of']):
        hints.append("HINT: Use COUNT(*) or COUNT(column)")
    
    if any(word in question_lower for word in ['average', 'avg', 'mean']):
        hints.append("HINT: Use AVG(column)")
    
    if any(word in question_lower for word in ['maximum', 'highest', 'largest', 'most']):
        hints.append("HINT: Use MAX(column) or ORDER BY column DESC LIMIT 1")
    
    if any(word in question_lower for word in ['minimum', 'lowest', 'smallest', 'least']):
        hints.append("HINT: Use MIN(column) or ORDER BY column ASC LIMIT 1")
    
    if any(word in question_lower for word in ['each', 'every', 'per', 'for each']):
        hints.append("HINT: This likely needs GROUP BY")
    
    if any(word in question_lower for word in ['contain', 'include', 'start with', 'end with']):
        hints.append("HINT: Use LIKE with % wildcards")
    
    if 'between' in question_lower or ('more than' in question_lower and 'less than' in question_lower):
        hints.append("HINT: Use BETWEEN or comparison operators")
    
    return hints


def generate_sql_with_ollama(question, schema, model_name):
    """Generate SQL menggunakan Ollama dengan Few-Shot Learning - IMPROVED VERSION"""
    
    hints = detect_query_pattern(question)
    hints_text = "\n".join(hints) if hints else "No specific hints detected."
    
    prompt = f"""You are an expert SQL query generator. Study these examples carefully:

{FEW_SHOT_EXAMPLES}

IMPORTANT RULES:
1. Generate SQL in a SINGLE LINE without line breaks
2. Use lowercase for SQL keywords (select, from, where, etc.)
3. Use table aliases (T1, T2) for JOINs
4. For "both conditions": use INTERSECT
5. For "either condition": use UNION
6. For "not/without": use NOT IN or EXCEPT
7. For "each/every": use GROUP BY
8. For "contains": use LIKE with %
9. Always match column names EXACTLY as in schema (case-sensitive)
10. Do not add explanations, comments, or markdown formatting

{hints_text}

Now generate SQL for this query:

Database Schema:
{schema}

Question: {question}

SQL Query (single line, lowercase keywords, no explanations):"""
    
    try:
        print(f"  [DEBUG] Prompt length: {len(prompt)} characters")
        print(f"  [DEBUG] Prompt tokens (approx): {len(prompt) // 4}")
        
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.3,           
                'top_p': 0.95,               
                'num_predict': 2000,        
                'stop': ['\n\n', 'Question:', 'Schema:', '# EXAMPLE', 'IMPORTANT'] 
            }
        )
        
        raw_sql = response['message']['content']
        print(f"  [DEBUG] Raw response length: {len(raw_sql)} characters")
        
        if len(raw_sql) > 150:
            print(f"  [DEBUG] Raw response: '{raw_sql[:150]}...'")
        else:
            print(f"  [DEBUG] Raw response: '{raw_sql}'")
        
        sql = raw_sql.strip()
        
        if not sql or len(sql) < 5:
            print(f"  [WARNING] Empty or too short response (length: {len(sql)})")
            print(f"  [WARNING] Using fallback query")
            return "SELECT 1"
        
        
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('#') or line.startswith('//') or line.startswith('--'):
                continue
            
            if line.lower().startswith(('note:', 'explanation:', 'this query', 'the sql')):
                continue
            
            if any(keyword in line.lower() for keyword in ['select', 'insert', 'update', 'delete', 'with', 'from', 'where']):
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
        
        sql = sql.strip()
        
        sql_lower = sql.lower()
        if not any(keyword in sql_lower for keyword in ['select', 'insert', 'update', 'delete', 'with']):
            print(f"  [WARNING] Invalid SQL - no SQL keyword found")
            print(f"  [WARNING] Cleaned result: '{sql[:100]}...'")
            return "SELECT 1"
        
        open_paren = sql.count('(')
        close_paren = sql.count(')')
        if open_paren != close_paren:
            print(f"  [WARNING] Unbalanced parentheses: {open_paren} open, {close_paren} close")
        
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
            print(f"  [WARNING] SQL too short after cleaning (length: {len(sql)})")
            print(f"  [WARNING] Result: '{sql}'")
            return "SELECT 1"
        
        if len(sql) > 150:
            print(f"  [DEBUG] Cleaned SQL: '{sql[:150]}...'")
        else:
            print(f"  [DEBUG] Cleaned SQL: '{sql}'")
        
        print(f"  [DEBUG] Final SQL length: {len(sql)} characters")
        
        return sql
        
    except KeyError as e:
        print(f"  [ERROR] Response format error: {e}")
        print(f"  [ERROR] Response structure may be invalid")
        import traceback
        traceback.print_exc()
        return "SELECT 1"
        
    except Exception as e:
        print(f"  [ERROR] Unexpected exception: {e}")
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
                    print(f"  [!] Line {i}: Fallback query (SELECT 1)")
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
    print(f"SQL GENERATION WITH ENHANCED FEW-SHOT LEARNING")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Few-Shot Examples: 20 comprehensive patterns")
    print(f"Pattern Detection: Enabled")
    print(f"Temperature: 0.1 (low for consistency)")
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
            print(f"  ERROR: {e}")
            pred_sql = "SELECT 1"
            error_count += 1
        
        end_time = time.time()
        latency = end_time - start_time
        
        if pred_sql.lower() == "select 1":
            fallback_count += 1
        
        if not pred_sql or pred_sql.strip() == "":
            pred_sql = "SELECT 1"
            error_count += 1
            print(f"  WARNING: Empty SQL, using fallback")
        
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
        print(f"Consider adjusting prompt or model parameters.")
    
    print(f"\n{'='*60}")
    print("NEXT STEP - Run Evaluation:")
    print(f"{'='*60}")
    pred_path = predictions_file.replace('\\', '/')
    eval_command = f"python scripts/spider/evaluation.py --gold database/gold_195_formatted.sql --pred {pred_path} --db data --table scripts/spider/evaluation_examples/examples/tables.json --etype match"
    print(eval_command)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()