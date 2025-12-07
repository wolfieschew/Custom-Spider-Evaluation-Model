import os
import time
import csv
import json
from datetime import datetime
from typing import List, Dict
import sqlite3

from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy import create_engine, MetaData

EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "deepseek-r1:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

DATA_DIR = "data"
DATABASE_DIR = "data"
GOLD_FILE = "database/gold_195.sql"
OUTPUT_DIR = "output"

USE_FEW_SHOT = True                
INCLUDE_SAMPLE_ROWS = False      

LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.95
LLM_NUM_PREDICT = 2000

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
    """Convert model name to safe filename"""
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_")


def parse_gold_file(gold_file_path):
    """Parse gold SQL file"""
    questions = []
    
    with open(gold_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('--'):
            continue
            
        if line.startswith('Question') and '|||' in line:
            try:
                parts = line.split('|||')
                question_part = parts[0].strip()
                db_id = parts[1].strip()
                
                colon_idx = question_part.find(':')
                q_num = int(question_part[:colon_idx].replace('Question', '').strip())
                question_text = question_part[colon_idx + 1:].strip()
                
                questions.append({
                    'id': q_num,
                    'question': question_text,
                    'db_id': db_id
                })
            except Exception as e:
                continue
    
    return questions


def setup_llamaindex():
    """Setup LlamaIndex with embedding and LLM models"""
    
    print(f"\n{'='*80}")
    print("SETTING UP LLAMAINDEX TEXT-TO-SQL WITH FEW-SHOT")
    print(f"{'='*80}")
    
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        request_timeout=120.0,
        additional_kwargs={
            'num_predict': LLM_NUM_PREDICT,
            'num_ctx': 4096,
            'stop': ['\n\n', 'Question:', 'Schema:', '# EXAMPLE']
        }
    )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print(f"  ✓ Embedding Model: {EMBEDDING_MODEL}")
    print(f"  ✓ LLM Model: {LLM_MODEL}")
    print(f"  ✓ Temperature: {LLM_TEMPERATURE}")
    print(f"  ✓ Top-P: {LLM_TOP_P}")
    print(f"  ✓ Few-Shot Learning: {USE_FEW_SHOT}")
    print(f"  ✓ Sample Rows: {INCLUDE_SAMPLE_ROWS}")
    print(f"{'='*80}\n")
    
    return embed_model, llm


def create_sql_database(db_path: str):
    """Create SQLDatabase object from SQLite file"""
    
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table_names = list(metadata.tables.keys())
        
        table_names = [t for t in table_names if not t.startswith('sqlite_')]
        
        sql_database = SQLDatabase(
            engine,
            include_tables=table_names,
            sample_rows_in_table_info=3 if INCLUDE_SAMPLE_ROWS else 0
        )
        
        return sql_database, table_names
        
    except Exception as e:
        print(f"  [ERROR] Failed to create SQL database: {e}")
        return None, []


def create_custom_text_to_sql_prompt() -> PromptTemplate:
    """
    ✅ NEW: Create custom Text-to-SQL prompt with few-shot examples
    """
    
    template = f"""You are an expert SQL query generator. Study these examples carefully:

{FEW_SHOT_EXAMPLES}

IMPORTANT RULES:
1. Generate SQL in a SINGLE LINE without line breaks
2. Use lowercase for SQL keywords (select, from, where, join, etc.)
3. Use table aliases (T1, T2, T3) when joining multiple tables
4. For "both X and Y" conditions: use INTERSECT
5. For "either X or Y" conditions: use UNION
6. For "not/without/exclude" conditions: use NOT IN or EXCEPT
7. For "each/every/per" aggregations: use GROUP BY
8. For "contains/includes" text matching: use LIKE with % wildcards
9. For finding "maximum/highest/most": use ORDER BY DESC LIMIT 1
10. For finding "minimum/lowest/least": use ORDER BY ASC LIMIT 1
11. Always match column names EXACTLY as shown in schema (case-sensitive)
12. Do NOT add explanations, comments, markdown formatting, or multiple queries

Now, given the following database schema and question, generate the SQL query:

Database Schema:
{{schema}}

Question: {{query_str}}

Remember: Generate ONLY the SQL query in a single line with lowercase keywords. No explanations or markdown.

SQL Query:"""
    
    return PromptTemplate(template)


def create_query_engine_with_fewshot(sql_database):
    """
    ✅ ENHANCED: Create query engine with few-shot examples
    """
    
    if USE_FEW_SHOT:
        text_to_sql_prompt = create_custom_text_to_sql_prompt()
        
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=sql_database.get_usable_table_names(),
            text_to_sql_prompt=text_to_sql_prompt,
            synthesize_response=False,  # Return only SQL
            streaming=False
        )
    else:
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=sql_database.get_usable_table_names(),
            synthesize_response=False,
            streaming=False
        )
    
    return query_engine


def extract_sql_from_response(response) -> str:
    """Extract and normalize SQL query from LlamaIndex response"""
    
    if hasattr(response, 'metadata') and 'sql_query' in response.metadata:
        raw_sql = response.metadata['sql_query'].strip()
    else:
        raw_sql = str(response)
    
    sql = raw_sql.strip()
    
    sql = sql.replace('```sql', '').replace('```', '').strip()
    
    lines = []
    for line in sql.split('\n'):
        line = line.strip()
        if not line or line.startswith('--') or line.startswith('#'):
            continue
        if '--' in line:
            line = line.split('--')[0].strip()
        if line:
            lines.append(line)
    
    sql = ' '.join(lines)
    
    sql = sql.rstrip(';').strip()
    
    import re
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*>\s*', ' > ', sql)
    sql = re.sub(r'\s*<\s*', ' < ', sql)
    sql = re.sub(r'\s*!=\s*', ' != ', sql)
    sql = re.sub(r'\s*<=\s*', ' <= ', sql)
    sql = re.sub(r'\s*>=\s*', ' >= ', sql)
    
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    if len(sql) < 10:
        return "SELECT 1"
    
    sql_lower = sql.lower()
    if not any(kw in sql_lower for kw in ['select', 'insert', 'update', 'delete']):
        return "SELECT 1"
    
    return sql


def validate_prediction_file(predictions_file):
    """Validate predictions file"""
    
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
            invalid_count += 1
        else:
            sql, db_id = parts
            if not sql or not db_id:
                invalid_count += 1
            elif sql.lower() == "select 1":
                warning_count += 1
                valid_count += 1
            else:
                valid_count += 1
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Valid: {valid_count}")
    print(f"[!] Warnings (SELECT 1): {warning_count}")
    print(f"[X] Invalid: {invalid_count}")
    print(f"{'='*60}\n")
    
    return valid_count, invalid_count



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_llm_name = get_safe_filename(LLM_MODEL)
    
    predictions_file = os.path.join(OUTPUT_DIR, f"prediksi_fewshot_{safe_llm_name}_{timestamp}.txt")
    metrics_file = os.path.join(OUTPUT_DIR, f"metric_fewshot_{safe_llm_name}_{timestamp}.csv")
    
    print(f"\n{'='*80}")
    print(f"SQL GENERATION: LLAMAINDEX + FEW-SHOT LEARNING")
    print(f"{'='*80}")
    print(f"Method: NLSQLTableQueryEngine with custom few-shot prompt")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Few-Shot Examples: 20 patterns")
    print(f"Temperature: {LLM_TEMPERATURE}")
    print(f"\nOutput files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    print(f"{'='*80}\n")
    

    embed_model, llm = setup_llamaindex()
    
    if not os.path.exists(GOLD_FILE):
        print(f"ERROR: {GOLD_FILE} not found!")
        return
    
    questions = parse_gold_file(GOLD_FILE)
    print(f"Loaded {len(questions)} questions\n")
    
    metrics_data = []
    predictions = []
    
    print(f"{'='*80}")
    print(f"STARTING SQL GENERATION WITH FEW-SHOT")
    print(f"{'='*80}\n")
    
    total_start_time = time.time()
    error_count = 0
    fallback_count = 0
    
    db_cache = {}
    
    for i, q in enumerate(questions):
        q_id = q['id']
        question = q['question']
        db_id = q['db_id']
        
        print(f"[{i+1}/{len(questions)}] Q{q_id} ({db_id})")
        print(f"  Question: {question[:70]}...")
        
        db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            print(f"  [ERROR] Database not found: {db_path}")
            predictions.append(f"SELECT 1\t{db_id}")
            error_count += 1
            continue
        
        start_time = time.time()
        
        try:
            if db_id not in db_cache:
                print(f"  [INFO] Loading database: {db_id}")
                sql_database, table_names = create_sql_database(db_path)
                
                if sql_database is None:
                    raise Exception("Failed to create SQL database")
                
                print(f"  [INFO] Found {len(table_names)} tables: {', '.join(table_names)}")
                
                query_engine = create_query_engine_with_fewshot(sql_database)
                
                db_cache[db_id] = {
                    'sql_database': sql_database,
                    'query_engine': query_engine,
                    'tables': table_names
                }
            else:
                query_engine = db_cache[db_id]['query_engine']
            
            print(f"  [PRCS] Generating SQL with examples...")
            response = query_engine.query(question)
            
            pred_sql = extract_sql_from_response(response)
            
            if not pred_sql or len(pred_sql) < 10:
                print(f"  [WARNING] Empty/short SQL, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1
            elif not any(kw in pred_sql.lower() for kw in ['select', 'insert', 'update', 'delete']):
                print(f"  [WARNING] No SQL keyword, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            pred_sql = "SELECT 1"
            error_count += 1
        
        end_time = time.time()
        latency = end_time - start_time
        
        predictions.append(f"{pred_sql}\t{db_id}")
        metrics_data.append({
            'question_id': q_id,
            'db_id': db_id,
            'question': question,
            'predicted_sql': pred_sql,
            'latency_seconds': round(latency, 4),
            'llm_model': LLM_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'method': 'NLSQLTableQueryEngine + Few-Shot',
            'timestamp': datetime.now().isoformat()
        })
        
        display_sql = pred_sql[:70] + "..." if len(pred_sql) > 73 else pred_sql
        print(f"  [OK] {latency:.2f}s: {display_sql}\n")
    
    total_time = time.time() - total_start_time
    avg_latency = total_time / len(questions)
    
    print(f"{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"  [OK] Saved predictions: {predictions_file}")
    
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'question', 'predicted_sql', 
                      'latency_seconds', 'llm_model', 'embedding_model', 
                      'method', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"  [OK] Saved metrics: {metrics_file}\n")
    
    valid_count, invalid_count = validate_prediction_file(predictions_file)
    
    print(f"{'='*80}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Method: LlamaIndex Native + Few-Shot Learning")
    print(f"LLM: {LLM_MODEL} (temp={LLM_TEMPERATURE})")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Few-Shot Examples: 20 patterns")
    print(f"\nResults:")
    print(f"Total: {len(questions)}")
    print(f"Successful: {len(questions) - error_count}")
    print(f"Errors: {error_count}")
    print(f"Fallbacks (SELECT 1): {fallback_count}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"\nPerformance:")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Avg Latency: {avg_latency:.2f}s/query")
    print(f"Success Rate: {((len(questions)-fallback_count)/len(questions)*100):.1f}%")
    print(f"\nOutput Files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    
    print(f"\n{'='*80}")
    print("NEXT STEP - Run Evaluation:")
    print(f"{'='*80}")
    pred_path = predictions_file.replace('\\', '/')
    eval_command = f"python scripts/spider/evaluation.py --gold database/gold_195_formatted.sql --pred {pred_path} --db data --table scripts/spider/evaluation_examples/examples/tables.json --etype all"
    print(eval_command)
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()