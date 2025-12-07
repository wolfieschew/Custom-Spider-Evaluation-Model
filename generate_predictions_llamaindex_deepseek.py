import os
import time
import csv
import re
from datetime import datetime
from typing import List, Dict
from sqlalchemy import create_engine, MetaData

from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "deepseek-r1:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

DATA_DIR = "data"
DATABASE_DIR = "data"
GOLD_FILE = "database/gold_195.sql"
OUTPUT_DIR = "output"

LLM_TEMPERATURE = 0.3      
LLM_TOP_P = 0.95           
LLM_TOP_K = 50            
LLM_NUM_PREDICT = 2000    
LLM_NUM_CTX = 4096         
LLM_REPEAT_PENALTY = 1.1  
LLM_REQUEST_TIMEOUT = 180.0

USE_FEW_SHOT = False

MAX_RETRIES = 3
RETRY_DELAY = 5

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
    """Setup LlamaIndex with SAME parameters as manual script"""
    
    print(f"\n{'='*80}")
    print("LLAMAINDEX + DEEPSEEK-R1 (MATCHED PARAMETERS)")
    print(f"{'='*80}")
    
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        request_timeout=LLM_REQUEST_TIMEOUT,
        additional_kwargs={
            'temperature': LLM_TEMPERATURE,     
            'top_p': LLM_TOP_P,                
            'top_k': LLM_TOP_K,               
            'num_predict': LLM_NUM_PREDICT,     
            'num_ctx': LLM_NUM_CTX,            
            'repeat_penalty': LLM_REPEAT_PENALTY,
            'stop': [] 
        }
    )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print(f"  ✓ Framework: LlamaIndex NLSQLTableQueryEngine")
    print(f"  ✓ LLM: {LLM_MODEL}")
    print(f"  ✓ Parameters (MATCHED WITH MANUAL):")
    print(f"    - Temperature: {LLM_TEMPERATURE}")
    print(f"    - Top-P: {LLM_TOP_P}")
    print(f"    - Top-K: {LLM_TOP_K}")
    print(f"    - Num Predict: {LLM_NUM_PREDICT}")
    print(f"    - Context Window: {LLM_NUM_CTX}")
    print(f"    - Repeat Penalty: {LLM_REPEAT_PENALTY}")
    print(f"  ✓ Embedding: {EMBEDDING_MODEL}")
    print(f"  ✓ Few-Shot: No (to save context)")
    print(f"  ✓ Max Retries: {MAX_RETRIES}")
    print(f"{'='*80}\n")
    
    return embed_model, llm


def create_sql_database(db_path: str):
    """Create SQLDatabase object from SQLite file"""
    
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table_names = [t for t in metadata.tables.keys() if not t.startswith('sqlite_')]
        
        sql_database = SQLDatabase(
            engine,
            include_tables=table_names,
            sample_rows_in_table_info=0  
        )
        
        return sql_database, table_names
        
    except Exception as e:
        print(f"  [ERROR] Failed to create SQL database: {e}")
        return None, []


def create_simple_prompt_for_deepseek():
    """
    SIMPLE PROMPT - SAME STYLE AS MANUAL SCRIPT
    """
    
    template = """You are an expert SQL query generator.

Database Schema:
{schema}

Question: {query_str}

Think step by step, then generate the SQL query.
Your final answer must be a single-line SQL query without explanations or comments.

SQL Query:"""
    
    return PromptTemplate(template)


def create_query_engine(sql_database):
    """Create query engine with simple prompt"""
    
    if USE_FEW_SHOT:
        text_to_sql_prompt = create_simple_prompt_for_deepseek()
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=sql_database.get_usable_table_names(),
            text_to_sql_prompt=text_to_sql_prompt,
            synthesize_response=False,
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
    """
    SAME CLEANING LOGIC AS MANUAL SCRIPT
    """
    
    if hasattr(response, 'metadata') and 'sql_query' in response.metadata:
        raw_sql = response.metadata['sql_query'].strip()
    else:
        raw_sql = str(response).strip()
    
    if not raw_sql or len(raw_sql) < 5:
        return "SELECT 1"
    
    sql = raw_sql
    
    print(f"  [DEBUG] Raw response length: {len(sql)} chars")
    if len(sql) > 150:
        print(f"  [DEBUG] Raw preview: {sql[:150]}...")
    
   
    print(f"  [DEBUG] Applying DeepSeek-R1 specific cleaning")
    
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


def query_with_retry(query_engine, question, max_retries=MAX_RETRIES):
    """Query with retry mechanism"""
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"  [RETRY] Attempt {attempt + 1}/{max_retries}")
                time.sleep(RETRY_DELAY)
            
            response = query_engine.query(question)
            return response, None
            
        except Exception as e:
            error_msg = str(e)
            
            if "llama runner process has terminated" in error_msg or "500" in error_msg:
                print(f"  [ERROR] Ollama crashed")
                if attempt < max_retries - 1:
                    print(f"  [WAIT] Restarting in {RETRY_DELAY}s...")
                    continue
                return None, "Ollama crash"
            
            elif "timeout" in error_msg.lower():
                print(f"  [ERROR] Timeout")
                if attempt < max_retries - 1:
                    continue
                return None, "Timeout"
            
            else:
                print(f"  [ERROR] {error_msg[:80]}")
                return None, error_msg
    
    return None, "Max retries exceeded"


def validate_prediction_file(predictions_file):
    """Validate predictions file"""
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    valid_count = 0
    fallback_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) == 2:
            sql, db_id = parts
            if sql.lower() == "select 1":
                fallback_count += 1
            valid_count += 1
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Valid predictions: {valid_count}")
    print(f"[!]  Fallbacks (SELECT 1): {fallback_count}")
    print(f"[✓]  Actual SQL: {valid_count - fallback_count}")
    print(f"{'='*60}\n")
    
    return valid_count, fallback_count



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_llm_name = get_safe_filename(LLM_MODEL)
    
    predictions_file = os.path.join(OUTPUT_DIR, f"prediksi_llamaindex_{safe_llm_name}_{timestamp}.txt")
    metrics_file = os.path.join(OUTPUT_DIR, f"metric_llamaindex_{safe_llm_name}_{timestamp}.csv")
    
    print(f"\n{'='*80}")
    print(f"SQL GENERATION: LLAMAINDEX + DEEPSEEK-R1 (MATCHED SETTINGS)")
    print(f"{'='*80}")
    print(f"Framework: LlamaIndex NLSQLTableQueryEngine")
    print(f"Model: {LLM_MODEL}")
    print(f"Settings: MATCHED with manual generate_predictions_deepseek.py")
    print(f"\nOutput files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    print(f"{'='*80}\n")
    
    embed_model, llm = setup_llamaindex()
    
    if not os.path.exists(GOLD_FILE):
        print(f"ERROR: {GOLD_FILE} not found!")
        return
    
    questions = parse_gold_file(GOLD_FILE)
    print(f"Loaded {len(questions)} questions from Spider benchmark\n")
    
    metrics_data = []
    predictions = []
    
    print(f"{'='*80}")
    print(f"STARTING SQL GENERATION WITH MATCHED SETTINGS")
    print(f"{'='*80}\n")
    
    total_start_time = time.time()
    success_count = 0
    fallback_count = 0
    crash_count = 0
    timeout_count = 0
    error_count = 0
    
    db_cache = {}
    
    for i, q in enumerate(questions):
        q_id = q['id']
        question = q['question']
        db_id = q['db_id']
        
        print(f"[{i+1}/{len(questions)}] Q{q_id} ({db_id})")
        print(f"  Question: {question[:70]}{'...' if len(question) > 70 else ''}")
        
        db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            print(f"  [ERROR] Database not found")
            predictions.append(f"SELECT 1\t{db_id}")
            error_count += 1
            continue
        
        start_time = time.time()
        
        try:
            if db_id not in db_cache:
                sql_database, table_names = create_sql_database(db_path)
                
                if sql_database is None:
                    raise Exception("Failed to create database")
                
                query_engine = create_query_engine(sql_database)
                
                db_cache[db_id] = {
                    'sql_database': sql_database,
                    'query_engine': query_engine,
                    'tables': table_names
                }
            else:
                query_engine = db_cache[db_id]['query_engine']
            
            response, error = query_with_retry(query_engine, question)
            
            if response is None:
                pred_sql = "SELECT 1"
                fallback_count += 1
                
                if "crash" in str(error).lower():
                    crash_count += 1
                elif "timeout" in str(error).lower():
                    timeout_count += 1
                else:
                    error_count += 1
            else:
                pred_sql = extract_sql_from_response(response)
                
                if pred_sql == "SELECT 1":
                    fallback_count += 1
                else:
                    success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] {str(e)[:80]}")
            import traceback
            traceback.print_exc()
            pred_sql = "SELECT 1"
            error_count += 1
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Save
        predictions.append(f"{pred_sql}\t{db_id}")
        metrics_data.append({
            'question_id': q_id,
            'db_id': db_id,
            'question': question,
            'predicted_sql': pred_sql,
            'latency_seconds': round(latency, 4),
            'llm_model': LLM_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'framework': 'LlamaIndex',
            'method': 'NLSQLTableQueryEngine',
            'parameters': f'temp={LLM_TEMPERATURE},top_p={LLM_TOP_P},predict={LLM_NUM_PREDICT}',
            'timestamp': datetime.now().isoformat()
        })
        
        display_sql = pred_sql if len(pred_sql) <= 70 else pred_sql[:67] + "..."
        print(f"  [OK] {latency:.2f}s: {display_sql}\n")
    
    total_time = time.time() - total_start_time
    avg_latency = total_time / len(questions)
    
    print(f"{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"  ✓ Predictions: {predictions_file}")
    
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'question', 'predicted_sql', 
                      'latency_seconds', 'llm_model', 'embedding_model', 
                      'framework', 'method', 'parameters', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"  ✓ Metrics: {metrics_file}\n")
    
    valid_count, fallback_only = validate_prediction_file(predictions_file)
    
    # Summary
    print(f"{'='*80}")
    print(f"GENERATION COMPLETE (MATCHED SETTINGS)")
    print(f"{'='*80}")
    print(f"Framework: LlamaIndex NLSQLTableQueryEngine")
    print(f"Model: {LLM_MODEL}")
    print(f"Settings: temp={LLM_TEMPERATURE}, top_p={LLM_TOP_P}, predict={LLM_NUM_PREDICT}")
    print(f"\n📊 Generation Statistics:")
    print(f"  Total Questions: {len(questions)}")
    print(f"  Successful: {success_count} ({success_count/len(questions)*100:.1f}%)")
    print(f"  Fallbacks: {fallback_count} ({fallback_count/len(questions)*100:.1f}%)")
    print(f"  Crashes: {crash_count}")
    print(f"  Timeouts: {timeout_count}")
    print(f"  Other Errors: {error_count}")
    print(f"\n⏱️ Performance:")
    print(f"  Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Avg Latency: {avg_latency:.2f}s per query")
    print(f"  Throughput: {len(questions)/total_time*60:.2f} queries/minute")
    print(f"\n📁 Output Files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    
    print(f"\n{'='*80}")
    print("EVALUATE WITH SPIDER METRICS:")
    print(f"{'='*80}")
    pred_path = predictions_file.replace('\\', '/')
    eval_command = f"python scripts/spider/evaluation.py --gold database/gold_195_formatted.sql --pred {pred_path} --db data --table scripts/spider/evaluation_examples/examples/tables.json --etype all"
    print(eval_command)
    
    print(f"\n{'='*80}")
    print("COMPARE WITH MANUAL APPROACH:")
    print(f"{'='*80}")
    print("1. Manual script: python generate_predictions_deepseek.py")
    print("2. Compare metrics:")
    print("   - Success rate (LlamaIndex vs Manual)")
    print("   - Average latency")
    print("   - Crash/timeout counts")
    print("   - Framework overhead impact")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()