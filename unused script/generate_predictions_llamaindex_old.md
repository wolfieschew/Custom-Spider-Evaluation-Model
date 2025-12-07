import ollama
import json
import os
import time
import csv
import sqlite3
from datetime import datetime
from typing import List, Dict

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# ============== KONFIGURASI ==============
# Model configuration
EMBEDDING_MODEL = "mxbai-embed-large"           # Embedding model untuk RAG
LLM_MODEL = "gemma3:4b"           # LLM untuk SQL generation
OLLAMA_BASE_URL = "http://localhost:11434"

# Directory configuration
DATA_DIR = "data"
DATABASE_DIR = "data"
GOLD_FILE = "database/gold_195.sql"
OUTPUT_DIR = "output"
INDEX_DIR = "storage"  # Directory untuk menyimpan index

# RAG configuration
TOP_K_EXAMPLES = 5      # Berapa banyak similar examples yang akan diretrieve
CHUNK_SIZE = 512        # Size per chunk untuk indexing
CHUNK_OVERLAP = 50      # Overlap antar chunks

# ============== FEW-SHOT EXAMPLES ==============
FEW_SHOT_EXAMPLES_RAW = [
    {
        "id": 1,
        "category": "basic_count",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "How many singers do we have?",
        "sql": "SELECT COUNT(*) FROM singer",
        "description": "Basic COUNT aggregation without WHERE clause"
    },
    {
        "id": 2,
        "category": "select_order",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "List the name, country and age for all singers ordered by age from the oldest to the youngest.",
        "sql": "SELECT name, country, age FROM singer ORDER BY age DESC",
        "description": "SELECT multiple columns with ORDER BY DESC"
    },
    {
        "id": 3,
        "category": "aggregation_where",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "What is the average, minimum, and maximum age of all singers from France?",
        "sql": "SELECT AVG(age), MIN(age), MAX(age) FROM singer WHERE country = 'France'",
        "description": "Multiple aggregations (AVG, MIN, MAX) with WHERE filter"
    },
    {
        "id": 4,
        "category": "distinct_where",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "Show the distinct countries where singers above age 20 are from.",
        "sql": "SELECT DISTINCT country FROM singer WHERE age > 20",
        "description": "DISTINCT with comparison operator in WHERE"
    },
    {
        "id": 5,
        "category": "group_by",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "How many singers do we have from each country?",
        "sql": "SELECT country, COUNT(*) FROM singer GROUP BY country",
        "description": "GROUP BY with COUNT aggregation"
    },
    {
        "id": 6,
        "category": "subquery",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER), song_name (TEXT)",
        "question": "Show names of songs whose singer's age is older than the average age.",
        "sql": "SELECT song_name FROM singer WHERE age > (SELECT AVG(age) FROM singer)",
        "description": "Subquery with AVG comparison"
    },
    {
        "id": 7,
        "category": "join_group",
        "schema": "Table concert: concert_id (INTEGER), concert_name (TEXT), stadium_id (INTEGER)\nTable stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER)",
        "question": "Show the stadium name and the number of concerts in each stadium.",
        "sql": "SELECT T2.name, COUNT(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id",
        "description": "JOIN with GROUP BY and COUNT"
    },
    {
        "id": 8,
        "category": "between",
        "schema": "Table stadium: stadium_id (INTEGER), location (TEXT), name (TEXT), capacity (INTEGER)",
        "question": "Show the location and name of stadiums which have some concerts with capacity between 5000 and 10000.",
        "sql": "SELECT location, name FROM stadium WHERE capacity BETWEEN 5000 AND 10000",
        "description": "BETWEEN operator for range queries"
    },
    {
        "id": 9,
        "category": "not_in",
        "schema": "Table stadium: stadium_id (INTEGER), name (TEXT)\nTable concert: concert_id (INTEGER), stadium_id (INTEGER)",
        "question": "Show the stadium names without any concert.",
        "sql": "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
        "description": "NOT IN with subquery for exclusion"
    },
    {
        "id": 10,
        "category": "intersect",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "What are the countries that have both singers above age 40 and singers below age 30?",
        "sql": "SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30",
        "description": "INTERSECT for finding items matching BOTH conditions"
    },
    {
        "id": 11,
        "category": "union",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), birth_year (INTEGER)",
        "question": "Show all birth years of singers from France or singers from USA.",
        "sql": "SELECT birth_year FROM singer WHERE country = 'France' UNION SELECT birth_year FROM singer WHERE country = 'USA'",
        "description": "UNION for combining results from multiple conditions"
    },
    {
        "id": 12,
        "category": "except",
        "schema": "Table stadium: stadium_id (INTEGER), name (TEXT)\nTable concert: concert_id (INTEGER), stadium_id (INTEGER), year (INTEGER)",
        "question": "Find names of stadiums that did not have a concert in 2014.",
        "sql": "SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.year = 2014",
        "description": "EXCEPT for set difference operations"
    },
    {
        "id": 13,
        "category": "like",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), song_name (TEXT)",
        "question": "List all singers whose song name contains 'Hey'.",
        "sql": "SELECT name FROM singer WHERE song_name LIKE '%Hey%'",
        "description": "LIKE with wildcards for pattern matching"
    },
    {
        "id": 14,
        "category": "having",
        "schema": "Table student: stuid (INTEGER), fname (TEXT), sex (TEXT)\nTable has_pet: stuid (INTEGER), petid (INTEGER)",
        "question": "Find the first name and gender of students who have more than one pet.",
        "sql": "SELECT T1.fname, T1.sex FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid GROUP BY T1.stuid HAVING COUNT(*) > 1",
        "description": "HAVING clause for filtering grouped results"
    },
    {
        "id": 15,
        "category": "multiple_joins",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT)\nTable singer_in_concert: singer_id (INTEGER), concert_id (INTEGER)\nTable concert: concert_id (INTEGER), concert_name (TEXT), year (INTEGER)",
        "question": "Show names of singers who performed in concerts in 2014.",
        "sql": "SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id JOIN concert AS T3 ON T1.concert_id = T3.concert_id WHERE T3.year = 2014",
        "description": "Multiple JOINs across three tables"
    },
    {
        "id": 16,
        "category": "max_limit",
        "schema": "Table stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER), average (REAL)",
        "question": "Show the stadium name and capacity with highest average attendance.",
        "sql": "SELECT name, capacity FROM stadium ORDER BY average DESC LIMIT 1",
        "description": "ORDER BY with LIMIT to find maximum"
    },
    {
        "id": 17,
        "category": "or_condition",
        "schema": "Table concert: concert_id (INTEGER), year (INTEGER)",
        "question": "How many concerts are there in 2014 or 2015?",
        "sql": "SELECT COUNT(*) FROM concert WHERE year = 2014 OR year = 2015",
        "description": "OR condition in WHERE clause"
    },
    {
        "id": 18,
        "category": "complex_join_group",
        "schema": "Table concert: concert_id (INTEGER), stadium_id (INTEGER), year (INTEGER)\nTable stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER)",
        "question": "Show stadium name and capacity for the stadium with most concerts after 2013, and sort by the number of concerts in descending order.",
        "sql": "SELECT T2.name, T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.year > 2013 GROUP BY T2.stadium_id ORDER BY COUNT(*) DESC LIMIT 1",
        "description": "Complex query with JOIN, WHERE, GROUP BY, ORDER BY, LIMIT"
    },
    {
        "id": 19,
        "category": "nested_subquery",
        "schema": "Table concert: concert_id (INTEGER), stadium_id (INTEGER)\nTable stadium: stadium_id (INTEGER), capacity (INTEGER)",
        "question": "How many concerts occurred in the stadium with the largest capacity?",
        "sql": "SELECT COUNT(*) FROM concert WHERE stadium_id = (SELECT stadium_id FROM stadium ORDER BY capacity DESC LIMIT 1)",
        "description": "Nested subquery with ORDER BY and LIMIT"
    },
    {
        "id": 20,
        "category": "multiple_aggregations",
        "schema": "Table pets: petid (INTEGER), weight (REAL), pettype (TEXT)",
        "question": "Find the maximum and minimum weight for each pet type.",
        "sql": "SELECT MAX(weight), pettype FROM pets GROUP BY pettype",
        "description": "Multiple aggregations with GROUP BY"
    }
]


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


# ============== LLAMAINDEX SETUP ==============

def setup_llamaindex():
    """Setup LlamaIndex dengan embedding model dan LLM"""
    
    print(f"\n{'='*60}")
    print("SETTING UP LLAMAINDEX")
    print(f"{'='*60}")
    
    # 1. Setup Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    
    # 2. Setup LLM
    print(f"Loading LLM: {LLM_MODEL}")
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        request_timeout=120.0
    )
    
    # 3. Configure Settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    
    print(f"  ✓ Embedding model: {EMBEDDING_MODEL}")
    print(f"  ✓ LLM: {LLM_MODEL}")
    print(f"  ✓ Chunk size: {CHUNK_SIZE}")
    print(f"  ✓ Chunk overlap: {CHUNK_OVERLAP}")
    print(f"{'='*60}\n")
    
    return embed_model, llm


def create_few_shot_index(examples: List[Dict], force_rebuild=False):
    """Create or load VectorStoreIndex dari few-shot examples"""
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "few_shot_index")
    
    # Check if index exists
    if os.path.exists(index_path) and not force_rebuild:
        print(f"\n{'='*60}")
        print("LOADING EXISTING INDEX")
        print(f"{'='*60}")
        print(f"Loading from: {index_path}")
        
        try:
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            print(f"  ✓ Successfully loaded existing index")
            print(f"{'='*60}\n")
            return index
        except Exception as e:
            print(f"  ✗ Failed to load index: {e}")
            print(f"  → Rebuilding index...")
    
    # Build new index
    print(f"\n{'='*60}")
    print("BUILDING NEW INDEX FROM FEW-SHOT EXAMPLES")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    
    # Convert examples to Documents
    documents = []
    for ex in examples:
        # Format text untuk semantic search
        text = f"""
Category: {ex['category']}
Description: {ex['description']}

Database Schema:
{ex['schema']}

Question: {ex['question']}

SQL Query: {ex['sql']}
"""
        
        doc = Document(
            text=text,
            metadata={
                "example_id": ex['id'],
                "category": ex['category'],
                "question": ex['question'],
                "sql": ex['sql'],
                "schema": ex['schema']
            }
        )
        documents.append(doc)
    
    print(f"Creating index with {len(documents)} documents...")
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    # Persist index
    print(f"Persisting index to: {index_path}")
    index.storage_context.persist(persist_dir=index_path)
    
    print(f"  ✓ Index created and saved")
    print(f"{'='*60}\n")
    
    return index


def retrieve_relevant_examples(index, question: str, top_k: int = 5):
    """Retrieve relevant few-shot examples menggunakan semantic search"""
    
    # Create retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )
    
    # Retrieve nodes
    nodes = retriever.retrieve(question)
    
    # Extract examples
    examples = []
    for i, node in enumerate(nodes):
        examples.append({
            'rank': i + 1,
            'score': node.score,
            'category': node.metadata.get('category', 'unknown'),
            'question': node.metadata.get('question', ''),
            'sql': node.metadata.get('sql', ''),
            'schema': node.metadata.get('schema', ''),
            'text': node.text
        })
    
    return examples


def generate_sql_with_llamaindex(question: str, schema: str, index, top_k: int = 5):
    """Generate SQL menggunakan LlamaIndex RAG"""
    
    print(f"  [RAG] Retrieving top-{top_k} relevant examples...")
    
    # 1. Retrieve relevant examples
    relevant_examples = retrieve_relevant_examples(index, question, top_k)
    
    # 2. Format examples untuk prompt
    examples_text = ""
    for ex in relevant_examples:
        examples_text += f"""
# EXAMPLE {ex['rank']} (Similarity: {ex['score']:.3f})
Category: {ex['category']}
Schema: {ex['schema']}
Question: {ex['question']}
SQL: {ex['sql']}
"""
    
    print(f"  [RAG] Retrieved examples from categories: {', '.join([ex['category'] for ex in relevant_examples])}")
    
    # 3. Build prompt dengan relevant examples
    prompt = f"""You are an expert SQL query generator. Study these RELEVANT examples:

{examples_text}

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

Now generate SQL for this query:

Database Schema:
{schema}

Question: {question}

SQL Query (single line, lowercase keywords, no explanations):"""
    
    print(f"  [RAG] Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    
    # 4. Generate with LLM
    llm = Settings.llm
    
    response = llm.complete(prompt)
    sql = response.text.strip()
    
    return sql, relevant_examples


def clean_sql(sql: str) -> str:
    """Clean dan normalize generated SQL"""
    
    # Remove markdown
    sql = sql.replace('```sql', '').replace('```', '').strip()
    
    # Filter non-SQL lines
    lines = sql.split('\n')
    sql_lines = []
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        if line.startswith('#') or line.startswith('//') or line.startswith('--'):
            continue
        
        if line.lower().startswith(('note:', 'explanation:', 'this query', 'the sql', 
                                   'answer:', 'result:')):
            continue
        
        if any(keyword in line.lower() for keyword in ['select', 'insert', 'update', 
                                                       'delete', 'with', 'from', 'where']):
            sql_lines.append(line)
    
    if sql_lines:
        sql = ' '.join(sql_lines)
    
    # Normalize whitespace
    sql = ' '.join(sql.split())
    
    # Remove comments
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
    
    # Take first query
    if ';' in sql:
        sql = sql.split(';')[0].strip()
    
    # Fix operators
    sql = sql.replace('=', ' = ')
    sql = sql.replace('>', ' > ')
    sql = sql.replace('<', ' < ')
    sql = sql.replace('! =', '!=')
    sql = sql.replace('< =', '<=')
    sql = sql.replace('> =', '>=')
    sql = sql.replace('< >', '<>')
    
    sql = ' '.join(sql.split())
    sql = sql.strip()
    
    return sql


def validate_prediction_file(predictions_file):
    """Validasi file predictions"""
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


# ============== MAIN FUNCTION ==============

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_llm_name = get_safe_filename(LLM_MODEL)
    safe_embed_name = get_safe_filename(EMBEDDING_MODEL)
    
    predictions_file = os.path.join(OUTPUT_DIR, f"prediksi_rag_{safe_llm_name}_{timestamp}.txt")
    metrics_file = os.path.join(OUTPUT_DIR, f"metric_rag_{safe_llm_name}_{timestamp}.csv")
    retrieval_log_file = os.path.join(OUTPUT_DIR, f"retrieval_log_{timestamp}.json")
    
    print(f"\n{'='*80}")
    print(f"SQL GENERATION WITH LLAMAINDEX RAG")
    print(f"{'='*80}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"RAG Top-K: {TOP_K_EXAMPLES}")
    print(f"Few-Shot Examples: {len(FEW_SHOT_EXAMPLES_RAW)}")
    print(f"\nOutput files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    print(f"  3. {retrieval_log_file}")
    print(f"{'='*80}\n")
    
    # Setup LlamaIndex
    embed_model, llm = setup_llamaindex()
    
    # Create/Load Index
    index = create_few_shot_index(FEW_SHOT_EXAMPLES_RAW, force_rebuild=False)
    
    # Load questions
    tables_json_path = os.path.join(DATA_DIR, "tables.json")
    if not os.path.exists(tables_json_path):
        tables_json_path = "scripts/spider/evaluation_examples/examples/tables.json"
    
    if not os.path.exists(GOLD_FILE):
        print(f"ERROR: File {GOLD_FILE} not found!")
        return
    
    questions = parse_gold_file(GOLD_FILE)
    
    if len(questions) == 0:
        print("ERROR: No questions found!")
        return
    
    # Generation
    metrics_data = []
    predictions = []
    retrieval_logs = []
    
    print(f"\n{'='*80}")
    print(f"STARTING SQL GENERATION WITH RAG")
    print(f"Total questions: {len(questions)}")
    print(f"{'='*80}\n")
    
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
            # Generate dengan RAG
            pred_sql, relevant_examples = generate_sql_with_llamaindex(
                question, schema, index, top_k=TOP_K_EXAMPLES
            )
            
            # Clean SQL
            pred_sql = clean_sql(pred_sql)
            
            # Validate
            if not pred_sql or len(pred_sql) < 10:
                print(f"  [WARNING] Generated SQL too short, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1
            
            sql_lower = pred_sql.lower()
            if not any(kw in sql_lower for kw in ['select', 'insert', 'update', 'delete']):
                print(f"  [WARNING] No SQL keyword, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1
            
            # Log retrieval
            retrieval_logs.append({
                'question_id': q_id,
                'question': question,
                'retrieved_categories': [ex['category'] for ex in relevant_examples],
                'similarity_scores': [ex['score'] for ex in relevant_examples]
            })
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            pred_sql = "SELECT 1"
            error_count += 1
            relevant_examples = []
        
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
            'top_k': TOP_K_EXAMPLES,
            'retrieved_categories': ','.join([ex['category'] for ex in relevant_examples]) if relevant_examples else '',
            'timestamp': datetime.now().isoformat()
        })
        
        display_sql = pred_sql[:70] + "..." if len(pred_sql) > 73 else pred_sql
        print(f"  [OK] {latency:.2f}s: {display_sql}\n")
    
    total_time = time.time() - total_start_time
    avg_latency = total_time / len(questions)
    
    # Save results
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
                      'latency_seconds', 'llm_model', 'embedding_model', 'top_k',
                      'retrieved_categories', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"  [OK] Saved {len(metrics_data)} rows\n")
    
    print(f"Saving retrieval log to: {retrieval_log_file}")
    with open(retrieval_log_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_logs, f, indent=2)
    print(f"  [OK] Saved retrieval logs\n")
    
    # Validate
    valid_count, invalid_count = validate_prediction_file(predictions_file)
    
    # Summary
    print(f"{'='*80}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"RAG Top-K: {TOP_K_EXAMPLES}")
    print(f"\nResults:")
    print(f"Total Questions: {len(questions)}")
    print(f"Successful: {len(questions) - error_count}")
    print(f"Errors: {error_count}")
    print(f"Fallback queries (SELECT 1): {fallback_count}")
    print(f"Valid Predictions: {valid_count}")
    print(f"Invalid Predictions: {invalid_count}")
    print(f"\nPerformance:")
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average Latency: {avg_latency:.2f} seconds/query")
    
    success_rate = ((len(questions) - fallback_count) / len(questions)) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Throughput: {len(questions)/total_time:.2f} queries/second")
    
    print(f"\nOutput Files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    print(f"  3. {retrieval_log_file}")
    
    print(f"\n{'='*80}")
    print("NEXT STEP - Run Evaluation:")
    print(f"{'='*80}")
    pred_path = predictions_file.replace('\\', '/')
    eval_command = f"python scripts/spider/evaluation.py --gold database/gold_195_formatted.sql --pred {pred_path} --db data --table scripts/spider/evaluation_examples/examples/tables.json --etype all"
    print(eval_command)
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()