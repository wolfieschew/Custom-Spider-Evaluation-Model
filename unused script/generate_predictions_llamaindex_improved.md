import ollama
import os
import time
import csv
import json
import re
from datetime import datetime
from typing import List, Dict
import sqlite3

# LlamaIndex imports
from llama_index.core import SQLDatabase, Settings, Document, VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy import create_engine, MetaData

# ============== KONFIGURASI ==============
# Model configuration
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Directory configuration
DATA_DIR = "data"
DATABASE_DIR = "data"
GOLD_FILE = "database/gold_195.sql"
OUTPUT_DIR = "output"
INDEX_DIR = "storage_improved"

# ✅ IMPROVED: Hybrid RAG + Text-to-SQL config
USE_FEW_SHOT_EXAMPLES = True       # Enable few-shot learning
TOP_K_EXAMPLES = 5                  # ✅ INCREASED from 3 to 5
SIMILARITY_THRESHOLD = 0.25         # ✅ LOWERED from 0.35 to 0.25
INCLUDE_SAMPLE_ROWS = False         # ✅ DISABLED (simpler like manual)
ENABLE_PATTERN_HINTS = True         # Add dynamic hints

# LLM parameters (optimized from manual version)
LLM_TEMPERATURE = 0.3               # Sweet spot for SQL
LLM_TOP_P = 0.95
LLM_NUM_PREDICT = 2000
LLM_STOP_SEQUENCES = ['\n\n', 'Question:', 'Schema:', '# EXAMPLE']

# ============== FEW-SHOT EXAMPLES ==============
FEW_SHOT_EXAMPLES_RAW = [
    {
        "id": 1,
        "category": "basic_count",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "How many singers do we have?",
        "sql": "SELECT COUNT(*) FROM singer"
    },
    {
        "id": 2,
        "category": "select_order",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "List the name, country and age for all singers ordered by age from the oldest to the youngest.",
        "sql": "SELECT name, country, age FROM singer ORDER BY age DESC"
    },
    {
        "id": 3,
        "category": "aggregation_where",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "What is the average, minimum, and maximum age of all singers from France?",
        "sql": "SELECT AVG(age), MIN(age), MAX(age) FROM singer WHERE country = 'France'"
    },
    {
        "id": 4,
        "category": "distinct_where",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "Show the distinct countries where singers above age 20 are from.",
        "sql": "SELECT DISTINCT country FROM singer WHERE age > 20"
    },
    {
        "id": 5,
        "category": "group_by",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "How many singers do we have from each country?",
        "sql": "SELECT country, COUNT(*) FROM singer GROUP BY country"
    },
    {
        "id": 6,
        "category": "subquery",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER), song_name (TEXT)",
        "question": "Show names of songs whose singer's age is older than the average age.",
        "sql": "SELECT song_name FROM singer WHERE age > (SELECT AVG(age) FROM singer)"
    },
    {
        "id": 7,
        "category": "join_group",
        "schema": "Table concert: concert_id (INTEGER), concert_name (TEXT), stadium_id (INTEGER)\nTable stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER)",
        "question": "Show the stadium name and the number of concerts in each stadium.",
        "sql": "SELECT T2.name, COUNT(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id"
    },
    {
        "id": 8,
        "category": "between",
        "schema": "Table stadium: stadium_id (INTEGER), location (TEXT), name (TEXT), capacity (INTEGER)",
        "question": "Show the location and name of stadiums which have some concerts with capacity between 5000 and 10000.",
        "sql": "SELECT location, name FROM stadium WHERE capacity BETWEEN 5000 AND 10000"
    },
    {
        "id": 9,
        "category": "not_in",
        "schema": "Table stadium: stadium_id (INTEGER), name (TEXT)\nTable concert: concert_id (INTEGER), stadium_id (INTEGER)",
        "question": "Show the stadium names without any concert.",
        "sql": "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)"
    },
    {
        "id": 10,
        "category": "intersect",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), age (INTEGER)",
        "question": "What are the countries that have both singers above age 40 and singers below age 30?",
        "sql": "SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30"
    },
    {
        "id": 11,
        "category": "union",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), country (TEXT), birth_year (INTEGER)",
        "question": "Show all birth years of singers from France or singers from USA.",
        "sql": "SELECT birth_year FROM singer WHERE country = 'France' UNION SELECT birth_year FROM singer WHERE country = 'USA'"
    },
    {
        "id": 12,
        "category": "except",
        "schema": "Table stadium: stadium_id (INTEGER), name (TEXT)\nTable concert: concert_id (INTEGER), stadium_id (INTEGER), year (INTEGER)",
        "question": "Find names of stadiums that did not have a concert in 2014.",
        "sql": "SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.year = 2014"
    },
    {
        "id": 13,
        "category": "like",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT), song_name (TEXT)",
        "question": "List all singers whose song name contains 'Hey'.",
        "sql": "SELECT name FROM singer WHERE song_name LIKE '%Hey%'"
    },
    {
        "id": 14,
        "category": "having",
        "schema": "Table student: stuid (INTEGER), fname (TEXT), sex (TEXT)\nTable has_pet: stuid (INTEGER), petid (INTEGER)",
        "question": "Find the first name and gender of students who have more than one pet.",
        "sql": "SELECT T1.fname, T1.sex FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid GROUP BY T1.stuid HAVING COUNT(*) > 1"
    },
    {
        "id": 15,
        "category": "multiple_joins",
        "schema": "Table singer: singer_id (INTEGER), name (TEXT)\nTable singer_in_concert: singer_id (INTEGER), concert_id (INTEGER)\nTable concert: concert_id (INTEGER), concert_name (TEXT), year (INTEGER)",
        "question": "Show names of singers who performed in concerts in 2014.",
        "sql": "SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id JOIN concert AS T3 ON T1.concert_id = T3.concert_id WHERE T3.year = 2014"
    },
    {
        "id": 16,
        "category": "max_limit",
        "schema": "Table stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER), average (REAL)",
        "question": "Show the stadium name and capacity with highest average attendance.",
        "sql": "SELECT name, capacity FROM stadium ORDER BY average DESC LIMIT 1"
    },
    {
        "id": 17,
        "category": "or_condition",
        "schema": "Table concert: concert_id (INTEGER), year (INTEGER)",
        "question": "How many concerts are there in 2014 or 2015?",
        "sql": "SELECT COUNT(*) FROM concert WHERE year = 2014 OR year = 2015"
    },
    {
        "id": 18,
        "category": "complex_join_group",
        "schema": "Table concert: concert_id (INTEGER), stadium_id (INTEGER), year (INTEGER)\nTable stadium: stadium_id (INTEGER), name (TEXT), capacity (INTEGER)",
        "question": "Show stadium name and capacity for the stadium with most concerts after 2013, and sort by the number of concerts in descending order.",
        "sql": "SELECT T2.name, T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.year > 2013 GROUP BY T2.stadium_id ORDER BY COUNT(*) DESC LIMIT 1"
    },
    {
        "id": 19,
        "category": "nested_subquery",
        "schema": "Table concert: concert_id (INTEGER), stadium_id (INTEGER)\nTable stadium: stadium_id (INTEGER), capacity (INTEGER)",
        "question": "How many concerts occurred in the stadium with the largest capacity?",
        "sql": "SELECT COUNT(*) FROM concert WHERE stadium_id = (SELECT stadium_id FROM stadium ORDER BY capacity DESC LIMIT 1)"
    },
    {
        "id": 20,
        "category": "multiple_aggregations",
        "schema": "Table pets: petid (INTEGER), weight (REAL), pettype (TEXT)",
        "question": "Find the maximum and minimum weight for each pet type.",
        "sql": "SELECT MAX(weight), pettype FROM pets GROUP BY pettype"
    }
]


# ============== HELPER FUNCTIONS ==============

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


def detect_query_pattern(question: str) -> Dict[str, str]:
    """
    Detect SQL patterns from question keywords
    Returns hints for SQL generation
    """
    question_lower = question.lower()
    hints = []
    
    # Set operations
    if 'both' in question_lower and ('and' in question_lower or 'as well as' in question_lower):
        hints.append("Use INTERSECT for items matching BOTH conditions")
    
    if 'either' in question_lower or ('or' in question_lower and 'either' in question_lower):
        hints.append("Use UNION for items matching EITHER condition")
    
    # Exclusion patterns
    if any(word in question_lower for word in ['not', 'without', 'exclude', "don't", 'never']):
        hints.append("Use NOT IN or EXCEPT for exclusion")
    
    # Aggregation patterns
    if any(word in question_lower for word in ['each', 'every', 'per', 'for each']):
        hints.append("Use GROUP BY for aggregating per group")
    
    if any(word in question_lower for word in ['most', 'highest', 'largest', 'maximum', 'top']):
        hints.append("Use ORDER BY DESC LIMIT 1 for maximum")
    
    if any(word in question_lower for word in ['least', 'lowest', 'smallest', 'minimum', 'bottom']):
        hints.append("Use ORDER BY ASC LIMIT 1 for minimum")
    
    # Pattern matching
    if any(word in question_lower for word in ['contain', 'include', 'has', 'with']):
        hints.append("Use LIKE with % wildcards for pattern matching")
    
    # Comparison
    if 'between' in question_lower:
        hints.append("Use BETWEEN for range queries")
    
    # Filtering after aggregation
    if any(word in question_lower for word in ['more than', 'less than', 'greater than']) and any(word in question_lower for word in ['each', 'every', 'per']):
        hints.append("Use HAVING for filtering grouped results")
    
    return {
        'hints': hints,
        'has_set_operation': any(word in question_lower for word in ['both', 'either']),
        'has_exclusion': any(word in question_lower for word in ['not', 'without']),
        'has_aggregation': any(word in question_lower for word in ['each', 'every', 'count', 'average', 'sum']),
        'has_pattern': any(word in question_lower for word in ['contain', 'include'])
    }


def setup_llamaindex():
    """Setup LlamaIndex with optimized parameters"""
    
    print(f"\n{'='*80}")
    print("SETTING UP LLAMAINDEX IMPROVED (HYBRID RAG + TEXT-TO-SQL)")
    print(f"{'='*80}")
    
    # Embedding model
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # LLM model with optimized parameters
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        request_timeout=120.0,
        additional_kwargs={
            'num_predict': LLM_NUM_PREDICT,
            'stop': LLM_STOP_SEQUENCES
        }
    )
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print(f"  ✓ Embedding Model: {EMBEDDING_MODEL}")
    print(f"  ✓ LLM Model: {LLM_MODEL}")
    print(f"  ✓ Temperature: {LLM_TEMPERATURE}")
    print(f"  ✓ Top-P: {LLM_TOP_P}")
    print(f"  ✓ Few-Shot Examples: {USE_FEW_SHOT_EXAMPLES}")
    print(f"  ✓ Top-K Examples: {TOP_K_EXAMPLES}")
    print(f"  ✓ Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"  ✓ Pattern Hints: {ENABLE_PATTERN_HINTS}")
    print(f"  ✓ Include Sample Rows: {INCLUDE_SAMPLE_ROWS}")
    print(f"{'='*80}\n")
    
    return embed_model, llm


def create_few_shot_index(examples: List[Dict], force_rebuild=False):
    """Create or load VectorStoreIndex from few-shot examples"""
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "few_shot_index")
    
    # Check if index exists
    if os.path.exists(index_path) and not force_rebuild:
        print(f"\n{'='*60}")
        print("LOADING EXISTING FEW-SHOT INDEX")
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
    print("BUILDING NEW FEW-SHOT INDEX")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    
    # ✅ IMPROVED: Convert examples to Documents with CLEAN format (like manual)
    documents = []
    for ex in examples:
        # Simple, clean format (matching manual version)
        text = f"""# EXAMPLE {ex['id']} - {ex['category'].upper()}
Schema: {ex['schema']}
Question: {ex['question']}
SQL: {ex['sql']}
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


def retrieve_relevant_examples(index, question: str, top_k: int = 5, threshold: float = 0.25):
    """Retrieve relevant few-shot examples using semantic search with threshold"""
    
    # Create retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k * 2  # Retrieve more for filtering
    )
    
    # Retrieve nodes
    nodes = retriever.retrieve(question)
    
    # Extract examples with threshold filtering
    examples = []
    for node in nodes:
        # Filter by similarity threshold
        if node.score < threshold:
            continue
        
        examples.append({
            'score': node.score,
            'category': node.metadata.get('category', 'unknown'),
            'question': node.metadata.get('question', ''),
            'sql': node.metadata.get('sql', ''),
            'schema': node.metadata.get('schema', '')
        })
        
        # Stop if we have enough examples
        if len(examples) >= top_k:
            break
    
    # Fallback: if no examples pass threshold, take best 3
    if len(examples) == 0:
        examples = [
            {
                'score': node.score,
                'category': node.metadata.get('category', 'unknown'),
                'question': node.metadata.get('question', ''),
                'sql': node.metadata.get('sql', ''),
                'schema': node.metadata.get('schema', '')
            }
            for node in nodes[:3]
        ]
    
    return examples


def create_sql_database(db_path: str):
    """Create SQLDatabase object from SQLite file"""
    
    try:
        # Create SQLAlchemy engine
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Get all tables
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table_names = list(metadata.tables.keys())
        
        # Filter out sqlite internal tables
        table_names = [t for t in table_names if not t.startswith('sqlite_')]
        
        # Create LlamaIndex SQLDatabase
        sql_database = SQLDatabase(
            engine,
            include_tables=table_names,
            sample_rows_in_table_info=3 if INCLUDE_SAMPLE_ROWS else 0
        )
        
        return sql_database, table_names
        
    except Exception as e:
        print(f"  [ERROR] Failed to create SQL database: {e}")
        return None, []


def create_custom_text_to_sql_prompt(few_shot_examples: List[Dict], pattern_hints: List[str]) -> PromptTemplate:
    """
    ✅ IMPROVED: Create custom Text-to-SQL prompt with CLEAN format (like manual)
    """
    
    # ✅ Format few-shot examples (CLEAN like manual)
    examples_text = ""
    for i, ex in enumerate(few_shot_examples, 1):
        examples_text += f"""# EXAMPLE {i} (Similarity: {ex['score']:.3f})
Schema: {ex['schema']}
Question: {ex['question']}
SQL: {ex['sql']}

"""
    
    # Format pattern hints
    hints_text = ""
    if pattern_hints:
        hints_text = "\n\n" + "\n".join([f"- {hint}" for hint in pattern_hints])
    
    # ✅ IMPROVED: Create SIMPLER prompt template (matching manual style)
    template = """You are an expert SQL query generator. Study these examples carefully:

{examples_text}

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
{hints_text}

Database Schema:
{schema}

Question: {query_str}

SQL Query (single line, lowercase keywords, no explanations):"""
    
    return PromptTemplate(template)


def create_hybrid_query_engine(sql_database, few_shot_index, question: str):
    """
    Create hybrid query engine combining:
    1. Few-shot example retrieval (from manual approach)
    2. LlamaIndex native Text-to-SQL engine
    3. Dynamic pattern hints
    """
    
    # 1. Retrieve relevant few-shot examples
    relevant_examples = []
    pattern_hints = []
    
    if USE_FEW_SHOT_EXAMPLES:
        relevant_examples = retrieve_relevant_examples(
            few_shot_index, 
            question, 
            top_k=TOP_K_EXAMPLES,
            threshold=SIMILARITY_THRESHOLD
        )
    
    # 2. Detect query patterns and generate hints
    if ENABLE_PATTERN_HINTS:
        pattern_info = detect_query_pattern(question)
        pattern_hints = pattern_info['hints']
    
    # 3. Create custom prompt with examples and hints
    text_to_sql_prompt = create_custom_text_to_sql_prompt(relevant_examples, pattern_hints)
    
    # 4. Create NLSQLTableQueryEngine with custom prompt
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=sql_database.get_usable_table_names(),
        text_to_sql_prompt=text_to_sql_prompt,
        synthesize_response=False,
        streaming=False
    )
    
    return query_engine, relevant_examples, pattern_hints


def normalize_sql_to_single_line(sql: str) -> str:
    """
    Normalize SQL query to single line format
    Compatible with Spider evaluation format
    """
    
    if not sql:
        return "SELECT 1"
    
    # 1. Remove markdown code blocks
    sql = sql.replace('```sql', '').replace('```', '').strip()
    
    # 2. Split into lines and remove comments
    lines = []
    for line in sql.split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip comment lines
        if line.startswith('--') or line.startswith('#'):
            continue
        
        # Remove inline comments
        if '--' in line:
            comment_pos = line.find('--')
            line = line[:comment_pos].strip()
        
        if line:
            lines.append(line)
    
    # 3. Join lines with space
    sql = ' '.join(lines)
    
    # 4. Remove semicolon
    sql = sql.rstrip(';').strip()
    
    # 5. Collapse multiple whitespaces
    sql = re.sub(r'\s+', ' ', sql)
    
    # 6. Fix comma spacing
    sql = re.sub(r'\s*,\s*', ', ', sql)
    
    # 7. Fix parentheses spacing
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    
    # 8. Fix operator spacing
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*>\s*', ' > ', sql)
    sql = re.sub(r'\s*<\s*', ' < ', sql)
    sql = re.sub(r'\s*!=\s*', ' != ', sql)
    sql = re.sub(r'\s*<=\s*', ' <= ', sql)
    sql = re.sub(r'\s*>=\s*', ' >= ', sql)
    
    # 9. Final whitespace normalization
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # 10. Validate
    if len(sql) < 10:
        return "SELECT 1"
    
    sql_lower = sql.lower()
    if not any(kw in sql_lower for kw in ['select', 'insert', 'update', 'delete']):
        return "SELECT 1"
    
    return sql


def extract_sql_from_response(response) -> str:
    """Extract and normalize SQL query from LlamaIndex response"""
    
    # Method 1: Check metadata
    if hasattr(response, 'metadata') and 'sql_query' in response.metadata:
        raw_sql = response.metadata['sql_query'].strip()
    else:
        # Method 2: Parse from response text
        raw_sql = str(response)
    
    # Normalize to single line
    normalized_sql = normalize_sql_to_single_line(raw_sql)
    
    return normalized_sql


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


# ============== MAIN FUNCTION ==============

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_llm_name = get_safe_filename(LLM_MODEL)
    
    predictions_file = os.path.join(OUTPUT_DIR, f"prediksi_improved_{safe_llm_name}_{timestamp}.txt")
    metrics_file = os.path.join(OUTPUT_DIR, f"metric_improved_{safe_llm_name}_{timestamp}.csv")
    retrieval_log_file = os.path.join(OUTPUT_DIR, f"retrieval_log_improved_{timestamp}.json")
    
    print(f"\n{'='*80}")
    print(f"SQL GENERATION WITH LLAMAINDEX IMPROVED (HYBRID APPROACH)")
    print(f"{'='*80}")
    print(f"Approach: LlamaIndex Native + Few-Shot Learning + Pattern Hints")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Few-Shot Examples: {len(FEW_SHOT_EXAMPLES_RAW)} (Top-{TOP_K_EXAMPLES} retrieved)")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Pattern Hints: {ENABLE_PATTERN_HINTS}")
    print(f"Include Sample Rows: {INCLUDE_SAMPLE_ROWS}")
    print(f"\nOutput files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    print(f"  3. {retrieval_log_file}")
    print(f"{'='*80}\n")
    
    # Setup
    embed_model, llm = setup_llamaindex()
    
    # Create/Load few-shot index
    few_shot_index = None
    if USE_FEW_SHOT_EXAMPLES:
        few_shot_index = create_few_shot_index(FEW_SHOT_EXAMPLES_RAW, force_rebuild=False)
    
    # Load questions
    if not os.path.exists(GOLD_FILE):
        print(f"ERROR: {GOLD_FILE} not found!")
        return
    
    questions = parse_gold_file(GOLD_FILE)
    print(f"Loaded {len(questions)} questions\n")
    
    # Generation
    metrics_data = []
    predictions = []
    retrieval_logs = []
    
    print(f"{'='*80}")
    print(f"STARTING HYBRID SQL GENERATION")
    print(f"{'='*80}\n")
    
    total_start_time = time.time()
    error_count = 0
    fallback_count = 0
    
    # Cache for databases
    db_cache = {}
    
    for i, q in enumerate(questions):
        q_id = q['id']
        question = q['question']
        db_id = q['db_id']
        
        print(f"[{i+1}/{len(questions)}] Q{q_id} ({db_id})")
        print(f"  Question: {question[:70]}...")
        
        # Get database path
        db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            print(f"  [ERROR] Database not found: {db_path}")
            predictions.append(f"SELECT 1\t{db_id}")
            error_count += 1
            continue
        
        start_time = time.time()
        
        try:
            # Create or get cached SQL database
            if db_id not in db_cache:
                print(f"  [INFO] Loading database: {db_id}")
                sql_database, table_names = create_sql_database(db_path)
                
                if sql_database is None:
                    raise Exception("Failed to create SQL database")
                
                print(f"  [INFO] Found {len(table_names)} tables: {', '.join(table_names)}")
                db_cache[db_id] = {'sql_database': sql_database, 'tables': table_names}
            else:
                sql_database = db_cache[db_id]['sql_database']
            
            # Create hybrid query engine with few-shot examples and pattern hints
            print(f"  [HYBRID] Creating query engine with few-shot + hints...")
            query_engine, relevant_examples, pattern_hints = create_hybrid_query_engine(
                sql_database, 
                few_shot_index, 
                question
            )
            
            # Log retrieval info
            if relevant_examples:
                categories = [ex['category'] for ex in relevant_examples]
                scores = [ex['score'] for ex in relevant_examples]
                print(f"  [HYBRID] Retrieved {len(relevant_examples)} examples: {', '.join(categories)}")
                print(f"  [HYBRID] Similarity scores: {', '.join([f'{s:.3f}' for s in scores])}")
            
            if pattern_hints:
                print(f"  [HYBRID] Pattern hints: {len(pattern_hints)} detected")
                for hint in pattern_hints:
                    print(f"    - {hint}")
            
            # Generate SQL
            print(f"  [HYBRID] Generating SQL with LlamaIndex engine...")
            response = query_engine.query(question)
            
            # Extract and normalize SQL
            pred_sql = extract_sql_from_response(response)
            
            # Validate
            if not pred_sql or len(pred_sql) < 10:
                print(f"  [WARNING] Empty/short SQL, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1
            elif not any(kw in pred_sql.lower() for kw in ['select', 'insert', 'update', 'delete']):
                print(f"  [WARNING] No SQL keyword, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1
            
            # Log retrieval details
            retrieval_logs.append({
                'question_id': q_id,
                'question': question,
                'retrieved_examples': [
                    {
                        'category': ex['category'],
                        'similarity': ex['score'],
                        'question': ex['question'],
                        'sql': ex['sql']
                    } for ex in relevant_examples
                ],
                'pattern_hints': pattern_hints,
                'generated_sql': pred_sql
            })
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            pred_sql = "SELECT 1"
            error_count += 1
            relevant_examples = []
            pattern_hints = []
        
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
            'method': 'Hybrid (LlamaIndex + Few-Shot + Hints)',
            'num_examples_retrieved': len(relevant_examples),
            'retrieved_categories': ','.join([ex['category'] for ex in relevant_examples]) if relevant_examples else '',
            'num_pattern_hints': len(pattern_hints),
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
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"  [OK] Saved predictions: {predictions_file}")
    
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'question', 'predicted_sql', 
                      'latency_seconds', 'llm_model', 'embedding_model', 
                      'method', 'num_examples_retrieved', 'retrieved_categories',
                      'num_pattern_hints', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"  [OK] Saved metrics: {metrics_file}")
    
    with open(retrieval_log_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_logs, f, indent=2)
    print(f"  [OK] Saved retrieval log: {retrieval_log_file}\n")
    
    # Validate
    valid_count, invalid_count = validate_prediction_file(predictions_file)
    
    # Summary
    print(f"{'='*80}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Method: Hybrid (LlamaIndex Native + Few-Shot Learning + Pattern Hints)")
    print(f"LLM: {LLM_MODEL} (temp={LLM_TEMPERATURE}, top_p={LLM_TOP_P})")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Few-Shot Examples: {len(FEW_SHOT_EXAMPLES_RAW)} (Top-{TOP_K_EXAMPLES} retrieved)")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
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