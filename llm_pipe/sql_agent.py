import gradio as gr
import pandas as pd
import sqlite3
import time
from difflib import get_close_matches
from config import app_config
import requests
import re
import os


# Configure Ollama API - use localhost as Ollama should be running locally
OLLAMA_API_BASE = app_config.OLLAMA_API_BASE
OLLAMA_MODEL = app_config.OLLAMA_MODEL

def check_ollama_available():
    """Check if Ollama is available by sending a request to list models"""
    try:
        print(f"Trying to connect to Ollama at {OLLAMA_API_BASE}...")
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            if OLLAMA_MODEL in available_models:
                print(f"Ollama is available with {OLLAMA_MODEL}")
                return True
            else:
                print(f"Warning: {OLLAMA_MODEL} not found in Ollama. Available models: {available_models}")
                return False
        else:
            print(f"Ollama API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        return False
    
ollama_available = check_ollama_available()

# SQL CODE START FROM HERE
# Function to get available models from Ollama
def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)  # Fixed: removed duplicate '/api'
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models.get('models', [])]
        return []
    except:
        return []

# Function to find closest column match
def find_closest_column(target_word, columns):
    """Find the closest matching column name using fuzzy matching"""
    # Convert to lowercase for comparison
    target_lower = target_word.lower()
    columns_lower = [col.lower() for col in columns]
    
    # Direct match first
    if target_lower in columns_lower:
        idx = columns_lower.index(target_lower)
        return columns[idx]
    
    # Try partial matches
    for i, col in enumerate(columns_lower):
        if target_lower in col or col in target_lower:
            return columns[i]
    
    # Use difflib for fuzzy matching
    matches = get_close_matches(target_lower, columns_lower, n=1, cutoff=0.6)
    if matches:
        idx = columns_lower.index(matches[0])
        return columns[idx]
    
    return None

# Enhanced SQL generation with better column matching
def generate_sql_with_ollama(query, table_name, columns, model=OLLAMA_MODEL):
    # Create a more detailed prompt with column information
    column_info = "\n".join([f"- {col}" for col in columns])
    
    prompt = f"""
You are an expert SQL query generator. Given a natural language question and database schema, generate a precise SQL query.

Database Information:
- Table name: {table_name}
- Available columns:
{column_info}

IMPORTANT RULES:
1. Generate ONLY the SQL query, no explanations or markdown
2. Use proper SQL syntax for SQLite
3. Column names must EXACTLY match the available columns listed above
4. Always use LIMIT clause for SELECT queries (default LIMIT 10)
5. For aggregation queries (COUNT, SUM, AVG, etc.), don't use LIMIT unless grouping
6. Use LIKE operator with % wildcards for text searches
7. If a column name in the question doesn't exist, find the closest matching column from the list above

Natural language question: {query}

SQL Query:"""

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 150
            }
        }
        
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",  # Fixed: removed duplicate '/api'
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            sql_query = result['response'].strip()
            
            # Clean up the response
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = re.sub(r'^SQL Query:\s*', '', sql_query, flags=re.IGNORECASE)
            
            # Extract the first line that looks like a SQL query
            lines = sql_query.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.upper().startswith(('SELECT', 'PRAGMA', 'SHOW', 'DESC', 'WITH'))):
                    return line
            
            return sql_query.split('\n')[0].strip() if sql_query else None
            
        else:
            return None
            
    except requests.exceptions.Timeout:
        print("Ollama request timed out")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

# Enhanced fallback function with better column matching
def generate_sql_fallback(query, table_name, columns):
    query_lower = query.lower()
    
    # Extract potential column names from the query
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    # For displaying all columns
    if any(phrase in query_lower for phrase in ["show all columns", "list all columns", "display all columns", "what columns", "which columns"]):
        return f"PRAGMA table_info({table_name})"
    
    # Find relevant columns mentioned in the query
    relevant_columns = []
    for word in query_words:
        closest_col = find_closest_column(word, columns)
        if closest_col and closest_col not in relevant_columns:
            relevant_columns.append(closest_col)
    
    # Detect query type and generate appropriate SQL
    if any(word in query_lower for word in ["average", "avg", "mean"]):
        if relevant_columns:
            # Try to find numeric columns
            numeric_cols = [col for col in relevant_columns if any(num_word in col.lower() for num_word in ['price', 'cost', 'amount', 'value', 'salary', 'age', 'count', 'number', 'score', 'rating'])]
            if numeric_cols:
                return f"SELECT AVG({numeric_cols[0]}) as average_{numeric_cols[0]} FROM {table_name}"
            else:
                return f"SELECT AVG({relevant_columns[0]}) as average_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} LIMIT 5"
    
    elif any(word in query_lower for word in ["sum", "total"]):
        if relevant_columns:
            numeric_cols = [col for col in relevant_columns if any(num_word in col.lower() for num_word in ['price', 'cost', 'amount', 'value', 'salary', 'count', 'number', 'score'])]
            if numeric_cols:
                return f"SELECT SUM({numeric_cols[0]}) as total_{numeric_cols[0]} FROM {table_name}"
            else:
                return f"SELECT SUM({relevant_columns[0]}) as total_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} LIMIT 5"
    
    elif any(word in query_lower for word in ["maximum", "max", "highest", "top"]):
        if relevant_columns:
            return f"SELECT MAX({relevant_columns[0]}) as max_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} ORDER BY {columns[0]} DESC LIMIT 5"
    
    elif any(word in query_lower for word in ["minimum", "min", "lowest", "bottom"]):
        if relevant_columns:
            return f"SELECT MIN({relevant_columns[0]}) as min_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} ORDER BY {columns[0]} ASC LIMIT 5"
    
    elif "count" in query_lower:
        if relevant_columns:
            return f"SELECT COUNT({relevant_columns[0]}) as count_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT COUNT(*) as total_count FROM {table_name}"
    
    elif any(word in query_lower for word in ["show", "display", "list", "all", "get", "find"]):
        limit = 10
        limit_match = re.search(r"(\d+)", query)
        if limit_match:
            limit = int(limit_match.group(1))
        
        if relevant_columns:
            return f"SELECT {', '.join(relevant_columns)} FROM {table_name} LIMIT {limit}"
        else:
            return f"SELECT * FROM {table_name} LIMIT {limit}"
    
    else:
        # Default: show sample data
        return f"SELECT * FROM {table_name} LIMIT 5"

# Function to create an SQLite database from a CSV file
def create_database_from_csv(csv_file_path):
    db_name = f"rag_database_{int(time.time())}.db"
    
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try different encodings if utf-8 fails
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read CSV file with any supported encoding")
    
    # Clean column names: remove special characters and convert to lowercase
    original_columns = df.columns.tolist()
    df.columns = [re.sub(r'[^\w]', '_', col).lower().strip('_') for col in df.columns]
    
    # Handle duplicate column names
    seen_columns = {}
    new_columns = []
    for col in df.columns:
        if col in seen_columns:
            seen_columns[col] += 1
            new_columns.append(f"{col}_{seen_columns[col]}")
        else:
            seen_columns[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    
    conn = sqlite3.connect(db_name)
    table_name = "data"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    return db_name, table_name, list(df.columns), original_columns

# Enhanced SQL execution with better error handling
def execute_sql_query(db_name, sql_query):
    try:
        conn = sqlite3.connect(db_name)
        
        if sql_query.strip().upper().startswith("PRAGMA"):
            cursor = conn.cursor()
            cursor.execute(sql_query)
            column_info = cursor.fetchall()
            df = pd.DataFrame(column_info, columns=["cid", "name", "type", "notnull", "default_value", "pk"])
            df = df[["name", "type"]]
            cursor.close()
            conn.close()
            return df
        else:
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
            
    except Exception as e:
        error_msg = str(e)
        if "no such column" in error_msg:
            # Extract the problematic column name
            column_match = re.search(r"no such column: (\w+)", error_msg)
            if column_match:
                problematic_column = column_match.group(1)
                return pd.DataFrame([{"Error": f"Column '{problematic_column}' not found in the data. Please check the available columns and try again."}])
        return pd.DataFrame([{"Error": f"Error executing SQL query: {error_msg}"}])

# Function to display columns when file is uploaded
def display_columns(csv_file):
    if csv_file is None:
        return "Please upload a CSV file first.", "No columns available"
    
    try:
        # Try to read with different encodings
        try:
            df = pd.read_csv(csv_file.name, encoding='utf-8')
        except UnicodeDecodeError:
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(csv_file.name, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return "❌ Error: Could not read CSV file with any supported encoding", "No columns available"
        
        cleaned_columns = [re.sub(r'[^\w]', '_', col).lower().strip('_') for col in df.columns]
        
        # Handle duplicate column names for display
        seen_columns = {}
        display_columns = []
        for col in cleaned_columns:
            if col in seen_columns:
                seen_columns[col] += 1
                display_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                display_columns.append(col)
        
        column_info = f"Available columns ({len(display_columns)}): {', '.join(display_columns)}"
        
        return f"✅ CSV uploaded: {len(df)} rows, {len(df.columns)} columns", column_info
        
    except Exception as e:
        return f"❌ Error reading CSV file: {str(e)}", "No columns available"

# Enhanced main processing function
def process_query_sql(csv_file, query, selected_model, use_ollama):
    if csv_file is None:
        return "Please upload a CSV file first.", "", pd.DataFrame()
    
    if not query.strip():
        return "Please enter a query.", "", pd.DataFrame()
    
    try:
        # Create database from CSV
        db_name, table_name, columns, original_columns = create_database_from_csv(csv_file.name)
        
        # Generate SQL query
        if use_ollama and check_ollama_available():
            sql_query = generate_sql_with_ollama(query, table_name, columns, selected_model)
            if sql_query is None:
                # Fallback to rule-based approach if Ollama fails
                sql_query = generate_sql_fallback(query, table_name, columns)
                status_msg = "⚠️ Ollama request failed/timed out, used fallback approach."
            else:
                status_msg = f"✅ Query generated using Ollama ({selected_model})."
        else:
            sql_query = generate_sql_fallback(query, table_name, columns)
            if use_ollama:
                status_msg = "⚠️ Ollama not available, used fallback approach."
            else:
                status_msg = "✅ Query generated using rule-based approach."
        
        # Execute SQL query
        result = execute_sql_query(db_name, sql_query)
        
        # Clean up the database file
        try:
            os.remove(db_name)
        except:
            pass
        
        return status_msg, sql_query, result
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}", "", pd.DataFrame()

# Function to update model dropdown
def refresh_models():
    models = get_available_models()
    if not models:
        return gr.Dropdown(choices=["No models available"], value="No models available")
    return gr.Dropdown(choices=models, value=models[0] if models else "")