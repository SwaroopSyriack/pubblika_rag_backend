import sqlite3


DB_NAME = "pubblika_rag_app.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_query TEXT,
    gpt_response TEXT,
    model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id,user_query,gpt_response,model):
    conn = get_connection()
    conn.execute('INSERT INTO application_logs(session_id,user_query,gpt_response,model) VALUES (?,?,?,?)',(session_id,user_query,gpt_response,model))
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('Select user_query,gpt_response from application_logs WHERE session_id = ? ORDER BY created_at',(session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role":"human","content": row["user_query"]},
            {"role":"ai","content":row["gpt_response"]}
            ])
    conn.close()
    return messages





