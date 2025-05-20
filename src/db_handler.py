import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
# Configuration (replace with your credentials or use environment variables)
password = os.environ.get("DB_PASSWORD", "default")


DB_CONFIG = {
    "dbname": "FolderBasedSearcher",
    "user": "postgres",
    "password": password,
    "host": "localhost",
    "port": "5432"
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def insert_question_answer(question, answer, model):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chatbot_history (question, answer, model_id)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (question, answer, model))
            inserted_id = cur.fetchone()[0]
            conn.commit()
            return inserted_id

def fetch_all_logs():
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM chatbot_history ORDER BY timestamp DESC;")
            return cur.fetchall()
        
def insert_feedback(message_id, feedback_type):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE chatbot_history
                SET feedback = %s
                WHERE id = %s;
            """, (feedback_type, message_id))
            conn.commit()

