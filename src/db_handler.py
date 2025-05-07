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


def insert_question_answer(question, answer):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chatbot_history (question, answer) VALUES (%s, %s);
            """, (question, answer))
            conn.commit()

def fetch_all_logs():
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM chatbot_history ORDER BY timestamp DESC;")
            return cur.fetchall()
