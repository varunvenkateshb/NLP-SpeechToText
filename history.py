import sqlite3
from datetime import datetime

# Function to check and setup the history database schema
def setup_history_database():
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()

    # Check if 'history' table already exists
    c.execute("PRAGMA table_info(history);")
    columns = [column[1] for column in c.fetchall()]

    # If 'timestamp' column is missing, drop and recreate the table with the correct schema
    if 'timestamp' not in columns:
        print("Recreating the history table with the correct schema...")

        # Drop the table if it exists (WARNING: This will delete all existing records)
        c.execute("DROP TABLE IF EXISTS history;")

        # Recreate the table with the correct schema
        c.execute('''
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                action_type TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
    conn.close()

# Function to add a translation record to the history
def add_translation_to_history(user_id, input_text, translated_text, source_lang, target_lang, action_type):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO history (user_id, input_text, translated_text, source_lang, target_lang, action_type, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, input_text, translated_text, source_lang, target_lang, action_type, timestamp))
    conn.commit()
    conn.close()

# Function to get the translation history for a specific user
def get_user_history(user_id):
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE user_id = ? ORDER BY date DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# Group history records by date for easy review
def group_history_by_date(user_id):
    history = get_user_history(user_id)
    grouped_history = {}

    for record in history:
        date = record[5].split(' ')[0]
        if date not in grouped_history:
            grouped_history[date] = []
        grouped_history[date].append(record)

    return grouped_history

# Delete history for a user
def delete_history(user_id):
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

# Get a specific history record by ID
def get_history_by_id(record_id):
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute('''
        SELECT id, input_text, translated_text, source_lang, target_lang, action_type, timestamp
        FROM history
        WHERE id = ?
    ''', (record_id,))
    record = c.fetchone()
    conn.close()
    return record

# Call the setup function on startup to ensure the schema is correct
setup_history_database()
