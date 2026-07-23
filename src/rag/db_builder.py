#!/usr/bin/env python3
"""
Kokodiv — SQLite Database Builder & Schema Setup
Reads `data/disease_profiles.json` and builds indexed SQLite database `data/kokodiv.db`.

Tables:
- `diseases`: Core disease metadata
- `disease_translations`: Multilingual content
- `chat_sessions`: Local user conversation threads
- `chat_messages`: Local messages for conversational history persistence
"""

import os
import json
import sqlite3

def build_database(json_path: str, db_path: str):
    print(f"Reading disease profiles JSON from: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Core Disease Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diseases (
        disease_id TEXT PRIMARY KEY,
        pathogen TEXT NOT NULL,
        severity TEXT NOT NULL,
        crin_reference TEXT NOT NULL,
        yield_impact TEXT NOT NULL,
        nigerian_regions TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS disease_translations (
        translation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        disease_id TEXT NOT NULL,
        language_code TEXT NOT NULL,
        common_name TEXT NOT NULL,
        symptoms TEXT NOT NULL,
        treatment_steps TEXT NOT NULL,
        prevention TEXT NOT NULL,
        FOREIGN KEY (disease_id) REFERENCES diseases (disease_id),
        UNIQUE(disease_id, language_code)
    );
    """)

    # Local Conversational Chat Session Persistence Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        language_code TEXT DEFAULT 'en'
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        disease_detected TEXT,
        confidence_score REAL,
        has_image INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
    );
    """)

    # Fast Query Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_lookup ON disease_translations (disease_id, language_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_msg_session ON chat_messages (session_id);")

    print(f"Inserting {len(profiles)} disease profiles into SQLite...")

    for item in profiles:
        disease_id = item["disease_id"]
        pathogen = item["pathogen"]
        severity = item["severity"]
        crin_ref = item["crin_reference"]
        yield_impact = item["yield_impact"]
        regions_str = ", ".join(item.get("nigerian_regions", []))

        cursor.execute("""
        INSERT OR REPLACE INTO diseases (disease_id, pathogen, severity, crin_reference, yield_impact, nigerian_regions)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (disease_id, pathogen, severity, crin_ref, yield_impact, regions_str))

        languages = list(item["common_name"].keys())

        for lang in languages:
            name = item["common_name"].get(lang, item["common_name"]["en"])
            symptoms = item["symptoms"].get(lang, item["symptoms"]["en"])
            treatments = item["treatment"].get(lang, item["treatment"]["en"])
            treatment_json = json.dumps(treatments, ensure_ascii=False)
            prevention = item["prevention"].get(lang, item["prevention"]["en"])

            cursor.execute("""
            INSERT OR REPLACE INTO disease_translations (disease_id, language_code, common_name, symptoms, treatment_steps, prevention)
            VALUES ((SELECT disease_id FROM diseases WHERE disease_id=?), ?, ?, ?, ?, ?)
            """, (disease_id, lang, name, symptoms, treatment_json, prevention))

    conn.commit()
    conn.close()

    print(f"✅ Success! SQLite database updated at: {db_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    json_file = os.path.join(base_dir, "data", "disease_profiles.json")
    db_file = os.path.join(base_dir, "data", "kokodiv.db")
    build_database(json_file, db_file)
