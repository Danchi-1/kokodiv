import sqlite3
import json
import os
import uuid
from typing import Dict, Any, Optional, List

class CocoaRAGRetriever:
    """
    Sub-5ms SQLite RAG Context Retriever & Chat Memory Engine for Kokodiv.
    """
    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite RAG database not found at: {db_path}")
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_disease_profile(self, disease_id: str, language_code: str = "en") -> Optional[Dict[str, Any]]:
        """
        Retrieves complete disease profile for a target disease and language.
        Fallback to English if target language translation is missing.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                d.disease_id, d.pathogen, d.severity, d.crin_reference, d.yield_impact, d.nigerian_regions,
                t.common_name, t.symptoms, t.treatment_steps, t.prevention
            FROM diseases d
            LEFT JOIN disease_translations t ON d.disease_id = t.disease_id AND t.language_code = ?
            WHERE d.disease_id = ?
        """, (language_code, disease_id))

        row = cursor.fetchone()

        if row and row["common_name"] is None and language_code != "en":
            cursor.execute("""
                SELECT 
                    d.disease_id, d.pathogen, d.severity, d.crin_reference, d.yield_impact, d.nigerian_regions,
                    t.common_name, t.symptoms, t.treatment_steps, t.prevention
                FROM diseases d
                LEFT JOIN disease_translations t ON d.disease_id = t.disease_id AND t.language_code = 'en'
                WHERE d.disease_id = ?
            """, (disease_id,))
            row = cursor.fetchone()

        conn.close()

        if not row:
            return None

        treatment_list = json.loads(row["treatment_steps"]) if row["treatment_steps"] else []

        return {
            "disease_id": row["disease_id"],
            "common_name": row["common_name"],
            "pathogen": row["pathogen"],
            "severity": row["severity"],
            "symptoms": row["symptoms"],
            "treatment": treatment_list,
            "prevention": row["prevention"],
            "yield_impact": row["yield_impact"],
            "crin_reference": row["crin_reference"],
            "regions": row["nigerian_regions"].split(", ") if row["nigerian_regions"] else []
        }

    def search_agronomy_topics(self, query: str, language_code: str = "en", limit: int = 2) -> List[Dict[str, Any]]:
        """
        Text-only Agronomy RAG Search: Searches symptoms, treatments, and common names for text-only questions.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        search_pattern = f"%{query.lower().strip()}%"

        cursor.execute("""
            SELECT DISTINCT d.disease_id
            FROM diseases d
            JOIN disease_translations t ON d.disease_id = t.disease_id
            WHERE LOWER(t.symptoms) LIKE ? 
               OR LOWER(t.treatment_steps) LIKE ? 
               OR LOWER(t.common_name) LIKE ?
               OR LOWER(d.pathogen) LIKE ?
            LIMIT ?
        """, (search_pattern, search_pattern, search_pattern, search_pattern, limit))

        rows = cursor.fetchall()
        conn.close()

        disease_ids = [r["disease_id"] for r in rows]
        if not disease_ids:
            disease_ids = ["black_pod", "mirid"] # Default core references

        profiles = []
        for dis_id in disease_ids:
            prof = self.get_disease_profile(dis_id, language_code)
            if prof:
                profiles.append(prof)
        return profiles

    def format_rag_context_for_prompt(self, disease_ids: List[str], language_code: str = "en") -> str:
        """
        Formats top retrieved disease profiles into structured text block for LLM prompt context.
        """
        context_blocks = []
        for idx, dis_id in enumerate(disease_ids, 1):
            profile = self.get_disease_profile(dis_id, language_code)
            if not profile:
                continue

            treatment_str = "\n".join([f"  - {step}" for step in profile["treatment"]])
            block = (
                f"--- CRIN AGRONOMY PROFILE #{idx}: {profile['common_name']} ({profile['disease_id']}) ---\n"
                f"Pathogen: {profile['pathogen']}\n"
                f"Severity Level: {profile['severity'].upper()}\n"
                f"Symptoms: {profile['symptoms']}\n"
                f"Treatment Protocol:\n{treatment_str}\n"
                f"Prevention Measure: {profile['prevention']}\n"
                f"CRIN Bulletin Ref: {profile['crin_reference']}\n"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    # Conversational Chat Memory Methods
    def create_chat_session(self, title: str = "New Diagnosis Session", language_code: str = "en") -> str:
        session_id = str(uuid.uuid4())[:8]
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_sessions (session_id, title, language_code)
            VALUES (?, ?, ?)
        """, (session_id, title, language_code))
        conn.commit()
        conn.close()
        return session_id

    def list_chat_sessions(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, title, created_at, language_code FROM chat_sessions ORDER BY created_at DESC LIMIT 30")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_chat_message(self, session_id: str, role: str, content: str, disease_detected: str = None, confidence_score: float = None, has_image: bool = False):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, content, disease_detected, confidence_score, has_image)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, disease_detected, confidence_score, 1 if has_image else 0))
        conn.commit()
        conn.close()

    def get_chat_messages(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT role, content, disease_detected, confidence_score, has_image, created_at FROM chat_messages WHERE session_id = ? ORDER BY message_id ASC", (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
