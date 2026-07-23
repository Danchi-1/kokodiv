import os
from typing import Dict, List, Tuple, Optional

from src.classifier.constants import DISEASE_DISPLAY_NAMES
from src.rag.retriever import CocoaRAGRetriever
from src.language.detector import FastLanguageDetector, SUPPORTED_LANGUAGES

SYSTEM_PROMPT_TEMPLATE = """You are Kokodiv, an expert cocoa farming doctor in West Africa.
Language: {language_name} ({language_code})
{pidgin_instruction}

CLASSIFIER: {classifier_confidence_note}
DISEASE CONTEXT:
{rag_retrieved_context}

User Question: {user_input}
Provide diagnosis and 3 simple treatment steps in {language_name}:"""

TEXT_CHAT_PROMPT_TEMPLATE = """You are Kokodiv, an expert cocoa farming doctor.
Respond concisely in {language_name} ({language_code}).
{pidgin_instruction}
{rag_retrieved_context}

User Question: {user_input}
Answer clearly in 2-3 bullet points:"""

class KokodivContextBuilder:
    """
    Central Pipeline Context Orchestrator for Kokodiv.
    Merges CNN confidence scores, RAG retrieved chunks, and language detection into the final LLM prompt.
    """
    def __init__(self, db_path: str, ft_model_path: Optional[str] = None):
        self.retriever = CocoaRAGRetriever(db_path)
        self.lang_detector = FastLanguageDetector(ft_model_path)

    def build_confidence_note(self, cnn_scores: Dict[str, float]) -> Tuple[str, List[str]]:
        """
        Applies PRD Section 9 Confidence Thresholding Logic.
        Returns formatted classifier note and list of top 2 candidate disease IDs.
        """
        sorted_scores = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
        top_disease, top_conf = sorted_scores[0]
        second_disease, second_conf = sorted_scores[1]

        top_display = DISEASE_DISPLAY_NAMES.get(top_disease, top_disease)
        second_display = DISEASE_DISPLAY_NAMES.get(second_disease, second_disease)

        candidate_ids = [top_disease, second_disease]

        if top_conf < 0.40:
            note = (
                f"CLASSIFIER NOTE: Low confidence ({top_conf:.0%}) — image quality may be poor "
                f"or disease presentation is atypical. Exercise additional caution.\n"
                f"Top candidate predictions: {top_display} ({top_conf:.0%}), {second_display} ({second_conf:.0%})"
            )
        elif top_conf < 0.65:
            note = (
                f"CLASSIFIER NOTE: Moderate confidence.\n"
                f"Primary prediction: {top_display} ({top_conf:.0%})\n"
                f"Differential candidate: {second_display} ({second_conf:.0%})"
            )
        else:
            note = (
                f"CLASSIFIER NOTE: High confidence.\n"
                f"Primary prediction: {top_display} ({top_conf:.0%})"
            )

        return note, candidate_ids

    def build_full_prompt(
        self,
        cnn_scores: Dict[str, float],
        user_input: str = "",
        override_language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Assembles complete LLM prompt ready for llama.cpp execution.
        """
        # Detect language or use manual override dropdown
        if override_language and override_language in SUPPORTED_LANGUAGES:
            target_lang = override_language
        else:
            target_lang = self.lang_detector.detect_language(user_input)

        lang_name = SUPPORTED_LANGUAGES.get(target_lang, "English")

        # Special instruction for Pidgin
        pidgin_inst = ""
        if target_lang == "pcm":
            pidgin_inst = "IF THE USER WRITES IN WEST AFRICA PIDGIN, RESPOND IN NATURAL WEST AFRICA PIDGIN."

        # 1. Apply confidence thresholding
        confidence_note, candidate_ids = self.build_confidence_note(cnn_scores)

        # 2. Retrieve RAG chunks in target language
        rag_context = self.retriever.format_rag_context_for_prompt(candidate_ids, language_code=target_lang)

        # 3. Format final prompt (Select fast text-only template vs photo diagnosis template)
        is_photo = bool(user_input and "image" in user_input.lower()) or (len(candidate_ids) > 0 and cnn_scores.get(candidate_ids[0], 0) > 0.6)
        
        if not is_photo and user_input.strip():
            template = TEXT_CHAT_PROMPT_TEMPLATE
        else:
            template = SYSTEM_PROMPT_TEMPLATE

        if not user_input.strip():
            user_input = "Please diagnose this cocoa plant image and provide treatment guidance."

        formatted_prompt = template.format(
            language_name=lang_name,
            language_code=target_lang,
            pidgin_instruction=pidgin_inst,
            classifier_confidence_note=confidence_note,
            rag_retrieved_context=rag_context,
            user_input=user_input
        )

        return {
            "prompt": formatted_prompt,
            "detected_language": target_lang,
            "language_name": lang_name,
            "confidence_note": confidence_note,
            "top_candidates": candidate_ids
        }
