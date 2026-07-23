import os
from typing import Dict, List, Tuple, Optional

from src.classifier.constants import DISEASE_DISPLAY_NAMES
from src.rag.retriever import CocoaRAGRetriever
from src.language.detector import FastLanguageDetector, SUPPORTED_LANGUAGES

SYSTEM_PROMPT_TEMPLATE = """SYSTEM:
You are Kokodiv — an expert cocoa disease diagnostician with deep knowledge 
of African cocoa farming, specifically Nigerian growing conditions. You help 
farmers identify diseases, understand treatments, and protect their yields.
You are precise, honest about uncertainty, and always practical.

RESPOND IN: {language_name} ({language_code})
{pidgin_instruction}

CLASSIFIER ANALYSIS:
{classifier_confidence_note}

RETRIEVED DISEASE PROFILES:
{rag_retrieved_context}

REASONING PROTOCOL:
Step 1 — Visual observation: Describe what you observe in the cocoa plant image
Step 2 — Cross-reference: Compare visual observations with classifier scores
Step 3 — Diagnosis: State your diagnosis with honest confidence level
Step 4 — If classifier confidence < 40%: explicitly state image quality or atypical limitations
Step 5 — Treatment: Provide specific, actionable treatment steps from retrieved profiles
Step 6 — Prevention: State one key prevention measure
Step 7 — Yield impact: Brief statement on expected yield effect if untreated

SAFETY & CONSTRAINTS:
- Maximum 200 tokens for core answer
- Never fabricate chemical names, dosages, or remedies not in the retrieved profiles
- Preserve trade names (e.g. Ridomil Gold, Nordox 75 WG, Imidacloprid) exactly as given
- Always recommend consulting a local agricultural extension officer for severe cases
- If the image does not appear to be a cocoa plant, state so immediately

USER QUESTION: {user_input}
"""

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

        # 3. Format final prompt
        if not user_input.strip():
            user_input = "Please diagnose this cocoa plant image and provide treatment guidance."

        formatted_prompt = SYSTEM_PROMPT_TEMPLATE.format(
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
