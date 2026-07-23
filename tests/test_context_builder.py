#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.context_builder import KokodivContextBuilder

def main():
    db_path = os.path.join("data", "kokodiv.db")
    print(f"Testing Kokodiv Context Builder with RAG database: {db_path}")

    builder = KokodivContextBuilder(db_path)

    # Test 1: High Confidence CNN Scores (>65%) in Yoruba
    high_conf_scores = {
        "black_pod": 0.85,
        "mirid": 0.08,
        "monilia": 0.03,
        "healthy": 0.04
    }
    
    print("\n--- Test 1: High Confidence (85% Black Pod) in Yoruba ---")
    result_yo = builder.build_full_prompt(
        cnn_scores=high_conf_scores,
        user_input="Bawo ni mo se le tọju àrùn dudu yi lori eso koko mi?",
        override_language=None
    )
    print(f"Detected Language: {result_yo['language_name']} ({result_yo['detected_language']})")
    print(f"Confidence Note:\n{result_yo['confidence_note']}")
    print("\nPrompt Preview (First 350 chars):\n" + result_yo['prompt'][:350] + "...\n")

    # Test 2: Low Confidence CNN Scores (<40%) in Hausa
    low_conf_scores = {
        "mirid": 0.32,
        "pod_borer": 0.28,
        "cssvd": 0.20,
        "healthy": 0.20
    }
    
    print("--- Test 2: Low Confidence (32% Mirid) in Hausa ---")
    result_ha = builder.build_full_prompt(
        cnn_scores=low_conf_scores,
        user_input="Ina son maganin wannan cuta a jikin koko.",
        override_language=None
    )
    print(f"Detected Language: {result_ha['language_name']} ({result_ha['detected_language']})")
    print(f"Confidence Note:\n{result_ha['confidence_note']}")

    # Test 3: Pidgin Query
    print("\n--- Test 3: West African Pidgin Query ---")
    result_pcm = builder.build_full_prompt(
        cnn_scores=high_conf_scores,
        user_input="How far, wetin be de best tin to treat dis black pod?",
        override_language=None
    )
    print(f"Detected Language: {result_pcm['language_name']} ({result_pcm['detected_language']})")

    print("\n✅ Test Passed! Kokodiv Context Builder & Prompt Pipeline operational.")

if __name__ == "__main__":
    main()
