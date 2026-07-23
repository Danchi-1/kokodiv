import os
import re
from typing import Optional, Dict

# Standard ISO code mappings for supported Tier 1 languages
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
    "pt": "Portuguese",
    "sw": "Swahili",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "pcm": "West African Pidgin"
}

# Offline keyword heuristics for fast backup
KEYWORD_HEURISTICS = {
    "yo": ["bawo", "koko", "ewe", "eso", "se", "kilode", "ti", "lori", "egbo", "ogun", "joo", "pelu", "fun", "won"],
    "ha": ["sannu", "koko", "ganye", "kwai", "cuta", "magani", "ina", "yaya", "ya", "tawada", "tare", "da", "daga"],
    "ig": ["kedu", "kooko", "akwukwo", "mkpuru", "oria", "ogwu", "nke", "na", "maka", "olee", "onye", "gboo"],
    "pcm": ["how", "far", "wey", "de", "dey", "dis", "koko", "leaf", "tin", "shey", "abeg", "wetin", "na", "so", "pikin"],
    "sw": ["jambo", "habari", "kakao", "jani", "ugonjwa", "dawa", "kwa", "katika", "na", "ya", "za", "gani"],
    "fr": ["bonjour", "cacao", "feuille", "maladie", "traitement", "comment", "sur", "avec", "pour", "dans"],
    "pt": ["ola", "cacau", "folha", "doenca", "tratamento", "como", "sobre", "com", "para", "em"],
    "ar": ["مرحبا", "كاكاو", "ورقة", "مرض", "علاج", "كيف", "على", "مع", "في"]
}

class FastLanguageDetector:
    """
    ML-Powered Language Identification Engine for Kokodiv.
    Supports FastText model classification with langdetect ML and keyword backup.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.ft_model = None
        self.langdetect_available = False
        
        # 1. Try loading fastText model if available
        if model_path and os.path.exists(model_path):
            try:
                import fasttext
                fasttext.FastText.eprint = lambda x: None
                self.ft_model = fasttext.load_model(model_path)
            except Exception:
                pass

        # 2. Try loading langdetect ML package
        try:
            import langdetect
            self.langdetect_available = True
        except ImportError:
            pass

    def detect_language(self, text: str) -> str:
        """
        Detects language of input user prompt string.
        Returns 2-letter ISO code: 'en', 'fr', 'ar', 'pt', 'sw', 'ha', 'yo', 'ig', or 'pcm'.
        Default fallback is 'en' (English).
        """
        if not text or not text.strip():
            return "en"

        cleaned_text = text.lower().strip()

        # 1. FastText ML Model Inference
        if self.ft_model:
            try:
                predictions, _ = self.ft_model.predict(cleaned_text.replace("\n", " "), k=1)
                lang_code = predictions[0].replace("__label__", "").split("_")[0]
                if lang_code in SUPPORTED_LANGUAGES:
                    return lang_code
            except Exception:
                pass

        # 2. Langdetect ML Inference
        if self.langdetect_available:
            try:
                import langdetect
                code = langdetect.detect(cleaned_text)
                if code in SUPPORTED_LANGUAGES:
                    return code
            except Exception:
                pass

        # 3. Arabic character set check
        if re.search(r'[\u0600-\u06FF]', cleaned_text):
            return "ar"

        # 4. Keyword Heuristic Fallback
        words = set(re.findall(r'\w+', cleaned_text))
        scores: Dict[str, int] = {lang: 0 for lang in KEYWORD_HEURISTICS}
        for lang, keywords in KEYWORD_HEURISTICS.items():
            for kw in keywords:
                if kw in words:
                    scores[lang] += 1

        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 0:
            return best_lang

        return "en"
