from typing import List, Dict

# Alphabetical class order corresponding exactly to PyTorch ImageFolder dataset index mapping
DISEASE_CLASSES: List[str] = [
    "anthracnose",
    "black_pod",
    "cssvd",
    "frosty_pod",
    "healthy",
    "mirid",
    "monilia",
    "pod_borer",
    "witches_broom"
]

DISEASE_DISPLAY_NAMES: Dict[str, str] = {
    "anthracnose": "Anthracnose",
    "black_pod": "Black Pod Disease",
    "cssvd": "Cocoa Swollen Shoot Virus (CSSVD)",
    "frosty_pod": "Frosty Pod Rot (Moniliophthora roreri)",
    "healthy": "Healthy Cocoa Plant",
    "mirid": "Mirid Bug (Capsids)",
    "monilia": "Monilia (Frosty Pod Rot)",
    "pod_borer": "Cocoa Pod Borer",
    "witches_broom": "Witches' Broom Disease"
}
