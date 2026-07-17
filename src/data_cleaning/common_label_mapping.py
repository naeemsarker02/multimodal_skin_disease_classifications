"""Shared canonical disease taxonomy for ISIC Archive 1 & 2.

Both archives are used for the same purpose (external validation, per
README section 6) and their class vocabularies overlap heavily with
each other and with PAD-UFES-20/HAM10000's disease_label values
(Basal Cell Carcinoma, Squamous Cell Carcinoma, Actinic Keratosis,
Nevus, Seborrheic Keratosis, Melanoma, Dermatofibroma, Vascular
Lesion), so a single shared mapping guarantees identical canonical
strings are used everywhere a disease is shared across sources,
instead of two independently-maintained mappings drifting apart.

Keys are the *lowercased* source label (ISIC Archive 1's folder names
are already lowercase; ISIC Archive 2's diagnosis_3 values are
title-cased and lowercased here before lookup).
"""

ISIC_LABEL_MAPPING = {
    "actinic keratosis": ("Actinic Keratosis", "Same disease as PAD-UFES-20's ACK code; standardized name shared across sources."),
    "solar or actinic keratosis": ("Actinic Keratosis", "ISIC Archive 2's diagnosis_3 wording for the same disease as Archive 1's 'actinic keratosis' and PAD-UFES-20's ACK."),
    "basal cell carcinoma": ("Basal Cell Carcinoma", "Same disease as PAD-UFES-20/HAM10000's BCC code; standardized name shared across sources."),
    "squamous cell carcinoma": ("Squamous Cell Carcinoma", "Same disease as PAD-UFES-20's SCC code; standardized name shared across sources."),
    "squamous cell carcinoma, nos": ("Squamous Cell Carcinoma", "ISIC Archive 2's diagnosis_3 wording ('NOS' = not otherwise specified) for the same disease as PAD-UFES-20's SCC."),
    "nevus": ("Nevus", "Same disease as PAD-UFES-20's NEV / HAM10000's nv code; standardized name shared across sources."),
    "epidermal nevus": ("Nevus", "Distinct histological variant of nevus; grouped with Nevus given only 1 occurrence in the source data."),
    "atypical melanocytic neoplasm": ("Nevus", "Grouped with Nevus (atypical/dysplastic nevus spectrum) given only 1 occurrence in the source data."),
    "seborrheic keratosis": ("Seborrheic Keratosis", "Same disease as PAD-UFES-20's SEK code; standardized name shared across sources."),
    "melanoma": ("Melanoma", "Same disease as PAD-UFES-20/HAM10000's MEL code; standardized name shared across sources."),
    "melanoma, nos": ("Melanoma", "ISIC Archive 2's diagnosis_3 wording for Melanoma; subtype detail (NOS) dropped for cross-source consistency."),
    "melanoma in situ": ("Melanoma", "Melanoma subtype (non-invasive); collapsed into Melanoma for cross-source consistency, since PAD-UFES-20/HAM10000 do not distinguish invasion stage."),
    "melanoma invasive": ("Melanoma", "Melanoma subtype (invasive); collapsed into Melanoma for cross-source consistency, since PAD-UFES-20/HAM10000 do not distinguish invasion stage."),
    "dermatofibroma": ("Dermatofibroma", "Same disease as HAM10000's df code; standardized name shared across sources."),
    "vascular lesion": ("Vascular Lesion", "Same disease as HAM10000's vasc code; standardized name shared across sources."),
    "pigmented benign keratosis": ("Pigmented Benign Keratosis", "Distinct from Seborrheic Keratosis in both ISIC archives' own taxonomy (they list both categories separately); kept distinct rather than merged."),
    "solar lentigo": ("Solar Lentigo", "ISIC Archive 2-specific class, not present in Archive 1, PAD-UFES-20, or HAM10000."),
}
