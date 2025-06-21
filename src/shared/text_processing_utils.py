import pyarabic.araby as araby
import re

def normalize_arabic_text_for_embedding(text: str) -> str:
    """
    Normalizes Arabic text for consistent embedding.
    Steps:
    1. Strip Tashkeel (diacritics).
    2. Normalize Alef forms (أ, إ, آ, ءا, أ) to plain Alef (ا).
    3. Normalize Teh Marbuta (ة) to Heh (ه).
    4. Remove Tatweel (kashida).
    5. Remove common punctuation that might interfere with semantic meaning for embeddings.
       (This step is basic; more sophisticated punctuation handling might be needed
        depending on the corpus and model sensitivity).
    6. Normalize whitespace (multiple spaces to one, strip leading/trailing).
    """
    if not text:
        return ""

    # 1. Strip Tashkeel
    text = araby.strip_tashkeel(text)

    # 2. Normalize Alef forms
    text = araby.normalize_alef(text) # Handles most Alef forms

    # Additional Alef normalizations if needed (pyarabic.araby.normalize_alef is usually good)
    # text = text.replace(araby.ALEF_HAMZA_ABOVE, araby.ALEF)
    # text = text.replace(araby.ALEF_HAMZA_BELOW, araby.ALEF)
    # text = text.replace(araby.ALEF_MADDA, araby.ALEF)

    # 3. Normalize Teh Marbuta (ة) to Heh (ه)
    # Direct replacement is often robust for this specific normalization.
    text = text.replace(araby.TEH_MARBUTA, araby.HEH)

    # 4. Remove Tatweel (kashida)
    text = araby.strip_tatweel(text)

    # 5. Basic punctuation removal (customize as needed)
    # This removes some common punctuation. You might want to be more selective
    # or handle specific punctuation (like hyphens in compound words) differently.
    # For embeddings, often less punctuation is better unless it's semantically crucial.
    text = re.sub(r"""[«»!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~،؛؟]""", " ", text)

    # 6. Normalize whitespace
    text = " ".join(text.split())

    return text.strip()

if __name__ == '__main__':
    # Test cases
    sample_texts = [
        "هَذَا نَصٌّ تَجْرِيْبِيٌّ لِلتَّطْبِيقِ.",
        "الإستشارة القانونية المٌقدمة كانت مٌفيدة.",
        "المادة رقم خمسة (٥) من قانون العمل.",
        "القانونُ    اليمنيّ... رائعٌ!",
        "مدرسة",
        "مدرسة الإدارة",
        "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"
    ]

    print("Original -> Normalized:")
    for t in sample_texts:
        normalized = normalize_arabic_text_for_embedding(t)
        print(f"'{t}' -> '{normalized}'")

    # Example showing Teh Marbuta normalization
    # text.replace(araby.TEH_MARBUTA, araby.HEH) should convert ة to ه
    print(normalize_arabic_text_for_embedding("مدرسة جميلة")) # Expected: مدرسه جميله
    print(normalize_arabic_text_for_embedding("السيارة الجديدة")) # Expected: السياره الجديده
