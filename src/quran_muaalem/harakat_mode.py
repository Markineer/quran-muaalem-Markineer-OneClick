"""
Harakat Mode Module for Quran Muaalem

This module provides functionality for the "Harakat-only" training mode,
which detects diacritic (harakat) mistakes and highlights wrong letters
in Quran text.

The module converts phoneme-level predictions to harakat-level feedback
displayed on standard Quran text (Uthmani script).
"""

from typing import Literal
import unicodedata
from dataclasses import dataclass


# =============================================================================
# CONSTANTS: Al-Fatiha Ayat (Fully Diacritized)
# =============================================================================

FATIHA_AYAT = [
    "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
    "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",
    "ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
    "مَـٰلِكِ يَوْمِ ٱلدِّينِ",
    "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
    "ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ",
    "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ",
]

# Full Fatiha text concatenated (for single-take recitation)
FATIHA_FULL_TEXT = " ۝ ".join(FATIHA_AYAT)


# =============================================================================
# CONSTANTS: Arabic Diacritics (Harakat)
# =============================================================================

# Arabic combining marks (diacritics/harakat)
ARABIC_DIACRITICS = {
    "\u064B",  # Tanween Fath (ً)
    "\u064C",  # Tanween Damm (ٌ)
    "\u064D",  # Tanween Kasr (ٍ)
    "\u064E",  # Fatha (َ)
    "\u064F",  # Damma (ُ)
    "\u0650",  # Kasra (ِ)
    "\u0651",  # Shadda (ّ)
    "\u0652",  # Sukoon (ْ)
    "\u0653",  # Maddah above (ٓ)
    "\u0654",  # Hamza above (ٔ)
    "\u0655",  # Hamza below (ٕ)
    "\u0656",  # Subscript alef (ٖ)
    "\u0657",  # Inverted damma (ٗ)
    "\u0658",  # Mark noon ghunna (٘)
    "\u0659",  # Zwarakay (ٙ)
    "\u065A",  # Vowel sign small v above (ٚ)
    "\u065B",  # Vowel sign inverted small v above (ٛ)
    "\u065C",  # Vowel sign dot below (ٜ)
    "\u065D",  # Reversed damma (ٝ)
    "\u065E",  # Fatha with two dots (ٞ)
    "\u065F",  # Wavy hamza below (ٟ)
    "\u0670",  # Dagger alif / superscript alef (ٰ)
    "\u06D6",  # Small high ligature sad with lam with alef maksura (ۖ)
    "\u06D7",  # Small high ligature qaf with lam with alef maksura (ۗ)
    "\u06D8",  # Small high meem initial form (ۘ)
    "\u06D9",  # Small high lam alef (ۙ)
    "\u06DA",  # Small high jeem (ۚ)
    "\u06DB",  # Small high three dots (ۛ)
    "\u06DC",  # Small high seen (ۜ)
    "\u06DF",  # Small high rounded zero (۟)
    "\u06E0",  # Small high upright rectangular zero (۠)
    "\u06E1",  # Small high dotless head of khah (ۡ)
    "\u06E2",  # Small high meem isolated form (ۢ)
    "\u06E3",  # Small low seen (ۣ)
    "\u06E4",  # Small high madda (ۤ)
    "\u06E7",  # Small high yeh (ۧ)
    "\u06E8",  # Small high noon (ۨ)
    "\u06EA",  # Empty centre low stop (۪)
    "\u06EB",  # Empty centre high stop (۫)
    "\u06EC",  # Rounded high stop with filled centre (۬)
    "\u06ED",  # Small low meem (ۭ)
}

# Phoneme vowel characters (from quran_transcript phoneme output)
PHONEME_VOWELS = {
    "\u064E",  # Fatha (َ)
    "\u064F",  # Damma (ُ)
    "\u0650",  # Kasra (ِ)
    "\u0652",  # Sukoon (ْ)
    "\u0651",  # Shadda (ّ)
    "ۦ",       # Yaa madd
    "ۥ",       # Waw madd
    "ا",       # Alif (as madd)
    "۪",       # Fatha momala
    "ـ",       # Alif momala
    "ٲ",       # Hamza mosahala
    "ڇ",       # Qalqala marker
    "ں",       # Noon mokhfah
    "۾",       # Meem mokhfah
    "ۜ",       # Sakt
    "ؙ",       # Damma mokhtalasa
}


# =============================================================================
# CONSTANTS: Harakat Class Labels
# =============================================================================

class HarakatClass:
    """Harakat classification labels."""
    FATHA = "FATHA"
    KASRA = "KASRA"
    DAMMA = "DAMMA"
    SUKOON = "SUKOON"
    SHADDA_FATHA = "SHADDA_FATHA"
    SHADDA_KASRA = "SHADDA_KASRA"
    SHADDA_DAMMA = "SHADDA_DAMMA"
    SHADDA_SUKOON = "SHADDA_SUKOON"
    TANWEEN_FATH = "TANWEEN_FATH"
    TANWEEN_KASR = "TANWEEN_KASR"
    TANWEEN_DAMM = "TANWEEN_DAMM"
    MADD = "MADD"
    NONE = "NONE"


# Arabic display names for harakat classes
HARAKAT_CLASS_ARABIC = {
    HarakatClass.FATHA: "فتحة",
    HarakatClass.KASRA: "كسرة",
    HarakatClass.DAMMA: "ضمة",
    HarakatClass.SUKOON: "سكون",
    HarakatClass.SHADDA_FATHA: "شدة وفتحة",
    HarakatClass.SHADDA_KASRA: "شدة وكسرة",
    HarakatClass.SHADDA_DAMMA: "شدة وضمة",
    HarakatClass.SHADDA_SUKOON: "شدة وسكون",
    HarakatClass.TANWEEN_FATH: "تنوين فتح",
    HarakatClass.TANWEEN_KASR: "تنوين كسر",
    HarakatClass.TANWEEN_DAMM: "تنوين ضم",
    HarakatClass.MADD: "مد",
    HarakatClass.NONE: "بدون حركة",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LetterSlot:
    """
    Represents a single letter slot in parsed Quran text.

    Attributes:
        base: The base Arabic letter character
        marks: String of diacritics following the letter
        start: Start index in original string
        end: End index (exclusive) in original string
        harakat_class: Computed harakat classification
        is_letter: True if this is a letter, False for spaces/separators
        original_text: The original text segment (base + marks)
    """
    base: str
    marks: str
    start: int
    end: int
    harakat_class: str
    is_letter: bool
    original_text: str


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def is_arabic_letter(char: str) -> bool:
    """
    Check if a character is an Arabic letter (base letter, not diacritic).

    Args:
        char: Single character to check

    Returns:
        True if the character is an Arabic letter
    """
    if not char:
        return False

    # Check if it's a diacritic
    if char in ARABIC_DIACRITICS:
        return False

    # Check Unicode category - Arabic letters are category 'Lo' (Letter, other)
    # Also include 'Lm' (Letter, modifier) for some special cases
    cat = unicodedata.category(char)

    # Check if it's in Arabic Unicode blocks
    code_point = ord(char)

    # Arabic block: U+0600 to U+06FF
    # Arabic Supplement: U+0750 to U+077F
    # Arabic Extended-A: U+08A0 to U+08FF
    # Arabic Presentation Forms-A: U+FB50 to U+FDFF
    # Arabic Presentation Forms-B: U+FE70 to U+FEFF

    is_arabic_range = (
        (0x0600 <= code_point <= 0x06FF) or
        (0x0750 <= code_point <= 0x077F) or
        (0x08A0 <= code_point <= 0x08FF) or
        (0xFB50 <= code_point <= 0xFDFF) or
        (0xFE70 <= code_point <= 0xFEFF)
    )

    # Must be a letter category AND in Arabic range AND not a diacritic
    return cat in ('Lo', 'Lm') and is_arabic_range


def marks_to_class(marks: str) -> str:
    """
    Classify a string of diacritics into a harakat class.

    The classification follows priority order:
    1. Shadda combinations (most specific)
    2. Tanween
    3. Basic vowels
    4. Sukoon
    5. None (no explicit vowel)

    Args:
        marks: String of diacritic characters

    Returns:
        A HarakatClass constant string
    """
    if not marks:
        return HarakatClass.NONE

    has_shadda = "\u0651" in marks  # ّ
    has_fatha = "\u064E" in marks   # َ
    has_kasra = "\u0650" in marks   # ِ
    has_damma = "\u064F" in marks   # ُ
    has_sukoon = "\u0652" in marks  # ْ

    has_tanween_fath = "\u064B" in marks  # ً
    has_tanween_kasr = "\u064D" in marks  # ٍ
    has_tanween_damm = "\u064C" in marks  # ٌ

    has_dagger_alif = "\u0670" in marks   # ٰ
    has_maddah = "\u0653" in marks        # ٓ

    # Check for Shadda combinations first
    if has_shadda:
        if has_fatha:
            return HarakatClass.SHADDA_FATHA
        elif has_kasra:
            return HarakatClass.SHADDA_KASRA
        elif has_damma:
            return HarakatClass.SHADDA_DAMMA
        elif has_sukoon:
            return HarakatClass.SHADDA_SUKOON
        else:
            # Shadda alone - treat as shadda + sukoon
            return HarakatClass.SHADDA_SUKOON

    # Check for Tanween
    if has_tanween_fath:
        return HarakatClass.TANWEEN_FATH
    if has_tanween_kasr:
        return HarakatClass.TANWEEN_KASR
    if has_tanween_damm:
        return HarakatClass.TANWEEN_DAMM

    # Check for basic vowels
    if has_sukoon:
        return HarakatClass.SUKOON
    if has_fatha:
        return HarakatClass.FATHA
    if has_kasra:
        return HarakatClass.KASRA
    if has_damma:
        return HarakatClass.DAMMA

    # Check for madd indicators
    if has_dagger_alif or has_maddah:
        return HarakatClass.MADD

    return HarakatClass.NONE


def parse_ayah_to_slots(text: str) -> list[LetterSlot]:
    """
    Parse Quran ayah text into letter slots.

    Each slot represents either:
    - A letter with its associated diacritics
    - A space or separator (non-letter segment)

    Args:
        text: The Quran text to parse (Uthmani script with tashkil)

    Returns:
        List of LetterSlot objects preserving the exact text structure
    """
    slots = []
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        # Check if this is a space or separator
        if char.isspace() or char in '۝۞':
            # Non-letter segment
            slots.append(LetterSlot(
                base=char,
                marks="",
                start=i,
                end=i + 1,
                harakat_class=HarakatClass.NONE,
                is_letter=False,
                original_text=char,
            ))
            i += 1
            continue

        # Check if this is an Arabic letter
        if is_arabic_letter(char):
            start = i
            base = char
            i += 1

            # Collect following diacritics
            marks = ""
            while i < n and text[i] in ARABIC_DIACRITICS:
                marks += text[i]
                i += 1

            end = i
            original = text[start:end]
            harakat_class = marks_to_class(marks)

            slots.append(LetterSlot(
                base=base,
                marks=marks,
                start=start,
                end=end,
                harakat_class=harakat_class,
                is_letter=True,
                original_text=original,
            ))
        else:
            # Other character (punctuation, special marks, etc.)
            slots.append(LetterSlot(
                base=char,
                marks="",
                start=i,
                end=i + 1,
                harakat_class=HarakatClass.NONE,
                is_letter=False,
                original_text=char,
            ))
            i += 1

    return slots


def extract_harakat_stream(phoneme_text: str) -> list[str]:
    """
    Extract a simplified harakat stream from phoneme text.

    This function converts the phoneme representation (from quran_transcript)
    into a list of harakat class labels, one per phoneme unit.

    The phoneme text contains Arabic letters mixed with diacritics. We extract
    the harakat class for each letter-based unit.

    Args:
        phoneme_text: The phoneme string from model output or reference

    Returns:
        List of harakat class labels (one per letter)
    """
    if not phoneme_text:
        return []

    harakat_stream = []
    i = 0
    n = len(phoneme_text)

    while i < n:
        char = phoneme_text[i]

        # Skip spaces
        if char.isspace():
            i += 1
            continue

        # Check if this is a base letter (consonant)
        # In phoneme representation, consonants are followed by their vowels
        if is_arabic_letter(char) or char in PHONEME_VOWELS:
            # If it's a vowel/madd character on its own, it indicates a vowel
            if char in PHONEME_VOWELS and char not in {"\u064E", "\u064F", "\u0650", "\u0652", "\u0651"}:
                # Madd characters - these extend previous vowel, skip
                i += 1
                continue

            # It's a consonant - collect following vowel marks
            start = i
            i += 1

            # Collect following diacritics/vowels
            marks = ""
            while i < n:
                next_char = phoneme_text[i]
                if next_char in {"\u064E", "\u064F", "\u0650", "\u0652", "\u0651",
                                "\u064B", "\u064C", "\u064D", "\u0670", "\u0653"}:
                    marks += next_char
                    i += 1
                elif next_char in PHONEME_VOWELS:
                    # Special phoneme vowels - treat as madd
                    marks += next_char
                    i += 1
                else:
                    break

            # Classify the marks
            harakat_class = marks_to_class(marks)

            # Handle special madd characters in marks
            if any(c in marks for c in ["ۦ", "ۥ", "ا"]):
                harakat_class = HarakatClass.MADD

            harakat_stream.append(harakat_class)
        else:
            # Skip other characters
            i += 1

    return harakat_stream


def compare_harakat_to_slots(
    slots: list[LetterSlot],
    pred_harakat_stream: list[str],
) -> tuple[set[int], dict[int, str]]:
    """
    Compare predicted harakat stream to Quran letter slots.

    This function finds which letter slots have incorrect harakat
    by comparing the predicted harakat stream to the expected
    harakat from the Quran slots.

    Args:
        slots: List of LetterSlot objects from parsed Quran text
        pred_harakat_stream: List of predicted harakat class labels

    Returns:
        Tuple of:
        - wrong_slot_idxs: Set of slot indices with errors
        - hint_map: Dict mapping slot index to hint text (expected vs heard)
    """
    # Extract expected harakat from letter slots only
    letter_slots = [(i, s) for i, s in enumerate(slots) if s.is_letter]
    expected = [s.harakat_class for _, s in letter_slots]
    predicted = pred_harakat_stream

    wrong_slot_idxs = set()
    hint_map = {}

    # Compare at minimum length
    n = min(len(expected), len(predicted))

    for i in range(n):
        slot_idx = letter_slots[i][0]  # Get actual slot index
        exp_class = expected[i]
        pred_class = predicted[i]

        if exp_class != pred_class:
            wrong_slot_idxs.add(slot_idx)
            exp_arabic = HARAKAT_CLASS_ARABIC.get(exp_class, exp_class)
            pred_arabic = HARAKAT_CLASS_ARABIC.get(pred_class, pred_class)
            hint_map[slot_idx] = f"المتوقع: {exp_arabic} | المقروء: {pred_arabic}"

    # Handle length mismatch - mark extra expected slots as wrong
    if len(expected) > len(predicted):
        for i in range(n, len(expected)):
            slot_idx = letter_slots[i][0]
            wrong_slot_idxs.add(slot_idx)
            exp_class = expected[i]
            exp_arabic = HARAKAT_CLASS_ARABIC.get(exp_class, exp_class)
            hint_map[slot_idx] = f"المتوقع: {exp_arabic} | المقروء: (غير موجود)"

    # If predicted is longer, we have extra phonemes - less common
    # We can optionally flag this, but for now we focus on missing/wrong

    return wrong_slot_idxs, hint_map


def render_ayah_with_highlights(
    text: str,
    slots: list[LetterSlot],
    wrong_slot_idxs: set[int],
    hint_map: dict[int, str] | None = None,
    uncertain_slot_idxs: set[int] | None = None,
) -> str:
    """
    Render Quran ayah text as HTML with wrong letters highlighted.

    Args:
        text: Original Quran text
        slots: Parsed letter slots
        wrong_slot_idxs: Set of slot indices that have errors
        hint_map: Optional dict mapping slot index to tooltip text
        uncertain_slot_idxs: Optional set of low-confidence slots (yellow highlight)

    Returns:
        HTML string with highlighted letters
    """
    if hint_map is None:
        hint_map = {}
    if uncertain_slot_idxs is None:
        uncertain_slot_idxs = set()

    html_parts = []

    for idx, slot in enumerate(slots):
        original = slot.original_text

        if idx in wrong_slot_idxs:
            # Error - red highlight
            tooltip = hint_map.get(idx, "خطأ في الحركة")
            escaped_original = original.replace('"', '&quot;')
            html_parts.append(
                f'<span class="harakat-wrong" title="{tooltip}">{escaped_original}</span>'
            )
        else:
            # Correct - no highlight
            html_parts.append(original)

    return "".join(html_parts)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_fatiha_reference_text(include_basmala: bool = True) -> str:
    """
    Get the full Al-Fatiha text for display.

    Args:
        include_basmala: Whether to include Basmala (verse 1)

    Returns:
        Full Al-Fatiha text with verse separators
    """
    if include_basmala:
        return " ۝ ".join(FATIHA_AYAT)
    else:
        return " ۝ ".join(FATIHA_AYAT[1:])


def get_fatiha_ayat_list(include_basmala: bool = True) -> list[str]:
    """
    Get Al-Fatiha ayat as a list.

    Args:
        include_basmala: Whether to include Basmala (verse 1)

    Returns:
        List of ayat strings
    """
    if include_basmala:
        return FATIHA_AYAT.copy()
    else:
        return FATIHA_AYAT[1:].copy()


def count_letter_slots(text: str) -> int:
    """
    Count the number of letter slots in text.

    Args:
        text: Quran text to analyze

    Returns:
        Number of Arabic letters (not including diacritics)
    """
    slots = parse_ayah_to_slots(text)
    return sum(1 for s in slots if s.is_letter)


# =============================================================================
# DEBUG/DIAGNOSTIC FUNCTIONS
# =============================================================================

def debug_slots(text: str) -> None:
    """
    Print debug information about parsed slots.

    Args:
        text: Text to parse and debug
    """
    slots = parse_ayah_to_slots(text)
    print(f"Total slots: {len(slots)}")
    print(f"Letter slots: {sum(1 for s in slots if s.is_letter)}")
    print("-" * 60)
    for i, slot in enumerate(slots):
        if slot.is_letter:
            print(f"[{i}] Letter: '{slot.base}' + marks: '{slot.marks}' "
                  f"= class: {slot.harakat_class}")


def debug_harakat_stream(phoneme_text: str) -> None:
    """
    Print debug information about extracted harakat stream.

    Args:
        phoneme_text: Phoneme text to analyze
    """
    stream = extract_harakat_stream(phoneme_text)
    print(f"Harakat stream length: {len(stream)}")
    print("-" * 40)
    for i, h in enumerate(stream):
        arabic = HARAKAT_CLASS_ARABIC.get(h, h)
        print(f"[{i}] {h} ({arabic})")


# =============================================================================
# REAL-TIME HELPER FUNCTIONS
# =============================================================================

def get_expected_harakat_sequence_for_ayah(ayah_text: str) -> list[str]:
    """
    Get the expected harakat sequence for an ayah.

    This function parses the ayah text and extracts the harakat class
    for each letter slot. This sequence is used for comparing against
    predicted harakat in real-time mode.

    Args:
        ayah_text: The ayah text (Uthmani script with tashkil)

    Returns:
        List of harakat class labels (one per letter)
    """
    slots = parse_ayah_to_slots(ayah_text)
    return [s.harakat_class for s in slots if s.is_letter]


def get_expected_sequences_for_all_ayat(ayat: list[str] = None) -> list[list[str]]:
    """
    Get expected harakat sequences for all ayat in a list.

    This is used by the real-time module for ayah detection via
    similarity scoring against the expected sequences.

    Args:
        ayat: List of ayah texts (defaults to FATIHA_AYAT)

    Returns:
        List of harakat sequences (one list per ayah)
    """
    if ayat is None:
        ayat = FATIHA_AYAT

    return [get_expected_harakat_sequence_for_ayah(a) for a in ayat]


def compare_harakat_sequences(
    predicted: list[str],
    expected: list[str],
) -> tuple[set[int], dict[int, str]]:
    """
    Compare two harakat sequences and identify differences.

    This is a simplified version of compare_harakat_to_slots that works
    directly with harakat sequences (not slots). Used for real-time mode
    where we need quick comparisons.

    Args:
        predicted: Predicted harakat class labels
        expected: Expected harakat class labels

    Returns:
        Tuple of:
        - wrong_idxs: Set of indices with errors (letter-slot indices)
        - hint_map: Dict mapping index to hint text
    """
    wrong_idxs = set()
    hint_map = {}

    n = min(len(predicted), len(expected))

    for i in range(n):
        if predicted[i] != expected[i]:
            wrong_idxs.add(i)
            exp_arabic = HARAKAT_CLASS_ARABIC.get(expected[i], expected[i])
            pred_arabic = HARAKAT_CLASS_ARABIC.get(predicted[i], predicted[i])
            hint_map[i] = f"المتوقع: {exp_arabic} | المقروء: {pred_arabic}"

    # Mark missing predictions
    if len(expected) > len(predicted):
        for i in range(n, len(expected)):
            wrong_idxs.add(i)
            exp_arabic = HARAKAT_CLASS_ARABIC.get(expected[i], expected[i])
            hint_map[i] = f"المتوقع: {exp_arabic} | المقروء: (غير موجود)"

    return wrong_idxs, hint_map
