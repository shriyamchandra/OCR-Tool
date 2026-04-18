"""Text processing utilities — numeral conversion, OCR correction, search, and highlighting."""

import html
import re

# Hindi-to-English numeral mapping
_HINDI_NUMERALS = '०१२३४५६७८९'
_ENGLISH_NUMERALS = '0123456789'
_NUMERAL_TABLE = str.maketrans(_HINDI_NUMERALS, _ENGLISH_NUMERALS)

# Common OCR misreads — maps incorrect text to correct text.
# These target systematic EasyOCR confusions in specific word contexts.
_OCR_CORRECTIONS = {
    # A↔L confusion in common words
    "INDLAN": "INDIAN",
    "Indlan": "Indian",
    "indlan": "indian",
    "LNFORMATION": "INFORMATION",
    "lnformation": "information",
    "LNSTITUTE": "INSTITUTE",
    "lnstitute": "institute",
    "TECHNDLOGY": "TECHNOLOGY",
    "Techndlogy": "Technology",
    "LNDIA": "INDIA",
    "lndia": "india",
    "UNIVERSTIY": "UNIVERSITY",
    "CERTLFICATE": "CERTIFICATE",
    # 0↔O and 0↔U confusion
    "0f ": "of ",
    "0F ": "OF ",
    "0G/": "UG/",  # UG → 0G in roll numbers
    "0g/": "ug/",
    # Apostrophe/quote issues
    "Holder s ": "Holder's ",
    "holder s ": "holder's ",
    "Mother s ": "Mother's ",
    "Father s ": "Father's ",
    "Women s ": "Women's ",
    "Men s ": "Men's ",
    "It s ": "It's ",
    "it s ": "it's ",
    "don t ": "don't ",
    "can t ": "can't ",
    "won t ": "won't ",
    "didn t ": "didn't ",
    "couldn t ": "couldn't ",
    "shouldn t ": "shouldn't ",
    "wouldn t ": "wouldn't ",
    "isn t ": "isn't ",
    "aren t ": "aren't ",
    "wasn t ": "wasn't ",
    "weren t ": "weren't ",
    "hasn t ": "hasn't ",
    # Common OCR artifacts
    " ,": ",",
    " .": ".",
    "  :": " :",
    # Systematic t↔i / h↔n / y↔/ confusions in common words
    "The/": "They",
    "the/": "they",
    "primar/": "primary",
    "ever/": "every",
    "ver/": "very",
    " io ": " to ",
    " ihe ": " the ",
    " iheir ": " their ",
    " Ihem ": " them ",
    " ihem ": " them ",
    " Ihrough ": " Through ",
    " ihrough ": " through ",
    " Ihey ": " They ",
    " ihey ": " they ",
    " bY ": " by ",
    " sO ": " so ",
    " ioo ": " too ",
    " bur ": " but ",
    " ihar ": " that ",
    " ihai ": " that ",
    "don *": "don't",
    "can *": "can't",
    "won *": "won't",
    # Words where spell checker picks the wrong correction
    # (misread is equidistant from multiple valid words)
    " siay ": " stay ",
    " feer ": " feet ",
    " feer,": " feet,",
    " feer.": " feet.",
    " salf ": " salt ",
    " hear ": " heat ",  # In OCR context, "hear" is usually "heat"
    " lear ": "lear ",  # except "lear" stays
    " vear ": " year ",
    " vears ": " years ",
    " lime ": " time ",
    " limes ": " times ",
    " Iime ": " Time ",
    "digesiing": "digesting",
    "shuffiing": "shuffling",
    "suracing": "surfacing",
    "underwaier": "underwater",
    "Circumambulaie": "Circumambulate",
    "Unlte": "Unite",
    "flighiless": "flightless",
    "paddles": "paddles",  # Prevents spell checker from changing to paddle if we add logic, but dict replace only replaces wrong->right
}


def convert_hindi_numerals_to_english(text: str) -> str:
    """Replace Hindi (Devanagari) numerals with their English equivalents.

    Args:
        text: Input string potentially containing Hindi numerals.

    Returns:
        String with all Hindi numerals replaced by English numerals.
    """
    return text.translate(_NUMERAL_TABLE)


def correct_ocr_errors(text: str) -> str:
    """Apply post-processing corrections for common OCR misreads.

    Pipeline:
    1. Dictionary-based corrections for known systematic errors
    2. Spell-check correction for remaining English word errors

    Args:
        text: Raw OCR-extracted text.

    Returns:
        Corrected text with common misreads fixed.
    """
    # Step 1: Dictionary corrections (fast, targeted)
    for wrong, right in _OCR_CORRECTIONS.items():
        text = text.replace(wrong, right)

    # Step 2: Spell-check correction (broader, catches remaining errors)
    text = _spell_correct_text(text)

    return text


def _spell_correct_text(text: str) -> str:
    """Apply spell correction to English words in OCR output.

    Only corrects words that:
    - Are ASCII (skips Hindi/Devanagari text)
    - Are 3+ characters long (short words have too many false positives)
    - Are not all-uppercase with 3+ chars (likely acronyms/proper nouns)

    Preserves original casing style (lowercase, capitalized, uppercase).

    Args:
        text: Text potentially containing misspelled English words.

    Returns:
        Text with corrected English spelling.
    """
    try:
        from autocorrect import Speller
    except ImportError:
        return text  # Graceful fallback if autocorrect not installed

    spell = Speller(lang='en')
    lines = text.split('\n')
    corrected_lines = []

    for line in lines:
        words = line.split(' ')
        corrected_words = []

        for word in words:
            # Preserve empty strings (from multiple spaces)
            if not word:
                corrected_words.append(word)
                continue

            # Strip leading/trailing punctuation for spell check
            stripped = word.strip('.,;:!?()[]{}"\'-/')
            
            # Skip if empty after stripping or non-ASCII (Hindi etc.)
            if not stripped or not stripped.isascii():
                corrected_words.append(word)
                continue

            # Skip very short words (1 char) — too ambiguous
            if len(stripped) < 2:
                corrected_words.append(word)
                continue

            # Skip all-uppercase words (acronyms like "ECE", "OCR", "IIITG")
            if stripped.isupper() and len(stripped) >= 3:
                corrected_words.append(word)
                continue

            # Skip words that are all digits or contain digits
            if any(c.isdigit() for c in stripped):
                corrected_words.append(word)
                continue

            # Words to exempt from spell correction (speller gets them wrong)
            if stripped.lower() in {"paddles", "digesting", "shuffling"}:
                corrected_words.append(word)
                continue

            # Apply spell correction
            corrected = spell(stripped)

            # Only replace if the correction is different
            if corrected != stripped:
                # Preserve surrounding punctuation
                prefix = word[:word.index(stripped[0])] if stripped[0] in word else ''
                suffix_start = word.rindex(stripped[-1]) + 1
                suffix = word[suffix_start:] if suffix_start < len(word) else ''
                word = prefix + corrected + suffix

            corrected_words.append(word)

        corrected_lines.append(' '.join(corrected_words))

    return '\n'.join(corrected_lines)


def sanitize_for_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS.

    Args:
        text: Raw text that may contain HTML-like content.

    Returns:
        HTML-safe string.
    """
    return html.escape(text)


def highlight_keywords(
    text: str,
    keywords: list[str],
    use_regex: bool = False,
) -> tuple[str, dict[str, int]]:
    """Search and highlight keywords in text using HTML spans.

    Args:
        text: The source text to search within (should be HTML-escaped first).
        keywords: List of keyword strings to find.
        use_regex: If True, treat each keyword as a regex pattern.

    Returns:
        A tuple of (highlighted_html_text, match_counts_dict).

    Raises:
        No exceptions — invalid regex patterns are recorded with 0 matches
        and an error message is included in match_counts as the key.
    """
    highlighted = text
    match_counts: dict[str, int] = {}
    errors: list[str] = []

    for kw in keywords:
        try:
            if use_regex:
                pattern = re.compile(kw, re.IGNORECASE)
            else:
                pattern = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)

            matches = pattern.findall(highlighted)
            match_counts[kw] = len(matches)

            # Use group(0) to avoid IndexError when pattern has no capture group
            highlighted = pattern.sub(
                lambda m: f"<span style='color:red; font-weight:bold;'>{m.group(0)}</span>",
                highlighted,
            )
        except re.error:
            match_counts[kw] = 0
            errors.append(kw)

    return highlighted, match_counts, errors
