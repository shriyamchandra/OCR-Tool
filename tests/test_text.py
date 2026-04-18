"""Unit tests for text utility functions."""

import pytest
from utils.text import (
    convert_hindi_numerals_to_english,
    sanitize_for_html,
    highlight_keywords,
)


class TestConvertHindiNumerals:
    def test_basic_conversion(self):
        assert convert_hindi_numerals_to_english("०१२३४५६७८९") == "0123456789"

    def test_mixed_text(self):
        assert convert_hindi_numerals_to_english("दिल्ली ११००१") == "दिल्ली 11001"

    def test_no_hindi_numerals(self):
        assert convert_hindi_numerals_to_english("Hello 123") == "Hello 123"

    def test_empty_string(self):
        assert convert_hindi_numerals_to_english("") == ""


class TestSanitizeForHtml:
    def test_escapes_script_tags(self):
        result = sanitize_for_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escapes_ampersand(self):
        assert sanitize_for_html("A & B") == "A &amp; B"

    def test_plain_text_unchanged(self):
        assert sanitize_for_html("Hello World") == "Hello World"


class TestHighlightKeywords:
    def test_single_keyword(self):
        text = "The quick brown fox"
        highlighted, counts, errors = highlight_keywords(text, ["quick"])
        assert counts["quick"] == 1
        assert "font-weight:bold" in highlighted
        assert errors == []

    def test_multiple_keywords(self):
        text = "The quick brown fox jumps over the lazy dog"
        highlighted, counts, errors = highlight_keywords(text, ["quick", "lazy"])
        assert counts["quick"] == 1
        assert counts["lazy"] == 1

    def test_case_insensitive(self):
        text = "Hello HELLO hello"
        _, counts, _ = highlight_keywords(text, ["hello"])
        assert counts["hello"] == 3

    def test_no_matches(self):
        text = "The quick brown fox"
        _, counts, errors = highlight_keywords(text, ["zebra"])
        assert counts["zebra"] == 0
        assert errors == []

    def test_regex_mode(self):
        text = "cat bat hat mat"
        _, counts, _ = highlight_keywords(text, [r"[cbhm]at"], use_regex=True)
        assert counts[r"[cbhm]at"] == 4

    def test_invalid_regex(self):
        text = "some text"
        _, counts, errors = highlight_keywords(text, [r"[invalid"], use_regex=True)
        assert counts[r"[invalid"] == 0
        assert r"[invalid" in errors

    def test_empty_keywords(self):
        text = "some text"
        highlighted, counts, errors = highlight_keywords(text, [])
        assert highlighted == text
        assert counts == {}
        assert errors == []
