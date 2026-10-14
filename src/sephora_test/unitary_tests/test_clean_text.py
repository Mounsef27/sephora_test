import pytest

# Adapte cet import à ton projet.
from sephora_test.apply_transformation.clean_text import clean_text


class TestCleanText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Lowercase.
            ("BONJOUR MONDE", "bonjour monde"),

            # HTML tags.
            ("<p>Bonjour</p><strong>monde</strong>", "bonjour monde"),

            # HTML entities.
            ("caf&eacute; &amp; parfum", "café parfum"),

            # Spaces, tabs and line breaks.
            ("  bonjour \t monde\n  ", "bonjour monde"),

            # Standard punctuation.
            ("bonjour, monde!", "bonjour monde"),

            # Additional punctuation.
            ("«bonjour»—“monde”…", "bonjour monde"),

            # French stopwords.
            ("le parfum et la crème", "parfum crème"),

            # Accents are preserved.
            ("CRÈME ÉCLAT", "crème éclat"),

            # Empty inputs.
            ("", ""),
            ("   \t\n", ""),

            # Punctuation only.
            ("!!!...", ""),

            # Stopwords only.
            ("le la les et", ""),

            # Empty HTML.
            ("<p></p>", ""),
        ],
    )
    def test_clean_text(self, text, expected):
        assert clean_text(text) == expected

    @pytest.mark.parametrize(
        "text",
        [None, 123, 3.14, [], {}, True],
    )
    def test_invalid_input_returns_empty_string(self, text):
        assert clean_text(text) == ""

    def test_clean_text_is_idempotent(self):
        text = "<p>La CRÈME, et le parfum !</p>"

        cleaned = clean_text(text)

        assert clean_text(cleaned) == cleaned