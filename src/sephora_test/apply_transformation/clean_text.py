"""Clean text by removing HTML tags, special characters,
and extra spaces.
"""

import html
import re
import string

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sephora_test.logger import setup_logger

logger = setup_logger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt_tab")
    logger.info("NLTK resources are already downloaded.")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)


penctuations = set(string.punctuation + "«»“”‘’—…")
frensh_stopwords = set(stopwords.words("french"))
logger.info(
    "Loaded %d French stopwords and %d punctuation characters.",
    len(frensh_stopwords),
    len(penctuations),
)


def clean_text(text: str) -> str:
    """
    Clean the input text by removing HTML tags,
      special characters, and extra spaces.

    Args:
        text (str): The input text to be cleaned.
    returns:
        str: The cleaned text.
    """
    if text is None or not isinstance(text, str):
        logger.warning(f"Invalid input: {text}. Returning empty string.")

    logger.info(
        f"Cleaning text: {text[:30]}..."
    )  # Log the first 30 characters of the text
    text = html.unescape(text)
    logger.info(f"After unescaping HTML: {text[:30]}...")

    text = BeautifulSoup(text, "lxml").get_text(" ", strip=True)
    logger.info(f"After removing HTML tags: {text[:30]}...")
    text = re.sub(
        r"\s+", " ", text
    )  # Replace multiple spaces with a single space
    logger.info(f"After normalizing spaces: {text[:30]}...")
    text = text.strip()  # Remove leading and trailing spaces
    logger.info(f"After stripping leading/trailing spaces: {text[:30]}...")

    for punctuation in penctuations:
        text = text.replace(punctuation, " ")
    logger.info(f"After removing punctuation: {text[:30]}...")

    text = text.lower()  # just lowercasing the text
    text = " ".join(
        word for word in text.split() if word not in frensh_stopwords
    )
    logger.info(f"After removing stopwords: {text[:30]}...")

    return text
