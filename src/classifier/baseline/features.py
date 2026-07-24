"""Hand-crafted stylometric / surface features for human-vs-LLM passage classification."""
import re
import numpy as np
from scipy import sparse

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was", "were",
    "be", "been", "being", "of", "to", "in", "on", "for", "with", "as", "by", "at",
    "this", "that", "these", "those", "it", "its", "from", "into", "than", "which",
    "who", "whom", "will", "would", "can", "could", "should", "may", "might", "must",
    "not", "no", "do", "does", "did", "has", "have", "had", "i", "you", "he", "she",
    "we", "they", "their", "his", "her", "our", "your",
}

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_SPLIT_RE = re.compile(r"[.!?]+")


def extract_features(texts):
    rows = []
    for t in texts:
        t = t if isinstance(t, str) else ""
        n_chars = len(t)
        words = WORD_RE.findall(t)
        n_words = len(words)
        word_lens = [len(w) for w in words]
        avg_word_len = float(np.mean(word_lens)) if word_lens else 0.0
        lower_words = [w.lower() for w in words]
        unique_ratio = len(set(lower_words)) / n_words if n_words else 0.0
        stop_ratio = sum(1 for w in lower_words if w in STOPWORDS) / n_words if n_words else 0.0

        sents = [s for s in SENT_SPLIT_RE.split(t) if s.strip()]
        n_sents = len(sents)
        avg_sent_len = n_words / n_sents if n_sents else float(n_words)

        n_digits = sum(c.isdigit() for c in t)
        n_upper = sum(c.isupper() for c in t)
        n_punct = sum(1 for c in t if c in ".,;:!?-()'\"")
        n_comma = t.count(",")
        n_period = t.count(".")

        digit_ratio = n_digits / n_chars if n_chars else 0.0
        upper_ratio = n_upper / n_chars if n_chars else 0.0
        punct_ratio = n_punct / n_chars if n_chars else 0.0

        # readability-ish: avg syllable proxy via vowel groups
        vowel_groups = sum(len(re.findall(r"[aeiouyAEIOUY]+", w)) for w in words)
        avg_syll = vowel_groups / n_words if n_words else 0.0

        rows.append([
            n_chars,
            n_words,
            avg_word_len,
            unique_ratio,
            stop_ratio,
            n_sents,
            avg_sent_len,
            digit_ratio,
            upper_ratio,
            punct_ratio,
            n_comma / n_sents if n_sents else 0.0,
            n_period / n_sents if n_sents else 0.0,
            avg_syll,
        ])
    return np.asarray(rows, dtype=np.float64)


FEATURE_NAMES = [
    "n_chars", "n_words", "avg_word_len", "unique_word_ratio", "stopword_ratio",
    "n_sents", "avg_sent_len", "digit_ratio", "upper_ratio", "punct_ratio",
    "comma_per_sent", "period_per_sent", "avg_syllables_per_word",
]
