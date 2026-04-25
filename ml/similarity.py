import re
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_vocab(texts: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for t in texts:
        for tok in _tokenize(t):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _vectorize(text: str, vocab: Dict[str, int]) -> List[float]:
    v = [0.0] * len(vocab)
    for tok in _tokenize(text):
        idx = vocab.get(tok)
        if idx is not None:
            v[idx] += 1.0
    return v


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y

    if na == 0.0 or nb == 0.0:
        return 0.0

    return dot / ((na**0.5) * (nb**0.5))


def top_match(samples: List[str], query: str) -> Tuple[str, float]:
    cleaned_samples = [s.strip() for s in samples if s and s.strip()]
    if not cleaned_samples:
        raise ValueError("samples must contain at least one non-empty string")

    q = query.strip()
    if not q:
        raise ValueError("query must be a non-empty string")

    vocab = _build_vocab(cleaned_samples + [q])
    qv = _vectorize(q, vocab)

    best_text = cleaned_samples[0]
    best_score = -1.0
    for s in cleaned_samples:
        sv = _vectorize(s, vocab)
        score = cosine_similarity(sv, qv)
        if score > best_score:
            best_score = score
            best_text = s

    return best_text, best_score


if __name__ == "__main__":
    samples = [
        "I want to capture leads for my business",
        "How do I build a sales funnel?",
        "Set up an AI chatbot for customer support",
        "Generate content for my landing page",
        "Automate my email follow-ups",
    ]
    query = "I need a chatbot to handle customer queries"

    match, score = top_match(samples, query)
    print({"query": query, "best_match": match, "score": score})
