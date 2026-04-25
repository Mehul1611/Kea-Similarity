import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple 


_MODEL_NAME = "all-MiniLM-L6-v2"


def top_match(samples: List[str], query: str) -> Tuple[str, float]:
    cleaned_samples = [s.strip() for s in samples if s and s.strip()]
    if not cleaned_samples:
        raise ValueError("samples must contain at least one non-empty string")

    q = query.strip()
    if not q:
        raise ValueError("query must be a non-empty string")

    model = SentenceTransformer(_MODEL_NAME)
    sample_embs = model.encode(cleaned_samples, normalize_embeddings=True)
    query_emb = model.encode([q], normalize_embeddings=True)[0]
    scores = np.dot(sample_embs, query_emb)

    best_idx = int(np.argmax(scores))
    return cleaned_samples[best_idx], float(scores[best_idx])


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
