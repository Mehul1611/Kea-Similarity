import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import SimilarityRequest, SimilarityResponse
from .similarity import SimilarityService


app = FastAPI(title="KeaBuilder ML Similarity API", version="1.0.0")
service = SimilarityService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(payload: SimilarityRequest):
    samples = [s for s in payload.samples if s and s.strip()]
    if not samples:
        raise HTTPException(status_code=400, detail="samples must contain at least one non-empty string")

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    sample_embeddings = service.embed(samples)
    query_embedding = service.embed([query])[0]
    scores = np.dot(sample_embeddings, query_embedding)

    ranked = sorted(
        [{"text": samples[i], "score": float(scores[i])} for i in range(len(samples))],
        key=lambda x: x["score"],
        reverse=True,
    )

    best = ranked[0]
    return {
        "query": query,
        "best_match": best["text"],
        "score": best["score"],
        "all_results": ranked,
    }

