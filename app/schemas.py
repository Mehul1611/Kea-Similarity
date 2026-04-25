from typing import List
from pydantic import BaseModel, Field


class SimilarityRequest(BaseModel):
    query: str = Field(..., min_length=1)
    samples: List[str] = Field(..., min_length=1)


class SimilarityResult(BaseModel):
    text: str
    score: float


class SimilarityResponse(BaseModel):
    query: str
    best_match: str
    score: float
    all_results: List[SimilarityResult]

