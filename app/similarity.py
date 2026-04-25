from sentence_transformers import SentenceTransformer
import numpy as np


class SimilarityService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def run_similarity_search(self, samples, query):
        sample_embeddings = self.embed(samples)
        query_embedding = self.embed([query])[0]
        scores = np.dot(sample_embeddings, query_embedding)

        best_index = np.argmax(scores)

        return {
            "query": query,
            "best_match": samples[best_index],
            "score": float(scores[best_index]),
        }


if __name__ == "__main__":
    samples = [
        "I want to capture leads for my business",
        "How do I build a sales funnel?",
        "Set up an AI chatbot for customer support",
        "Generate content for my landing page",
        "Automate my email follow-ups",
    ]

    query = "I need a chatbot to handle customer queries"

    service = SimilarityService()
    result = service.run_similarity_search(samples, query)
    print(result)
