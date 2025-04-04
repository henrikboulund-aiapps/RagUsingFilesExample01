from numpy import dot
from numpy.linalg import norm

class SimilaritySearch:

    def cosine_similarity(vec1, vec2):
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


    def rank_documents(self, documents, query_embedding):
        results = sorted(
            documents, key=lambda x: self.cosine_similarity(query_embedding, x["embedding"]), reverse=True)
        return results
