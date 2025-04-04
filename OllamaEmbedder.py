from langchain_ollama import OllamaEmbeddings

class OllamaEmbedder:

    ollama_embedder = OllamaEmbeddings(model="nomic-embed-text")

    def get_embedding(self, text):
        return self.ollama_embedder.embed_query(text)