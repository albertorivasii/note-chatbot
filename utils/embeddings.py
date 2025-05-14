from sentence_transformers import SentenceTransformer

model= SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class EmbeddingModel:
    def __init__(self, model:SentenceTransformer):
        self.model= model
    
    def encode_text(self, text:list[str]) -> list:
        return model.encode(text), text
