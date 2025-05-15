from sentence_transformers import SentenceTransformer
from typing import List, Union

class EmbeddingModel:
    def __init__(self, model:str="sentence-transformers/all-MiniLM-L6-v2", device:str="cuda"):
        """
        initializes the embedding model.

        Args:
            model_name (str): Name of the SentenceTranformer model.
            device (str): Device to load the model on ("cpu" or "cuda")
        """
        self.model_name= model
        self.device= device
        self.model= SentenceTransformer(model, device=device)
    

    def embed(self, text:List[str]) -> Union[List[float], List[List[float]]]:
        """
        Embed a list of sentences with the encoding model.

        Args:
            text (str or list): Text or list of texts to embed.

        Returns:
            list[float]: Embedding vector(s).
        """
        return self.model.encode(text, convert_to_numpy=True).tolist()


    def dim(self) -> int:
        """
        Returns the dimensionality of the embedding vectors.
        """
        dummy= self.model.encode("test")
        return len(dummy) if isinstance(dummy, list) and not isinstance(dummy[0], list) else len(dummy[0])
    
    
    def model_info(self) -> dict:
        return {
            "model_name":self.model_name,
            "device":self.device,
            "embedding_dim":self.dim()
        }


    def embed_with_ids(self, texts:List[str], ids:List[str]) -> List[dict]:
        """
        Create a dictionary where keys are the ids and values are the embeddings.

        Args:
            texts (list[str]): texts to be embedded.
            ids (list[str]): IDs for each text
        
        Returns:
            list[dict]: List of dictionaries where keys are IDs and values are the embeddings for each sentence provided.
        """
        vectors= self.embed(texts)
        return [{"id":id_, "vector":vec} for id_, vec in zip(ids, vectors)]
    
    
    def embed_batches(self, texts:List[str], batch_size:int=32) -> List[List[float]]:
        """
        Generate embeddings for batches of sentences.

        Args:
            texts (List[str]): list of texts to embed.
            batch_size (int): Number of embeddings to generate at a time.
        
        Returns:
            List[List[float]]: List containing lists corresponding to each vector for a given text.
        """
        all_vectors= []

        for i in range(0, len(texts), batch_size):
            batch= texts[i:i+batch_size]
            vectors= self.model.encode(batch, convert_to_numpy=True)
            all_vectors.extend(vectors)
        
        return all_vectors
    
    # TODO: Create save_embeddings method
    def save_embeddings(self, path:str):
        """
        Save embeddings for given sentence(s) to disk.

        Args:
            path (str): Path to save the embeddings to.
        """
        pass

    # TODO: Create load_embeddings method
    def load_embeddings(self, path:str):
        """
        Load embeddings from disk to RAM.

        Args:
            path (str): Path to file containing the embeddings.
        """
        pass

