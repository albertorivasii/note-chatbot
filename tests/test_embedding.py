import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.pop("SSL_CERT_FILE", None)

from utils.embeddings import EmbeddingModel
import numpy as np


def test_embedding_model():
    embedder= EmbeddingModel()

    sentence= "This is a sentence"

    sentenceList= ["This is the first sentence", "This is the second sentence"]

    # test if the correct embedding model is being loaded
    dim= embedder.dim()
    assert dim == 384, f"Dimension is not as expected. Dim: {dim}"

    # test embed method with 1 sentence
    embeddings= embedder.embed_one(sentence)
    assert len(embeddings) == 384
    assert any(val != 0 for val in embeddings)

    # test embed method with 2 sentences
    embeddings_2= embedder.embed_many(sentenceList)
    assert len(embeddings_2) == 2
    assert len(embeddings_2[0]) == 384

    # test model info
    model_info= embedder.model_info()
    assert model_info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert model_info["device"] == "cuda"
    assert model_info["embedding_dim"] == 384

    # test embed with ids
    ids= ["1", "2"]
    embed_dict= embedder.embed_with_ids(sentenceList, ids)
    assert isinstance(embed_dict, list)
    assert embed_dict[0]["id"] == "1"
    assert len(embed_dict[0]["vector"]) == 384

    # test embed in batches
    batch_embeddings= embedder.embed_batches(sentenceList, 2)
    assert len(batch_embeddings) == 2
    assert len(batch_embeddings[0]) == 384


    # testing similarity of two sentences.
    similar1= embedder.embed_one("This is a test.")
    similar2= embedder.embed_one("This is a test!")

    similarity= np.dot(similar1, similar2) / (np.linalg.norm(similar1) * np.linalg.norm(similar2))
    assert similarity > 0.90, f"Expected high similarity. Got: {similarity}"

    # testing for valid input
    try:
        embedder.embed_one(123)
        assert False, "Should have raised a type error"
    except TypeError:
        pass

