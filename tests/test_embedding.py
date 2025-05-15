from utils.embeddings import EmbeddingModel


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

    # test embed method with 2 sentences
    embeddings_2= embedder.embed_many(sentenceList)
    assert len(embeddings_2) == 2
    assert len(embeddings_2[0]) == 384
