from utils.qdrant_helpers import *
import numpy as np

def test_qdrant_flow():
    client= QdrantClient(host="localhost", port=6333)
    col_name= "test_collection"
    creation_result= create_collection(col_name, 384)
    assert creation_result == f"Collection Created. Name: {col_name}"

    # create dummy data
    embeddings= np.random.rand(5, 384)
    payload= [{"location":"london", "score":45},
              {"location":"Los Angeles", "score":100},
              {"location":"Chicago", "score":78},
              {"location":"San Diego", "score":89},
              {"location":"San Francisco", "score":60}
              ]
    
    # upsert dummy data
    upsert_result= upsert_embeddings(col_name, embeddings, payload)
    assert upsert_result == "Upsert Complete."
    
    # test delete collection function at the end
    deletion_result= delete_collection(col_name)
    assert deletion_result == f"Collection {col_name} deleted."

    client.close()
    