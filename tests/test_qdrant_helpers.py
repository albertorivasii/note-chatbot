from qdrant_client import QdrantClient
from utils.qdrant_helpers import QdrantHelper
import numpy as np

def test_qdrant_flow():
    client= QdrantClient(host="localhost", port=6333)
    col_name= "test_collection"
    db= QdrantHelper(client)
    creation_result= db.create_collection(col_name, 384)
    assert creation_result == f"Collection Created. Name: {col_name}", creation_result
    
    collections= db.list_collections()
    assert col_name in collections, collections

    # create dummy data
    embeddings= np.random.rand(5, 384)
    payload= [{"location":"london", "score":45},
              {"location":"Los Angeles", "score":100},
              {"location":"Chicago", "score":78},
              {"location":"San Diego", "score":89},
              {"location":"San Francisco", "score":60}
              ]
    
    # upsert dummy data
    upsert_result= db.upsert_embeddings(col_name, embeddings, payload)
    assert upsert_result == "Upsert Complete.", upsert_result
    
    # test delete collection function at the end
    deletion_result= db.delete_collection(col_name)
    assert deletion_result == f"Collection {col_name} deleted.", deletion_result

    client.close()
    