import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.pop("SSL_CERT_FILE", None)

from qdrant_client import QdrantClient
from utils.qdrant_helpers import QdrantHelper
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=".env.test")

QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
QDRANT_URL= os.getenv("QDRANT_URL")


def test_qdrant_flow():
    try:
        client= QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, verify=False)
        col_name= "test_collection"
        db= QdrantHelper(client)
        creation_result= db.create_collection(col_name, 384)
        assert creation_result == f"Collection Created. Name: {col_name}.", creation_result
        
        collections= db.list_collections()
        assert col_name in collections, collections

        # create dummy data
        now_str= datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
        embeddings= np.random.rand(5, 384)
        payload= [{"location":"London", "score":45, "created_at":now_str},
                {"location":"Los Angeles", "score":100, "created_at":now_str},
                {"location":"Chicago", "score":78, "created_at":now_str},
                {"location":"San Diego", "score":89, "created_at":now_str},
                {"location":"San Francisco", "score":60, "created_at":now_str}
                ]
        
        # upsert dummy data
        upsert_result= db.upsert_embeddings(col_name, embeddings, payload)
        assert upsert_result == "Upsert Complete.", upsert_result
        
        # generate indices on payload fields
        location_idx= db.create_field_index(col_name, "location", "keyword")
        assert location_idx == "Index created on location."

        score_idx= db.create_field_index(col_name, "score", "float")
        assert score_idx == "Index created on score."

        # generate filter
        filter_params = {
            "must": {
                "location": "San Francisco"
            },
            "should": {
                "score": (">", 50)
            }
        }
        qdrant_filter= db.create_filter(filter_params)
        # testing search function
        query= embeddings[0]
        results= db.search_collection(col_name, query, filters=qdrant_filter)
        assert results, "No results"

        # print for debugging
        # print(f"[DEBUG]: Results: ", results)
        # print(f"[DEBUG] Type of Results: {type(results)}")
        # print(f"[DEBUG] Type of first entry of Results: {type(results[0])}")

        # look for SF in results
        for res in results:
            payload= res.payload
            assert payload["location"] == "San Francisco", f"Unexpected Payload: {payload}"
            assert payload["score"] >= 50, f"Score too low. Check Filter. Payload: {payload}"


    finally:
        try:
            db.delete_collection(col_name)
        except Exception as e:
            print(f"Unable to delete collection {col_name}. Closing out db. Details: {e}")
        client.close()
    