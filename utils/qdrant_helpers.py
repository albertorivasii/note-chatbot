from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition
import numpy as np

client= QdrantClient(host="localhost", port=6333)


def create_collection(name:str, vec_len:int, ) -> str:
	"""
	Create a Collection in Qdrant if it does not exist.

	Args:
		name (str): Name of collection
		vec_len (int): Length of the vectors to store
	
	Returns:
		str: "Collection created. Name: {name}" if successful else "Collection {name} already exists."
	
	"""	
	if not client.collection_exists(name):
		client.create_collection(
			collection_name=name,
			vectors_config=VectorParams(size=vec_len, distance=Distance.COSINE)
		)
		return f"Collection Created. Name: {name}."
	else:
		return f"Collection {name} already exists."


def delete_collection(name:str) -> str:
	"""
	Delete collection from Qdrant client
	
	Args:
		name (str): Name of collection to delete

	Returns:
		str: "Collection {name} deleted." if successful. Else "Collection {name} not deleted."
	"""
	try:
		client.delete_collection(name)
		return f"Collection {name} deleted"
	except Exception as e:
		raise ValueError(f"Unable to delete collection. {e}")


def search_collection(name:str, query_vec:np.array, max_results:int=5) -> PointStruct:
	"""
	Return the top X most similar searches to the query vector.

	Args:
		name (str): Name of the collection
		query_vec (np.array): vectorized query
		limit (int): maximum number of results to return

	Returns:
		hits (dict): dictionary including payload of top X most similar vectors in collection.
	"""
	try:
		hits= client.search(
			collection_name=name,
			query_vector=query_vec,
			limit=max_results
		)

		return hits
	except Exception as e:
		raise ValueError(f"Unable to access results: {e}")
	

def create_filter(params:dict, optional:dict) -> Filter:
	"""
	Create a Qdrant Filter object using the params argument

	Args:
		params (dict): arguments that must be true for the search query
		optional (dict): arguments that are optional for the search query

	Returns:
		filters (Filter): Qdrant Filter object with 
	"""
	# TODO: add optional dictionary support
	filters= Filter(
		must= [
			FieldCondition(
				key= field,
				
			)
		for field, val in params.items()]
	)
	pass
