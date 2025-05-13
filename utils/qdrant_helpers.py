from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range
import numpy as np
from uuid import uuid4

client= QdrantClient(host="localhost", port=6333)

class QdrantHelper:
	def __init__(self, client):
		self.client= client


	def create_collection(self, name:str, vec_len:int) -> str:
		"""
		Create a Collection in Qdrant if it does not exist.

		Args:
			name (str): Name of collection
			vec_len (int): Length of the vectors to store
		
		Returns:
			str: "Collection created. Name: {name}" if successful else "Collection {name} already exists."
		
		"""	
		if not self.client.collection_exists(name):
			self.client.create_collection(
				collection_name=name,
				vectors_config=VectorParams(size=vec_len, distance=Distance.COSINE)
			)
			return f"Collection Created. Name: {name}."
		else:
			return f"Collection {name} already exists."


	def delete_collection(self, name:str) -> str:
		"""
		Delete collection from Qdrant client
		
		Args:
			name (str): Name of collection to delete

		Returns:
			str: "Collection {name} deleted." if successful. Else "Collection {name} not deleted."
		"""
		try:
			self.client.delete_collection(name)
			return f"Collection {name} deleted."
		except Exception as e:
			raise ValueError(f"Unable to delete collection. {e}")


	def create_filter(self, params:dict) -> Filter:
		"""
		Create a Qdrant Filter object using the params argument

		Args:
			params (dict): arguments that must be true for the search query
			optional (dict): arguments that are optional for the search query

		Returns:
			filters (Filter): Qdrant Filter object with specified parameters
		"""
		def parse_condition(field, condition):
			if isinstance(condition, tuple):
				op, val = condition
				return FieldCondition(
					key=field,
					range=Range(
						gte=val if op == ">=" else None,
						gt=val if op == ">" else None,
						lte=val if op == "<=" else None,
						lt=val if op == "<" else None
					)
				)
			else:
				return FieldCondition(
					key=field,
					match=MatchValue(value=condition)
				)
		
		must= []
		should= []

		for clause in ["must", "should"]:
			clause_info= params.get(clause, {})
			for field, condition in clause_info.items():
				cond= parse_condition(field, condition)
				if clause == "must":
					must.append(cond)
				else:
					should.append(cond)
		return Filter(must=must, should=should)


	def search_collection(self, name:str, query_vec:np.array, max_results:int=5) -> PointStruct:
		"""
		Return the most similar searches to the query vector.

		Args:
			name (str): Name of the collection
			query_vec (np.array): vectorized query
			limit (int): maximum number of results to return

		Returns:
			hits (dict): dictionary including payload of top X most similar vectors in collection.
		"""
		try:
			hits= self.client.search(
				collection_name=name,
				query_vector=query_vec,
				limit=max_results
			)

			return hits
		except Exception as e:
			raise ValueError(f"Unable to access results. Details: {e}")
		

	def upsert_embeddings(self, name:str, embeddings:np.array, payload:dict) -> str:
		"""
		Upsert embeddings and payload into Qdrant Client.

		Args:
			embedding (np.array[float]): Numpy array of embeddings
			payload (dict): Payload associated with the embeddings
		
		Returns:
			str: "Upsert Complete." if successful else "Upsert Failed. Details: {e}" where e is the error
		"""
		try:
			self.client.upsert(
				collection_name=name,
				points= [
					PointStruct(
						id=uuid4(),
						vector=vector.tolist(),
						payload=payload
					)
					for vector in embeddings
				]
			)
			return "Upsert Complete."
		except Exception as e:
			raise ValueError(f"Upsert Failed. Details: {e}")

