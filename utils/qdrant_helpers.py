from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range, ScoredPoint
import numpy as np
from uuid import uuid4

client= QdrantClient(host="localhost", port=6333)

class QdrantHelper:
	def __init__(self, client:QdrantClient):
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


	def list_collections(self) -> list[str]:
		"""
		Return list of collection names.

		Args: None

		Returns:
			list[str]: Python list of strings corresponding to collection names in client.
		"""
		return [c.name for c in self.client.get_collections().collections]


	def create_filter(self, params:dict) -> Filter:
		"""
		Create a Qdrant Filter object using the params argument

		Args:
			params (dict): arguments that must be true for the search query

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
		must_not= []

		for clause in ["must", "should", "must_not"]:
			clause_info= params.get(clause, {})
			for field, condition in clause_info.items():
				cond= parse_condition(field, condition)
				if clause == "must":
					must.append(cond)
				elif clause == "must_not":
					must_not.append(cond)
				else:
					should.append(cond)
		return Filter(must=must or None, should=should or None, must_not=must_not or None)


	def search_collection(self, name:str, query_vec:np.array, max_results:int=5, filters:Filter=None) -> list[ScoredPoint]:
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
			hits= self.client.query_points(
				collection_name=name,
				query= query_vec.tolist(),
				limit=max_results,
				query_filter=filters,
				with_payload=True,
				with_vectors=False
			)

			return hits.points
		except Exception as e:
			raise ValueError(f"Unable to access results. Details: {e}")
		

	def upsert_embeddings(self, name:str, embeddings:np.array, payloads:list[dict]) -> str:
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
						id=str(uuid4()),
						vector=vector.tolist(),
						payload=payload
					)
					for vector, payload in zip(embeddings, payloads)
				]
			)
			return "Upsert Complete."
		except Exception as e:
			raise ValueError(f"Upsert Failed. Details: {e}")


	def create_field_index(self, name:str, field:str, schema:str) -> str:
		"""
		Creates a Qdrant Index on a given Payload field.

		Args:
			name (str): Name of the Qdrant collection.
			field (str): Name of the field in the collection to create an index on.
			field_schema (str): Schema corresponding to the field

		Returns:
			str: "Index created on {field}." if successful, else "Index creation failed. Details: {e}" where e is the error.
		"""
		try:
			self.client.create_payload_index(name, field, schema)
			return f"Index created on {field}."
			
		except Exception as e:
			return f"Index creation failed. Details: {e}"


