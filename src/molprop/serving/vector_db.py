import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

log = logging.getLogger(__name__)

class MolecularVectorStore:
    """
    Interfaces with Qdrant to store and retrieve molecular embeddings.
    """
    def __init__(self, collection_name: str = "molecules", location: str = ":memory:"):
        self.client = QdrantClient(location=location)
        self.collection_name = collection_name
        self.vector_size = None

    def create_collection(self, vector_size: int):
        """Build a new collection with cosine similarity."""
        self.vector_size = vector_size
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        log.info(f"Created Qdrant collection: {self.collection_name} (size={vector_size})")

    def upsert_molecules(self, points: List[Dict]):
        """
        Expects a list of dicts with: 'id', 'vector', 'payload' (smiles, etc).
        """
        point_structs = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"]
            )
            for p in points
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=point_structs
        )

    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """ KNN search for the most similar SMILES strings."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "smiles": r.payload.get("smiles"),
                "task_value": r.payload.get("task_value")
            }
            for r in results
        ]

# Global instance for easy API access
vector_store = MolecularVectorStore()
