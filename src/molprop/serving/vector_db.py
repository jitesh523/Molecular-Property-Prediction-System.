import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

log = logging.getLogger(__name__)


class MolecularVectorStore:
    """
    Interfaces with Qdrant to store and retrieve molecular embeddings.
    """

    def __init__(self, collection_name: str = "molecules", location: str = ":memory:"):
        self.client = QdrantClient(location=location)
        self.collection_name = collection_name
        self.vector_size: Optional[int] = None

    def create_collection(self, vector_size: int, force_recreate: bool = False) -> None:
        """Build (or recreate) a collection with cosine similarity."""
        self.vector_size = vector_size
        exists = self.client.collection_exists(self.collection_name)
        if exists and force_recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            log.info(f"Created Qdrant collection: {self.collection_name} (size={vector_size})")
        else:
            log.info(f"Reusing existing Qdrant collection: {self.collection_name}")

    def upsert_molecules(self, points: List[Dict]) -> None:
        """
        Upsert a list of molecule embeddings into the collection.

        Each item must have keys: 'id' (int), 'vector' (list[float]), 'payload' (dict).
        """
        point_structs = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points
        ]
        self.client.upsert(collection_name=self.collection_name, points=point_structs)
        log.debug(f"Upserted {len(point_structs)} molecules into {self.collection_name}")

    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """KNN cosine search — returns top-k most similar molecules."""
        results = self.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=top_k
        ).points

        return [
            {
                "id": r.id,
                "score": round(r.score, 6),
                "smiles": r.payload.get("smiles"),
                "task_value": r.payload.get("task_value"),
            }
            for r in results
        ]

    def count(self) -> int:
        """Return the number of indexed molecules."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(collection_name=self.collection_name).count

    def delete_collection(self) -> None:
        """Drop the entire collection (use with care)."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            self.vector_size = None
            log.info(f"Deleted Qdrant collection: {self.collection_name}")


# Global instance for easy API access
vector_store = MolecularVectorStore()
