# import uuid
# from typing import List, Dict
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import Distance, VectorParams, PointStruct
# import numpy as np

# class QdrantDB:
#     def __init__(self, url, api_key, collection_name, vector_size=768, timeout=180):
#         # REST-only: do not enable gRPC; set a higher timeout for bulk ops
#         self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
#         self.collection_name = collection_name
#         self.vector_size = vector_size
#         self._ensure_collection()

#     def _ensure_collection(self):
#         try:
#             collections = self.client.get_collections()
#             collection_names = [col.name for col in collections.collections]
#             if self.collection_name not in collection_names:
#                 print(f"Creating collection: {self.collection_name}")
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
#                 )
#                 print("Collection created")
#             else:
#                 print("Collection exists")
#         except Exception as e:
#             print(f"Collection error: {e}")
#             raise

#     def collection_exists(self):
#         try:
#             collections = self.client.get_collections()
#             return self.collection_name in [col.name for col in collections.collections]
#         except Exception:
#             return False

#     def get_collection_info(self):
#         try:
#             info = self.client.get_collection(self.collection_name)
#             return {
#                 "points_count": info.points_count,
#                 "vectors_count": getattr(info, "vectors_count", None),
#                 "status": getattr(info, "status", "unknown"),
#             }
#         except Exception as e:
#             print(f"Info error: {e}")
#             return {"points_count": 0, "vectors_count": 0, "status": "error"}

#     def add_documents(self, texts, embeddings, metadatas):
#         points = []
#         for text, embedding, metadata in zip(texts, embeddings, metadatas):
#             point_id = str(uuid.uuid4())
#             payload = {
#                 "text": text,
#                 "source": metadata.get("source", "unknown"),
#                 "page": metadata.get("page", 0),
#                 **metadata,
#             }
#             vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
#             points.append(PointStruct(id=point_id, vector=vec, payload=payload))

#         batch_size = 100  # moderate batch to avoid REST read timeouts
#         total_batches = (len(points) + batch_size - 1) // batch_size
#         for i in range(0, len(points), batch_size):
#             batch = points[i:i + batch_size]
#             try:
#                 # wait=True ensures the write is applied before returning
#                 self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
#                 print(f"Batch {i//batch_size + 1}/{total_batches} uploaded")
#             except Exception as e:
#                 print(f"Upload error: {e}")
#                 raise

#         print(f"Added {len(points)} points")

#     def search(self, query_embedding, top_k=6):
#         try:
#             if isinstance(query_embedding, np.ndarray):
#                 if query_embedding.ndim == 2:
#                     query_vector = query_embedding[0].astype(np.float32).tolist()
#                 else:
#                     query_vector = query_embedding.astype(np.float32).tolist()
#             else:
#                 query_vector = list(query_embedding)

#             # Prefer the modern Query API to avoid deprecation warnings
#             results = self.client.query_points(
#                 collection_name=self.collection_name,
#                 query=query_vector,
#                 limit=top_k,
#                 with_payload=True,
#             ).points

#             hits = []
#             for rank, hit in enumerate(results, 1):
#                 # Qdrant returns higher-is-better scores; present as percentage-like scale if needed
#                 relevance = float(hit.score) if hit.score is not None else 0.0
#                 hits.append({
#                     "rank": rank,
#                     "score": relevance,
#                     "text": hit.payload.get("text", ""),
#                     "payload": hit.payload,
#                 })
#             return hits

#         except Exception as e:
#             print(f"Search error: {e}")
#             return []

#     def clear_collection(self):
#         try:
#             self.client.delete_collection(self.collection_name)
#             self._ensure_collection()
#             print("Collection cleared")
#         except Exception as e:
#             print(f"Clear error: {e}")




import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import numpy as np

class QdrantDB:
    def __init__(self, url, api_key, collection_name, vector_size=768, timeout=180):
        # REST-only: do not enable gRPC; set a higher timeout for bulk ops
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                print("Collection created")
            else:
                print("Collection exists")
        except Exception as e:
            print(f"Collection error: {e}")
            raise

    def collection_exists(self):
        try:
            collections = self.client.get_collections()
            return self.collection_name in [col.name for col in collections.collections]
        except Exception:
            return False

    def get_collection_info(self):
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", None),
                "status": getattr(info, "status", "unknown"),
            }
        except Exception as e:
            print(f"Info error: {e}")
            return {"points_count": 0, "vectors_count": 0, "status": "error"}

    def add_documents(self, texts, embeddings, metadatas):
        points = []
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            point_id = str(uuid.uuid4())
            payload = {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", 0),
                **metadata,
            }
            vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        batch_size = 100  # moderate batch to avoid REST read timeouts
        total_batches = (len(points) + batch_size - 1) // batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                # wait=True ensures the write is applied before returning
                self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
                print(f"Batch {i//batch_size + 1}/{total_batches} uploaded")
            except Exception as e:
                print(f"Upload error: {e}")
                raise

        print(f"Added {len(points)} points")

    def search(self, query_embedding, top_k=6):
        try:
            if isinstance(query_embedding, np.ndarray):
                if query_embedding.ndim == 2:
                    query_vector = query_embedding[0].astype(np.float32).tolist()
                else:
                    query_vector = query_embedding.astype(np.float32).tolist()
            else:
                query_vector = list(query_embedding)

            # Prefer the modern Query API to avoid deprecation warnings
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            ).points

            hits = []
            for rank, hit in enumerate(results, 1):
                # Qdrant returns higher-is-better scores; present as percentage-like scale if needed
                relevance = float(hit.score) if hit.score is not None else 0.0
                hits.append({
                    "rank": rank,
                    "score": relevance,
                    "text": hit.payload.get("text", ""),
                    "payload": hit.payload,
                })
            return hits

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            print("Collection cleared")
        except Exception as e:
            print(f"Clear error: {e}")
