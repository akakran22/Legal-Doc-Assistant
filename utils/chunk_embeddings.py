# import os
# import requests
# import numpy as np
# from tqdm import tqdm
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# class HuggingFaceEmbeddings:
#     def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
#         self.model_name = model_name
#         self.api_key = os.getenv("HUGGINGFACE_API_KEY")
#         if not self.api_key:
#             raise ValueError("HUGGINGFACE_API_KEY required")
        
#         self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
#         self.headers = {"Authorization": f"Bearer {self.api_key}"}

#     def encode(self, texts, batch_size=8, show_progress=True):
#         if isinstance(texts, str):
#             texts = [texts]
        
#         all_embeddings = []
#         batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
#         if show_progress:
#             batches = tqdm(batches, desc="Creating embeddings")
        
#         for batch in batches:
#             try:
#                 response = requests.post(
#                     self.api_url,
#                     headers=self.headers,
#                     json={"inputs": batch, "options": {"wait_for_model": True}}
#                 )
                
#                 if response.status_code == 200:
#                     batch_embeddings = response.json()
#                     if isinstance(batch_embeddings[0], list) and isinstance(batch_embeddings[0][0], (int, float)):
#                         batch_embeddings = [batch_embeddings]
#                     all_embeddings.extend(batch_embeddings)
#                 else:
#                     print(f"API Error: {response.status_code}")
#                     dummy_embedding = [0.0] * 768
#                     all_embeddings.extend([dummy_embedding] * len(batch))
                    
#             except Exception as e:
#                 print(f"Error: {e}")
#                 dummy_embedding = [0.0] * 768
#                 all_embeddings.extend([dummy_embedding] * len(batch))
        
#         embeddings = np.array(all_embeddings, dtype=np.float32)
        
#         # Normalize
#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         norms[norms == 0] = 1
#         embeddings = embeddings / norms
        
#         return embeddings

# def chunk_documents(docs, chunk_size=3000, chunk_overlap=300):
#     separators = ["\n\n", "\n", "Section ", "SECTION ", "Sec. ", "CHAPTER ", "Chapter ", " "]
    
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=separators
#     )
    
#     chunks = splitter.split_documents(docs)
#     print(f"Created {len(chunks)} chunks")
#     return chunks

# def chunk_and_embed_documents(docs, vdb, chunk_size=3000, chunk_overlap=300, batch_size=8):
#     print("Chunking documents...")
#     chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    
#     texts = [chunk.page_content for chunk in chunks]
#     metadatas = [chunk.metadata for chunk in chunks]
    
#     print("Creating embeddings...")
#     embeddings_model = HuggingFaceEmbeddings()
#     embeddings = embeddings_model.encode(texts, batch_size=batch_size)
    
#     print("Storing in Qdrant...")
#     vdb.add_documents(texts, embeddings, metadatas)
#     print(f"Stored {len(chunks)} chunks")
















import os
import time
import requests
import numpy as np
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY required")

        # Router-based hf-inference pipeline for feature-extraction (embeddings)
        self.api_url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{model_name}/pipeline/feature-extraction"
        )

        # Optional fallback: same task route (useful if you want to toggle options)
        self.fallback_url = self.api_url

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_embeddings(self, batch, url, normalize=True, truncate=True, wait_for_model=True, timeout=60):
        payload = {
            "inputs": batch,  # string or list[str]
            "parameters": {"normalize": normalize, "truncate": truncate},
            "options": {"wait_for_model": wait_for_model},
        }
        return requests.post(url, headers=self.headers, json=payload, timeout=timeout)

    def encode(self, texts, batch_size=8, show_progress=True, normalize=True):
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        if show_progress:
            batches = tqdm(batches, desc="Creating embeddings")

        for batch in batches:
            success = False
            for attempt in range(3):  # retry up to 3 times
                try:
                    response = self._request_embeddings(batch, self.api_url, normalize=normalize)
                    if response.status_code != 200:
                        # Optional fallback (same route, different wait policy)
                        response = self._request_embeddings(batch, self.fallback_url, normalize=normalize, wait_for_model=True)

                    if response.status_code == 200:
                        data = response.json()
                        # Ensure shape is list-of-vectors
                        # HF may return [float, ...] for single input; wrap to [ [float, ...] ]
                        if isinstance(data, list) and data and isinstance(data[0], float):
                            data = [data]
                        all_embeddings.extend(data)
                        success = True
                        break
                    else:
                        print(f"API Error {response.status_code}: {response.text[:200]}, retrying in 5s...")
                        time.sleep(5)
                except Exception as e:
                    print(f"Request error: {e}, retrying in 5s...")
                    time.sleep(5)

            if not success:
                print("❌ Failed to get embeddings after retries, using dummy vectors")
                dim = 768  # all-mpnet-base-v2
                dummy_embedding = [0.0] * dim
                all_embeddings.extend([dummy_embedding] * len(batch))

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Normalize client-side (useful if server normalize=False or for consistency)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        return embeddings

def chunk_documents(docs, chunk_size=3000, chunk_overlap=300):
    separators = ["\n\n", "\n", "Section ", "SECTION ", "Sec. ", "CHAPTER ", "Chapter ", " "]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def chunk_and_embed_documents(docs, vdb, chunk_size=3000, chunk_overlap=300, batch_size=8):
    print("📖 Chunking documents...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    print("🧠 Creating embeddings...")
    embeddings_model = HuggingFaceEmbeddings()
    embeddings = embeddings_model.encode(texts, batch_size=batch_size)

    print("💾 Storing in Qdrant...")
    vdb.add_documents(texts, embeddings, metadatas)
    print(f"✅ Stored {len(chunks)} chunks")
