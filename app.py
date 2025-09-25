import os
import time
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Local utils
from utils.read_process_data import load_and_preprocess_pdfs
from utils.chunk_embeddings import chunk_and_embed_documents
from utils.vector_database import QdrantDB
from utils.retrieval_qa import RetrievalQA

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL environment variable required")
if not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_API_KEY environment variable required")

collection_name = "indian_legal_docs"
vdb = QdrantDB(url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=collection_name)

# Optional: Auto-reset collection every time
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "False").lower() == "true"
if RESET_COLLECTION:
    print("Resetting Qdrant collection...")
    vdb.clear_collection()

# Check collection status
try:
    info = vdb.get_collection_info()
    points = info.get("points_count", 0)
except Exception:
    points = 0

# Build collection if empty
if points == 0:
    print("Building Qdrant vector collection from PDFs in data/ ...")
    t0 = time.time()
    docs = load_and_preprocess_pdfs(DATA_DIR)
    print(f"Loaded {len(docs)} pages; starting chunk+embed...")

    chunk_size = int(os.getenv("CHUNK_SIZE", 1500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    batch_size = int(os.getenv("EMBED_BATCH", 8))

    chunk_and_embed_documents(
        docs=docs,
        vdb=vdb,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size
    )
    print(f"Collection built in {time.time() - t0:.1f}s")
else:
    print(f"Using existing collection with {points} vectors")

qa = RetrievalQA(vdb)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    try:
        info = vdb.get_collection_info()
        return jsonify({"status": "OK", "points": info.get("points_count", 0)})
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    query = (data.get("question") or data.get("q") or "").strip()

    if not query:
        return jsonify({"error": "Empty question"}), 400

    try:
        answer, hits = qa.ask(query, top_k=6, max_tokens=1200)

        sources = []
        for idx, h in enumerate(hits, start=1):
            payload = h.get("payload", {}) or {}
            src = payload.get("source", payload.get("act_name", "Legal Document"))
            try:
                src_short = os.path.basename(src)
            except Exception:
                src_short = src
            sources.append({
                "rank": idx,
                "score": h.get("score", 0),
                "text": (h.get("text") or "")[:800],
                "meta": {
                    "source": src_short,
                    "page": payload.get("page", "N/A"),
                    "act_name": payload.get("act_name", "")
                }
            })

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        print(f"Error in /ask: {e}")
        return jsonify({"error": f"Processing error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)), debug=True)

