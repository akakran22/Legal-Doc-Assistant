import os
import re
from langchain_community.document_loaders import PyPDFLoader

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\.{3,}', '...', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_documents(docs):
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        # Ensure source and page metadata exist
        if 'source' not in doc.metadata or not doc.metadata.get('source'):
            doc.metadata['source'] = 'unknown'
        # ensure page number exists if loader provided it
        if 'page' not in doc.metadata:
            # some loaders set page; fallback to 1
            doc.metadata['page'] = doc.metadata.get('page', 1)
    return docs

def load_all_pdfs(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pdf_names = [n for n in os.listdir(data_dir) if n.lower().endswith(".pdf")]
    if not pdf_names:
        raise FileNotFoundError(f"No PDFs found in: {data_dir}")

    print(f"Found {len(pdf_names)} PDF files")
    docs = []

    for name in pdf_names:
        path = os.path.join(data_dir, name)
        try:
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            # Normalize metadata.source to just filename and set page
            for i, page in enumerate(pdf_docs, start=1):
                # ensure metadata exists
                if not isinstance(page.metadata, dict):
                    page.metadata = {}
                page.metadata['source'] = name
                # Some loaders include page number already; set/overwrite to be consistent
                page.metadata['page'] = page.metadata.get('page', i)
            docs.extend(pdf_docs)
            print(f"Loaded {name}: {len(pdf_docs)} pages")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    return docs

def load_and_preprocess_pdfs(data_dir):
    print("Loading PDF documents...")
    docs = load_all_pdfs(data_dir)
    print(f"Total pages: {len(docs)}")

    print("Preprocessing documents...")
    return preprocess_documents(docs)

