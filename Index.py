
import os
import re
import hashlib
from pathlib import Path

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR
INDEX_DIR = BASE_DIR / "faiss_index"

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)

API_URL = "https://api.openai.com/v1/embeddings"


class APIEmbeddings(Embeddings):
    """Embeddings through HTTPS; no local model."""

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()

    def embed_documents(self, texts):
        if not texts:
            return []

        results = [None] * len(texts)
        batch_size = 64

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = self.session.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": batch,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()

            for item in payload["data"]:
                results[start + item["index"]] = item["embedding"]

        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def clean_text(text):
    """Conservative cleanup; preserve Tamil and English content."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^[\s|·•\-_=]{1,}$", "", text)
    return text.strip()


def extract_pdf(pdf_path):
    documents = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text", sort=True))

            # Avoid indexing empty/scanned pages as useful evidence.
            if len(text) < 40:
                print(
                    f"WARNING: {pdf_path.name}, page {page_number}: "
                    "very little text extracted; check whether OCR is needed."
                )
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    return documents


def main():
    if not API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    print(f"Found {len(pdf_files)} PDF files.")

    page_documents = []
    for pdf_file in pdf_files:
        print(f"Reading: {pdf_file.name}")
        page_documents.extend(extract_pdf(pdf_file))

    if not page_documents:
        raise RuntimeError(
            "No usable text extracted. Check PDF text or OCR requirements."
        )

    # Remove exact duplicate pages while retaining source metadata.
    unique_pages = []
    seen = set()

    for doc in page_documents:
        digest = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()

        if digest not in seen:
            seen.add(digest)
            unique_pages.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        length_function=len,
        separators=[
            "\n\n", "\n", "。", "!", "?", ".", " ", ""
        ],
    )

    chunks = splitter.split_documents(unique_pages)

    # Ignore extremely short, low-value fragments.
    chunks = [
        doc for doc in chunks
        if len(doc.page_content.strip()) >= 60
    ]

    if not chunks:
        raise RuntimeError("No usable chunks were created.")

    print(f"Extracted {len(unique_pages)} unique pages.")
    print(f"Created {len(chunks)} chunks.")
    print("Creating API embeddings...")

    embeddings = APIEmbeddings(API_KEY, EMBEDDING_MODEL)

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    vector_db.save_local(str(INDEX_DIR))

    # A manifest helps detect stale indexes when PDFs change.
    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "pdfs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in pdf_files
        },
        "page_count": len(unique_pages),
        "chunk_count": len(chunks),
    }

    import json
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"SUCCESS: FAISS index saved to {INDEX_DIR}")
    print("You can now start the Streamlit app.")


if __name__ == "__main__":
    main()
