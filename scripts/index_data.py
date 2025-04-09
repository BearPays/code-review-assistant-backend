#!/usr/bin/env python3
"""
Script to index documents from data/ directory using LlamaIndex and ChromaDB.
This creates a persisted index that can be loaded by the FastAPI application.
"""

import os
import sys
import shutil
import traceback
from pathlib import Path
from dotenv import load_dotenv
from typing import List

import chromadb
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# === Constants and Configuration ===
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
COLLECTION_NAME = "rag_collection"

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".cs",
    ".php", ".rb", ".go", ".swift", ".kt", ".rs", ".scala", ".sh",
    ".html", ".css", ".json", ".yaml", ".yml"
}

TEXT_EXTENSIONS = {".txt", ".md"}

EXCLUDE_DIRS = {"node_modules", "__pycache__", "venv", ".git", ".idea", ".vscode", "dist", "build"}

DEFAULT_HF_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
USE_HF_EMBEDDING = os.getenv("USE_HF_EMBEDDING", "false").lower() == "true"
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"

SUPPORTED_CODE_LANGUAGES = {
    "c", "cpp", "csharp", "go", "html", "java", "javascript",
    "python", "ruby", "rust", "bash"
}

EXTENSION_TO_LANGUAGE = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "javascript",
    ".tsx": "typescript", ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
    ".cs": "csharp", ".php": "php", ".rb": "ruby", ".go": "go", ".swift": "swift",
    ".kt": "kotlin", ".rs": "rust", ".scala": "scala", ".sh": "bash",
    ".html": "html", ".css": "css", ".sql": "sql", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml"
}

# === Helper Functions ===
def validate_env():
    if not USE_HF_EMBEDDING:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_openai_api_key_here":
            print("\n❌ OPENAI_API_KEY is missing or set to placeholder.")
            sys.exit(1)

def configure_settings():
    if USE_HF_EMBEDDING:
        print("🔧 Using Hugging Face embedding model...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_HF_EMBEDDING_MODEL)
    else:
        print("🔧 Using OpenAI embedding model...")
        Settings.embed_model = OpenAIEmbedding(model=DEFAULT_OPENAI_EMBEDDING_MODEL)

def get_all_files(data_dir: Path) -> List[str]:
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if Path(file).suffix in CODE_EXTENSIONS or Path(file).suffix in TEXT_EXTENSIONS:
                file_paths.append(os.path.join(root, file))
    return file_paths

def load_documents(file_paths: List[str]):
    print(f"📥 Loading {len(file_paths)} files...")
    reader = SimpleDirectoryReader(input_files=file_paths, recursive=True, exclude_hidden=True)
    return reader.load_data()

def create_index(documents):
    if FORCE_REINDEX and INDEX_DIR.exists():
        print(f"🗑️  Removing old index at {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)

    print("⚙️  Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("📐 Splitting documents and creating index...")

    transformed_documents = []
    for doc in documents:
        file_name = doc.metadata.get('file_name', '')
        file_extension = Path(file_name).suffix
        file_path = doc.metadata.get('file_path', '')
        language = EXTENSION_TO_LANGUAGE.get(file_extension)

        use_code_splitter = language in SUPPORTED_CODE_LANGUAGES if language else False

        if use_code_splitter:
            try:
                splitter = CodeSplitter(language=language)
                chunks = splitter.split(doc.text)
            except Exception as e:
                print(f"⚠️  CodeSplitter failed for {file_name}, falling back to SentenceSplitter: {e}")
                splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
                chunks = splitter.split_text(doc.text)
        else:
            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
            chunks = splitter.split_text(doc.text)

        for i, chunk in enumerate(chunks):
            trimmed_file_path = str(Path(file_path).relative_to(DATA_DIR))
            transformed_documents.append(
                Document(text=chunk, metadata={"file_name": file_name, "file_path": trimmed_file_path, "chunk": i, "language": language})
            )

    index = VectorStoreIndex.from_documents(transformed_documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    print("✅ Indexing complete.")

# === Entry Point ===
def main():
    try:
        validate_env()
        configure_settings()
        file_paths = get_all_files(DATA_DIR)
        if not file_paths:
            print("⚠️  No files found to index.")
            return
        documents = load_documents(file_paths)
        create_index(documents)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
