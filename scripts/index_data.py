#!/usr/bin/env python3
"""
Script to index documents from data/ directory using LlamaIndex and ChromaDB.
This creates a persisted index that can be loaded by the FastAPI application.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indexes")

# Code file extensions to include
CODE_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", 
    ".cs", ".php", ".rb", ".go", ".swift", ".kt", ".rs", ".scala", ".sh",
    ".html", ".css", ".sql", ".json", ".yaml", ".yml", ".md", ".txt"
]

# Directories to exclude
EXCLUDE_DIRS = ["node_modules", "__pycache__", "venv", ".git", ".idea", ".vscode", "dist", "build"]

def main():
    """Main function to index documents."""
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("Error: OPENAI_API_KEY not found in environment variables or is set to the default value.")
        print("Please set it in your .env file or export it.")
        sys.exit(1)

    print(f"Using OpenAI API key: {os.getenv('OPENAI_API_KEY')[:5]}...")
    
    # Configure LlamaIndex settings
    Settings.llm = OpenAI(model="gpt-4")
    
    # Create a text splitter for code and text files
    text_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200
    )
    
    print(f"Loading documents from {DATA_DIR}...")
    try:
        # Find all files in the data directory that match our criteria
        all_files = []
        for ext in CODE_EXTENSIONS:
            for file_path in Path(DATA_DIR).glob(f"**/*{ext}"):
                # Check if the file is in an excluded directory
                path_str = str(file_path)
                if not any(excluded_dir in path_str for excluded_dir in EXCLUDE_DIRS):
                    all_files.append(str(file_path))
        
        print(f"Found {len(all_files)} files matching criteria.")
        
        # Use SimpleDirectoryReader with explicit file paths
        documents = SimpleDirectoryReader(
            input_files=all_files,
            recursive=True,
            exclude_hidden=True
        ).load_data()
        
        print(f"Loaded {len(documents)} documents.")
        
        if not documents:
            print("No code files found. Please add code files to the data/ directory.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)
    
    print("Creating ChromaDB client and collection...")
    
    # Clean up old index directory if it exists
    if os.path.exists(INDEX_DIR):
        import shutil
        print(f"Removing existing index directory at {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)
    
    # Create the chroma client and collection
    chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
    chroma_collection = chroma_client.create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Creating index...")
    try:
        # Use the text splitter for chunking
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            transformations=[text_splitter]
        )
        
        # Persist index
        print(f"Persisting index to {INDEX_DIR}...")
        index.storage_context.persist(persist_dir=INDEX_DIR)
        
        print("Indexing completed successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 