import os
from typing import List, Optional
from pathlib import Path
from llama_index.core import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_openai_api_key() -> bool:
    """Check if OpenAI API key is available in environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key) and api_key != "your_openai_api_key_here"

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load all text files from a directory into LlamaIndex Document objects.
    
    Args:
        directory_path: Path to the directory containing text files
        
    Returns:
        List of Document objects
    """
    documents = []
    dir_path = Path(directory_path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory_path}")
    
    for file_path in dir_path.glob("**/*.txt"):
        with open(file_path, "r") as f:
            text = f.read()
        
        # Create a Document with metadata
        doc = Document(
            text=text,
            metadata={
                "filename": file_path.name,
                "filepath": str(file_path),
                "file_size": file_path.stat().st_size,
                "created_at": file_path.stat().st_ctime,
                "modified_at": file_path.stat().st_mtime
            }
        )
        documents.append(doc)
    
    return documents 