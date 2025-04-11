from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="RAG API", description="API for Retrieval-Augmented Generation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# === Updated Constants ===
PROJECT_NAME = ""  # Specify the project name to load its index (subfolder in the indexes directory)

# === Updated Index Loading ===
try:
    if not PROJECT_NAME:
        raise ValueError("PROJECT_NAME is not set. Please specify a project to load.")

    # Get absolute path to the specific project's index directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    project_index_dir = os.path.join(project_root, "indexes", PROJECT_NAME)

    print(f"Loading index for project '{PROJECT_NAME}' from {project_index_dir}")

    # Connect to the Chroma collection
    chroma_client = chromadb.PersistentClient(path=project_index_dir)
    chroma_collection = chroma_client.get_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load the index using the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=project_index_dir)
    index = load_index_from_storage(storage_context)

    # Create retriever and query engine
    retriever = index.as_retriever(similarity_top_k=3)
    query_engine = index.as_query_engine(llm=OpenAI(model="gpt-4"))

    print(f"Index for project '{PROJECT_NAME}' loaded successfully!")
except Exception as e:
    print(f"Error loading index for project '{PROJECT_NAME}': {e}")
    print("Ensure you've run the indexing script first and specified a valid project name.")
    index = None
    query_engine = None

# Define request models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API. Use /query endpoint to query the system."}

@app.post("/query", response_model=QueryResponse)
def query_index(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=500, detail="Index not loaded. Run the indexing script first.")
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables or is set to the default value.")
    
    # Query the index
    try:
        response = query_engine.query(request.query)
        
        # Extract sources metadata
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                if hasattr(node, "metadata") and node.metadata:
                    sources.append(node.metadata)
                else:
                    sources.append({"text": node.text[:100] + "..."})
        
        return QueryResponse(
            answer=str(response),
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying index: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)