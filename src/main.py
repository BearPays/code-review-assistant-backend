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
from typing import Dict, List, Optional
import json

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
PROJECT_NAME = "project_2"  # Specify the project name to load its index (subfolder in the indexes directory)

# === Updated Index Loading ===
query_engines: Dict[str, Dict] = {}  # Store query engines for each collection

def get_collection_plan(query: str) -> Dict:
    """Have the LLM analyze the query and determine which collections to query and in what order."""
    available_collections = list(query_engines.keys())
    
    system_prompt = f"""You are a Code Review Assistant with access to these specific collections:
{', '.join(available_collections)}

You are working with structured pull request (PR) data stored in a RAG index named `project_2_pr_data`.

Each indexed PR contains fields like:
- "pr_number" (int): The pull request number
- "title" (string): Title of the PR
- "description" (string): Detailed PR description including references and rationale
- "state" (string): Open/closed status
- "created_at" / "updated_at" (timestamp)
- "author" (string): Username of the PR author
- "files" (list of dicts): **This contains all the changed files and diffs**:
  - "filename" (string): File path
  - "status" (string): Type of change (`modified`, `added`, `removed`)
  - "additions" (int): Number of lines added
  - "deletions" (int): Number of lines removed
  - "diff" (string): Full unified diff format (git-style)

Your task is to analyze the user's query and determine which collections to query.
IMPORTANT: You MUST ONLY use the exact collection names listed above.

Analyze the user's query and determine:
1. Which collections from the available ones are relevant
2. In what order they should be queried
3. What specific aspects to look for in each collection

For PR-related queries:
- When asked for files changed in the PR, look for the `files` list and extract `filename` fields
- For PR summaries, use `title`, `description`, and key changes from `files`
- For specific file diffs, locate the `files` item with matching `filename` and return the `diff`

Return your analysis as a JSON object with:
- collections: List of collections to query in order (MUST match exact names from the available collections)
- reasoning: Brief explanation of why each collection is needed
- search_focus: What to look for in each collection, including specific fields to examine
"""
    
    llm = OpenAI(model="gpt-4")
    response = llm.complete(
        system_prompt + f"\n\nUser Query: {query}\n\nAnalysis:"
    )
    
    try:
        # Try to parse the response as JSON
        plan = json.loads(str(response))
        
        # Validate that the collections exist
        for collection in plan.get("collections", []):
            if collection not in query_engines:
                print(f"⚠️  LLM suggested invalid collection '{collection}', removing it")
                plan["collections"].remove(collection)
        
        if not plan.get("collections"):
            # If no valid collections, use all available ones
            plan["collections"] = available_collections
            plan["reasoning"] = "Using all available collections as no specific ones were determined"
            plan["search_focus"] = "General information"
            
        return plan
    except json.JSONDecodeError:
        # If not valid JSON, create a default plan
        return {
            "collections": available_collections,
            "reasoning": "Could not parse LLM response, using all collections",
            "search_focus": "General information"
        }

def query_collection(collection_name: str, query: str, focus: str) -> Dict:
    """Query a specific collection with focused context."""
    if collection_name not in query_engines:
        return None
        
    try:
        # Add focus to the query for better context
        focused_query = f"{query}\n\nFocus on: {focus}"
        
        # Let LLM determine the appropriate similarity_top_k
        llm = OpenAI(model="gpt-4")
        top_k_prompt = f"""Analyze this query and determine how many relevant chunks to retrieve:
Query: {focused_query}

Consider:
1. How specific or broad the query is
2. How many different aspects need to be covered
3. Whether it's asking for a list or comprehensive information

Return a JSON object with:
- similarity_top_k: number between 3 and 20
- reasoning: brief explanation of your choice
"""
        top_k_response = llm.complete(top_k_prompt)
        try:
            top_k_decision = json.loads(str(top_k_response))
            similarity_top_k = min(max(int(top_k_decision.get("similarity_top_k", 3)), 3), 20)
            print(f"Using similarity_top_k={similarity_top_k} for query: {focused_query}")
        except:
            similarity_top_k = 3  # Default if parsing fails
        
        query_engine = query_engines[collection_name]["engine"]
        retriever = query_engines[collection_name]["retriever"]
        retriever.similarity_top_k = similarity_top_k
        
        # First attempt at querying
        response = query_engine.query(focused_query)
        
        # Have LLM verify if it has enough information
        verification_prompt = f"""You have retrieved {similarity_top_k} chunks and generated this response:
{str(response)}

Original Query: {focused_query}

Analyze if this response:
1. Fully answers the query
2. Contains all necessary information
3. Has any gaps or missing details

Return a JSON object with:
- has_enough_info: boolean
- reasoning: explanation of your assessment
- missing_info: list of any missing information or gaps
"""
        verification_response = llm.complete(verification_prompt)
        try:
            verification = json.loads(str(verification_response))
            if not verification.get("has_enough_info", True):
                # If not enough info, try with more chunks
                new_top_k = min(similarity_top_k * 2, 20)
                if new_top_k > similarity_top_k:
                    print(f"Increasing similarity_top_k to {new_top_k} to get more information")
                    retriever.similarity_top_k = new_top_k
                    response = query_engine.query(focused_query)
        except:
            pass  # Continue with original response if verification fails
        
        # Extract sources metadata
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                if hasattr(node, "metadata") and node.metadata:
                    sources.append(node.metadata)
                else:
                    sources.append({"text": node.text[:100] + "..."})
        
        return {
            "answer": str(response),
            "sources": sources,
            "collection": collection_name
        }
    except Exception as e:
        print(f"Error querying collection '{collection_name}': {e}")
        return None

def synthesize_responses(query: str, responses: List[Dict]) -> Dict:
    """Have the LLM synthesize responses from multiple collections into a coherent answer."""
    system_prompt = """You are a Code Review Assistant. You have gathered information from multiple sources about a code review query.
Your task is to synthesize this information into a coherent, comprehensive answer.

Available information:
{responses}

User Query: {query}

Provide a well-structured answer that:
1. Directly addresses the user's query
2. Incorporates relevant information from all sources
3. Highlights any discrepancies or important findings
4. Provides clear, actionable insights
"""
    
    # Format the responses for the prompt
    formatted_responses = "\n\n".join([
        f"From {resp['collection']}:\n{resp['answer']}"
        for resp in responses
    ])
    
    llm = OpenAI(model="gpt-4")
    response = llm.complete(
        system_prompt.format(
            responses=formatted_responses,
            query=query
        )
    )
    
    return {
        "answer": str(response),
        "sources": [source for resp in responses for source in resp["sources"]],
        "collections_used": [resp["collection"] for resp in responses]
    }

try:
    if not PROJECT_NAME:
        raise ValueError("PROJECT_NAME is not set. Please specify a project to load.")

    # Get absolute path to the specific project's index directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    project_index_dir = os.path.join(project_root, "indexes", PROJECT_NAME)

    print(f"Loading index for project '{PROJECT_NAME}' from {project_index_dir}")

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=project_index_dir)
    
    # Get all collections for this project
    collections = chroma_client.list_collections()
    print(f"Found collections: {[col.name for col in collections]}")

    # Create query engines for each collection
    for collection in collections:
        collection_name = collection.name
        print(f"Loading collection '{collection_name}'...")
        
        try:
            # Get the collection
            chroma_collection = chroma_client.get_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Load the index using the vector store - Fixed path to match actual directory structure
            storage_dir = os.path.join(project_index_dir, f"storage_{collection_name.replace(PROJECT_NAME + '_', '')}")
            
            # Check if storage directory exists and has required files
            if not os.path.exists(storage_dir) or not os.path.exists(os.path.join(storage_dir, "docstore.json")):
                print(f"⚠️  Storage files missing for collection '{collection_name}', skipping...")
                continue
                
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            retriever = index.as_retriever(similarity_top_k=3)
            query_engine = index.as_query_engine(llm=OpenAI(model="gpt-4"))
            
            query_engines[collection_name] = {
                "engine": query_engine,
                "retriever": retriever
            }
            print(f"✅ Collection '{collection_name}' loaded successfully!")
        except Exception as e:
            print(f"⚠️  Error loading collection '{collection_name}': {e}")
            continue

    if not query_engines:
        raise ValueError("No collections were successfully loaded.")
    else:
        print(f"\n✅ Successfully loaded {len(query_engines)} collections: {list(query_engines.keys())}")

except Exception as e:
    print(f"❌ Error initializing project '{PROJECT_NAME}': {e}")
    print("Ensure you've run the indexing script first and specified a valid project name.")
    query_engines = {}

# Define request models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    collections_used: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API. Use /query endpoint to query the system."}

@app.post("/query", response_model=QueryResponse)
def query_index(request: QueryRequest):
    if not query_engines:
        raise HTTPException(status_code=500, detail="No collections loaded. Run the indexing script first.")
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables or is set to the default value.")
    
    try:
        # Get the collection query plan from the LLM
        plan = get_collection_plan(request.query)
        print(f"Query plan: {json.dumps(plan, indent=2)}")
        
        # Query each collection according to the plan
        responses = []
        for collection_name in plan["collections"]:
            response = query_collection(
                collection_name,
                request.query,
                plan["search_focus"]
            )
            if response:
                responses.append(response)
        
        if not responses:
            raise HTTPException(status_code=500, detail="No valid responses from any collection")
        
        # Synthesize the responses into a coherent answer
        final_response = synthesize_responses(request.query, responses)
        return QueryResponse(**final_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)