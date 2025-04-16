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
from typing import Dict, List, Optional, Literal
import json
import uuid

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

# === In-Memory Session Storage ===
sessions: Dict[str, Dict] = {}

# === Session Data Model ===
class SessionData(BaseModel):
    pr_id: str
    mode: Literal["co_reviewer", "interactive_assistant"]
    chat_history: List[Dict] = []
    initial_review_generated: bool = False
    query_engines: Dict = {}
    collections: List[str] = []

# Function to load index for a specific project
def load_project_index(pr_id: str) -> Dict:
    """Loads the index and query engines for a given project ID."""
    query_engines_for_pr = {}
    collections_for_pr = []
    try:
        # Get absolute path to the specific project's index directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        project_index_dir = os.path.join(project_root, "indexes", pr_id)

        if not os.path.isdir(project_index_dir):
            raise FileNotFoundError(f"Index directory not found for project: {pr_id}")

        print(f"Loading index for project '{pr_id}' from {project_index_dir}")

        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path=project_index_dir)
        
        # Get all collections for this project
        db_collections = chroma_client.list_collections()
        print(f"Found collections in DB: {[col.name for col in db_collections]}")

        # Create query engines for each collection
        for collection in db_collections:
            collection_name = collection.name
            collections_for_pr.append(collection_name)
            print(f"Loading collection '{collection_name}'...")
            
            try:
                # Get the collection
                chroma_collection = chroma_client.get_collection(collection_name)
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                
                # Derive storage dir name based on common pattern or specific logic if needed
                # Assuming storage dir is named like 'storage_suffix' where collection is 'project_id_suffix'
                suffix = collection_name.replace(pr_id + '_', '')
                storage_dir = os.path.join(project_index_dir, f"storage_{suffix}")
                
                # Check if storage directory exists and has required files
                if not os.path.exists(storage_dir) or not os.path.exists(os.path.join(storage_dir, "docstore.json")):
                    print(f"⚠️  Storage files missing for collection '{collection_name}' at {storage_dir}, skipping...")
                    continue
                    
                storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=storage_dir)
                index = load_index_from_storage(storage_context)
                retriever = index.as_retriever(similarity_top_k=3) # Default top_k
                query_engine = index.as_query_engine(llm=OpenAI(model="gpt-4"))
                
                query_engines_for_pr[collection_name] = {
                    "engine": query_engine,
                    "retriever": retriever
                }
                print(f"✅ Collection '{collection_name}' loaded successfully!")
            except Exception as e:
                print(f"⚠️  Error loading collection '{collection_name}': {e}")
                continue

        if not query_engines_for_pr:
            raise ValueError(f"No collections were successfully loaded for project {pr_id}.")
        else:
            print(f"✅ Successfully loaded {len(query_engines_for_pr)} collections for {pr_id}: {list(query_engines_for_pr.keys())}")
        
        return {"query_engines": query_engines_for_pr, "collections": collections_for_pr}

    except Exception as e:
        print(f"❌ Error initializing project '{pr_id}': {e}")
        # Re-raise the exception to be caught by the endpoint
        raise HTTPException(status_code=500, detail=f"Error loading index for project {pr_id}: {str(e)}")

def get_collection_plan(query: str, available_collections: List[str], pr_id: str) -> Dict:
    """Have the LLM analyze the query and determine which collections to query."""
    # Note: Using available_collections passed from the session
    
    system_prompt = f"""You are a Code Review Assistant with access to these specific collections for PR '{pr_id}':
{', '.join(available_collections)}

You are working with structured pull request (PR) data. Assume the collections contain relevant PR data, code diffs, and requirements based on their names (e.g., {pr_id}_pr_data, {pr_id}_code, {pr_id}_requirements).

Each indexed PR might contain fields like:
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
IMPORTANT: You MUST ONLY use the exact collection names listed in `available_collections`.

Analyze the user's query and determine:
1. Which collections from the available ones are relevant
2. In what order they should be queried (optional, can be parallel)
3. What specific aspects to look for in each collection

For PR-related queries:
- When asked for files changed in the PR, look for the `files` list and extract `filename` fields (likely in a code or PR data collection).
- For PR summaries, use `title`, `description`, and key changes from `files` (likely in a PR data collection).
- For specific file diffs, locate the `files` item with matching `filename` and return the `diff` (likely in a code collection).

Return your analysis as a JSON object with:
- collections: List of collections to query (MUST match exact names from the available collections)
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
        valid_collections = []
        for collection in plan.get("collections", []):
            if collection in available_collections:
                valid_collections.append(collection)
            else:
                print(f"⚠️  LLM suggested invalid collection '{collection}', removing it")
        
        if not valid_collections:
            # If no valid collections, use all available ones
            plan["collections"] = available_collections
            plan["reasoning"] = "Using all available collections as no specific ones were determined or validated"
            plan["search_focus"] = "General information"
        else:
            plan["collections"] = valid_collections
            
        return plan
    except json.JSONDecodeError:
        # If not valid JSON, create a default plan using all available collections
        return {
            "collections": available_collections,
            "reasoning": "Could not parse LLM response for collection plan, using all available collections",
            "search_focus": "General information"
        }

def query_collection(session_query_engines: Dict, collection_name: str, query: str, focus: str) -> Optional[Dict]:
    """Query a specific collection using the session's query engine."""
    if collection_name not in session_query_engines:
        print(f"Error: Collection '{collection_name}' not found in session query engines.")
        return None
        
    try:
        # Add focus to the query for better context
        focused_query = f"{query}\n\nFocus on: {focus}"
        
        # TODO: Consider making top_k dynamic or configurable if needed
        similarity_top_k = 5 # Increased default
        
        query_engine_info = session_query_engines[collection_name]
        query_engine = query_engine_info["engine"]
        retriever = query_engine_info["retriever"]
        retriever.similarity_top_k = similarity_top_k
        
        print(f"Querying collection '{collection_name}' with top_k={similarity_top_k}")
        response = query_engine.query(focused_query)
        
        # Extract sources metadata
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                metadata = getattr(node, "metadata", {})
                text_preview = getattr(node, "text", "")[:100] + "..."
                source_info = {"text_preview": text_preview}
                if metadata:
                    source_info.update(metadata) # Add metadata like filename if available
                sources.append(source_info)

        return {
            "answer": str(response),
            "sources": sources,
            "collection": collection_name
        }
    except Exception as e:
        print(f"Error querying collection '{collection_name}': {e}")
        return None

def synthesize_responses(query: str, responses: List[Dict], chat_history: List[Dict], mode: str) -> Dict:
    """Synthesize responses, potentially incorporating chat history."""
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # Format the responses for the prompt before defining the system prompt
    formatted_responses = "\n\n".join([
        f"From {resp['collection']}:\n{resp['answer']}"
        for resp in responses if resp # Ensure response is not None
    ])
    
    # Define system_prompt as a regular string with placeholders
    system_prompt = """You are a Code Review Assistant in '{mode}' mode. 
Your task is to synthesize this information into a coherent, comprehensive answer based on the provided chat history and new information.

Chat History:
{history_str}

Available New Information:
{formatted_responses}

User Query: {query}

Provide a well-structured answer that:
1. Directly addresses the user's query, considering the chat history for context.
2. Incorporates relevant information from all new sources gathered for this query.
3. If in 'co_reviewer' mode and this is the initial review (history might be short or just the initial query), provide a structured multi-aspect code review.
4. If in 'interactive_assistant' mode or a follow-up in 'co_reviewer', provide a conversational response based on the history and new info.
5. Highlights any discrepancies or important findings.
6. Provides clear, actionable insights where applicable.
Synthesized Answer:""" # Added a label for the expected output
    
    llm = OpenAI(model="gpt-4")
    
    # Format the prompt correctly using the variables
    final_response = llm.complete(
        system_prompt.format(
            mode=mode,
            history_str=history_str if chat_history else "No history yet.",
            formatted_responses=formatted_responses if responses else "No new information gathered.",
            query=query
        )
    )
    
    all_sources = []
    collections_used = []
    for resp in responses:
        if resp:
            all_sources.extend(resp.get("sources", []))
            collections_used.append(resp.get("collection"))

    return {
        "answer": str(final_response),
        "sources": all_sources,
        "collections_used": list(set(filter(None, collections_used))) # Unique, non-None collection names
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
class ChatRequest(BaseModel):
    query: str
    pr_id: str
    mode: Literal["co_reviewer", "interactive_assistant"]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list
    collections_used: list
    mode: str
    pr_id: str

# === Health Check Endpoint ===
@app.get("/")
def read_root():
    return {"message": "Code Review RAG API is running. Use the /chat endpoint."}

# === Main Chat Endpoint ===
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    pr_id = request.pr_id
    mode = request.mode
    user_query = request.query

    # --- Session Handling ---
    if session_id and session_id in sessions:
        session = sessions[session_id]
        # Ensure session matches request (optional, but good practice)
        if session.pr_id != pr_id or session.mode != mode:
             raise HTTPException(status_code=400, detail=f"Session ID {session_id} exists but with different pr_id or mode.")
        print(f"Using existing session: {session_id}")
    else:
        # Create new session
        session_id = str(uuid.uuid4())
        print(f"Creating new session: {session_id} for PR: {pr_id}, Mode: {mode}")
        try:
            # Load index specific to this PR
            index_data = load_project_index(pr_id)
            session = SessionData(
                pr_id=pr_id,
                mode=mode,
                query_engines=index_data["query_engines"],
                collections=index_data["collections"]
            )
            sessions[session_id] = session
        except HTTPException as e:
            # Propagate errors from index loading
            raise e
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to create session and load index for {pr_id}: {str(e)}")

    # --- API Key Check ---
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")

    # --- Mode-Specific Logic ---
    is_first_message = not session.chat_history
    
    # Special handling for the first message in co_reviewer mode
    if mode == "co_reviewer" and not session.initial_review_generated:
        print(f"Mode A (co_reviewer): Generating initial review for session {session_id}")
        # Use a predefined query or the user's first query to generate the initial summary
        initial_query = "Generate a comprehensive initial code review summary for this PR." # Or use request.query
        session.chat_history.append({"role": "user", "content": initial_query}) # Log the effective query
        current_query = initial_query
        session.initial_review_generated = True # Mark as generated
    else:
         # Add user query to history
        session.chat_history.append({"role": "user", "content": user_query})
        current_query = user_query


    # --- RAG Query Process ---
    try:
        # Get the collection query plan
        plan = get_collection_plan(current_query, session.collections, session.pr_id)
        print(f"Session {session_id} - Query plan: {json.dumps(plan, indent=2)}")
        
        # Query each relevant collection
        responses = []
        for collection_name in plan["collections"]:
            response = query_collection(
                session.query_engines,
                collection_name,
                current_query,
                plan["search_focus"]
            )
            if response:
                responses.append(response)
        
        if not responses:
            # Handle case where no collections provided useful info
            # Maybe generate a response indicating no info found or ask clarifying questions
            ai_response_content = "I couldn't retrieve specific information for your query from the available data sources."
            final_response_data = {
                 "answer": ai_response_content,
                 "sources": [],
                 "collections_used": plan["collections"] # Show which were attempted
            }
        else:
             # Synthesize the responses into a coherent answer, considering history
            final_response_data = synthesize_responses(current_query, responses, session.chat_history, session.mode)

        # Add AI response to history
        session.chat_history.append({"role": "assistant", "content": final_response_data["answer"]})
        
        # Keep history length manageable (e.g., last 10 messages) - Optional
        MAX_HISTORY = 10
        if len(session.chat_history) > MAX_HISTORY * 2: # User+Assistant pairs
             session.chat_history = session.chat_history[-(MAX_HISTORY*2):]

        return ChatResponse(
            session_id=session_id,
            answer=final_response_data["answer"],
            sources=final_response_data["sources"],
            collections_used=final_response_data["collections_used"],
            mode=session.mode,
            pr_id=session.pr_id
        )
        
    except Exception as e:
        # Log error for debugging
        import traceback
        print(f"Error processing chat for session {session_id}: {str(e)}")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Make sure reload is False if you want sessions to persist across saves in dev
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)