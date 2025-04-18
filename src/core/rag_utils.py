import os
import json
from typing import Dict, List, Optional

from fastapi import HTTPException
import chromadb
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# Function to load index for a specific project
def load_project_index(pr_id: str) -> Dict:
    """Loads the index and query engines for a given project ID."""
    query_engines_for_pr = {}
    collections_for_pr = []
    try:
        # Get absolute path to the specific project's index directory
        # Assumes this file is in src/core/
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_file_dir)
        project_root = os.path.dirname(src_dir)
        project_index_dir = os.path.join(project_root, "indexes", pr_id)

        if not os.path.isdir(project_index_dir):
            raise FileNotFoundError(f"Index directory not found for project: {pr_id} at {project_index_dir}")

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

# RAG Planning Function
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

# RAG Querying Function
def query_collection(session_query_engines: Dict, collection_name: str, query: str, focus: str) -> Optional[Dict]:
    """Query a specific collection using the session's query engine."""
    if collection_name not in session_query_engines:
        print(f"Error: Collection '{collection_name}' not found in session query engines.")
        return None
        
    try:
        # Add focus to the query for better context
        focused_query = f"""{query}

Focus on: {focus}"""
        
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

# RAG Synthesis Function for Co-Reviewer Mode
def synthesize_co_reviewer_response(query: str, responses: List[Dict], chat_history: List[Dict], mode: str, is_initial_review: bool) -> Dict:
    """Synthesize responses for co-reviewer mode (initial or follow-up)."""
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # Format the responses for the prompt before defining the system prompt
    formatted_responses = "\n\n".join([
        f"From {resp['collection']}:\n{resp['answer']}"
        for resp in responses if resp # Ensure response is not None
    ])
    
    system_prompt = "" # Initialize

    # --- Select Prompt Based on Initial vs Follow-up ---
    if is_initial_review:
        # Prompt for Initial Co-Reviewer Summary
        system_prompt = """You are a Code Review Assistant generating an initial review summary.
Your task is to create a structured code review summary based *only* on the provided context.

Available Information (Extracted from PR data, code, requirements):
{formatted_responses}

Instructions:
1.  **Extract key details** (like PR number, title, author, status) **directly from the 'Available Information' section.**
2.  Generate a structured multi-aspect code review summary using this Markdown format:
    ```markdown
    ## Initial Code Review Summary: PR {{pr_number}} - {{pr_title}}

    **Author:** {{extracted_author}}
    **Status:** {{extracted_status}}

    **1. Overview:**
    [Briefly summarize the PR's purpose based *only* on the description found in the available information.]

    **2. Key Changes:**
    [Summarize the main file changes and the nature of diffs based *only* on the provided 'Available Information'. Mention key added/modified files.]

    **3. Potential Areas for Focus:**
    [Based *only* on the provided info, suggest 1-2 general areas the user might want to look closer at, e.g., specific complex files, security aspects if mentioned, or major logic changes.]

    **4. Next Steps:**
    Please ask follow-up questions about specific files, logic, or concerns.
    ```
3.  **If the 'Available Information' seems insufficient to fill the template, state that clearly instead of refusing.** Example: "I have retrieved some initial information, but crucial details like [missing detail] were not found in the available context. Here's what I could gather: [Provide partial summary]." Fill the template fields with "[Data not available]" if specific data points are missing.

Initial Review Summary:"""

    else: # Co-reviewer follow-up
        system_prompt = """You are a Code Review Assistant in 'co_reviewer' mode, responding to a follow-up query.
Your task is to provide a concise, helpful answer based on the chat history and newly retrieved information.

Chat History:
{history_str}

Available New Information (Extracted for the latest query):
{formatted_responses}

User Query: {query}

Instructions:
1.  Analyze the User Query in the context of the Chat History.
2.  Use the Available New Information section to answer the query, supplementing with history context if needed.
3.  Provide a clear, conversational response directly addressing the query.

Assistant Response:"""

    llm = OpenAI(model="gpt-4")
    
    # Format the selected prompt correctly using the variables
    try:
        final_response = llm.complete(
            system_prompt.format(
                mode=mode, # Although mode is fixed to co_reviewer here, keep for consistency if prompt uses it
                history_str=history_str if chat_history else "No history yet.",
                formatted_responses=formatted_responses if responses else "No new information gathered.",
                query=query
            )
        )
    except KeyError as e:
        print(f"Error formatting co_reviewer prompt: Missing key {e}")
        print(f"Selected prompt template:\n{system_prompt}")
        final_response = "Sorry, there was an internal error formatting the response."
    except Exception as e:
        print(f"Error during LLM completion (co_reviewer): {e}")
        final_response = "Sorry, there was an error generating the response."
    
    all_sources = []
    collections_used = []
    for resp in responses:
        if resp:
            all_sources.extend(resp.get("sources", []))
            collections_used.append(resp.get("collection"))

    return {
        "answer": str(final_response),
        "sources": all_sources,
        "collections_used": list(set(filter(None, collections_used)))
    }

# RAG Synthesis Function for Interactive Assistant Mode
def synthesize_interactive_response(query: str, responses: List[Dict], chat_history: List[Dict], mode: str) -> Dict:
    """Synthesize responses for interactive assistant mode."""
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    formatted_responses = "\n\n".join([
        f"From {resp['collection']}:\n{resp['answer']}"
        for resp in responses if resp
    ])
    
    system_prompt = """You are an Interactive Code Assistant.
Your task is to provide helpful answers to the user's query based on the chat history and newly retrieved information.

Chat History:
{history_str}

Available New Information (Extracted for the latest query):
{formatted_responses}

User Query: {query}

Instructions:
1.  Analyze the User Query in the context of the Chat History.
2.  Use the Available New Information section and the history to formulate your answer.
3.  Provide a clear, conversational, and helpful response.

Assistant Response:"""

    llm = OpenAI(model="gpt-4")
    
    try:
        final_response = llm.complete(
            system_prompt.format(
                mode=mode, # Should be interactive_assistant
                history_str=history_str if chat_history else "No history yet.",
                formatted_responses=formatted_responses if responses else "No new information gathered.",
                query=query
            )
        )
    except KeyError as e:
        print(f"Error formatting interactive prompt: Missing key {e}")
        print(f"Selected prompt template:\n{system_prompt}")
        final_response = "Sorry, there was an internal error formatting the response."
    except Exception as e:
        print(f"Error during LLM completion (interactive): {e}")
        final_response = "Sorry, there was an error generating the response."

    all_sources = []
    collections_used = []
    for resp in responses:
        if resp:
            all_sources.extend(resp.get("sources", []))
            collections_used.append(resp.get("collection"))

    return {
        "answer": str(final_response),
        "sources": all_sources,
        "collections_used": list(set(filter(None, collections_used)))
    } 