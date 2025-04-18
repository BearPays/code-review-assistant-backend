import os
import json
import traceback

from fastapi import HTTPException

from src.schemas.chat_schemas import ChatRequest, SessionData
from src.core import session_manager, rag_utils

async def process_chat_request(request: ChatRequest) -> dict:
    """Orchestrates the chat logic for a request."""
    session_id = request.session_id
    pr_id = request.pr_id
    mode = request.mode
    user_query = request.query
    session: SessionData | None = None

    # --- Session Handling ---
    if session_id:
        session = session_manager.get_session(session_id)
        if session:
            # Ensure session matches request (optional, but good practice)
            if session.pr_id != pr_id or session.mode != mode:
                 raise HTTPException(
                     status_code=400, 
                     detail=f"Session ID {session_id} exists but with different pr_id or mode."
                 )
            print(f"Using existing session: {session_id}")
        else:
            # If session_id provided but not found, it's an error
            raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found.")

    if not session:
        # Create new session if no valid ID was provided or found
        session_id = session_manager.create_session_id()
        print(f"Creating new session: {session_id} for PR: {pr_id}, Mode: {mode}")
        try:
            # Load index specific to this PR
            # Note: load_project_index can raise HTTPException
            index_data = rag_utils.load_project_index(pr_id)
            session = SessionData(
                pr_id=pr_id,
                mode=mode,
                query_engines=index_data["query_engines"],
                collections=index_data["collections"]
            )
            session_manager.store_session(session_id, session)
        except HTTPException as e:
            # Propagate errors from index loading
            raise e
        except Exception as e:
             # Catch other potential errors during session creation
             print(f"Error creating session {session_id}: {e}")
             traceback.print_exc()
             raise HTTPException(status_code=500, detail=f"Failed to create session and load index for {pr_id}: {str(e)}")

    # --- API Key Check ---
    # This check should happen early, maybe even in middleware or router if preferred
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")

    # --- Determine the query to process ---
    # Use a predefined query for the first message in co_reviewer mode
    query_to_process = user_query
    is_initial_review = False
    if mode == "co_reviewer" and not session.initial_review_generated:
        print(f"Mode A (co_reviewer): Generating initial review for session {session_id}")
        initial_query = "Generate a comprehensive initial code review summary for this PR." 
        query_to_process = initial_query
        is_initial_review = True
        # Add the *effective* query to history immediately
        session.chat_history.append({"role": "user", "content": query_to_process})
        session_manager.set_initial_review_generated(session_id) # Mark before RAG
    else:
        # Only add the actual user query if it's not the initial review trigger
        session.chat_history.append({"role": "user", "content": user_query})


    # --- RAG Query Process ---
    try:
        # 1. Get the collection query plan
        plan = rag_utils.get_collection_plan(query_to_process, session.collections, session.pr_id)
        print(f"Session {session_id} - Query plan: {json.dumps(plan, indent=2)}")
        
        # 2. Query each relevant collection
        # TODO: Consider running queries in parallel using asyncio for performance
        responses = []
        for collection_name in plan["collections"]:
            response = rag_utils.query_collection(
                session.query_engines,
                collection_name,
                query_to_process,
                plan["search_focus"]
            )
            if response:
                responses.append(response)
        
        # 3. Synthesize the responses
        if not responses:
            # Handle case where RAG returns no useful info
            ai_response_content = "I couldn't retrieve specific information for your query from the available data sources."
            final_response_data = {
                 "answer": ai_response_content,
                 "sources": [],
                 "collections_used": plan["collections"] # Show which were attempted
            }
        else:
             # Synthesize, passing the *current* chat history (including the query processed)
             # Call the appropriate synthesis function based on the mode
             if session.mode == "co_reviewer":
                 final_response_data = rag_utils.synthesize_co_reviewer_response(
                     query_to_process, 
                     responses, 
                     session.chat_history, 
                     session.mode,
                     is_initial_review=is_initial_review
                 )
             elif session.mode == "interactive_assistant":
                 final_response_data = rag_utils.synthesize_interactive_response(
                     query_to_process, 
                     responses, 
                     session.chat_history, 
                     session.mode
                 )
             else:
                 # Should not happen due to request validation, but good to handle
                 print(f"ERROR: Unknown mode '{session.mode}' in service layer.")
                 raise HTTPException(status_code=500, detail="Internal server error: Unknown mode.")

        # --- Update Session History and Return --- 
        # Add AI response to history (managed by session_manager now)
        session_manager.update_session_history(session_id, query_to_process, final_response_data["answer"])
        
        # Return data needed for the ChatResponse model
        return {
            "session_id": session_id,
            "answer": final_response_data["answer"],
            "sources": final_response_data["sources"],
            "collections_used": final_response_data["collections_used"],
            "mode": session.mode,
            "pr_id": session.pr_id
        }
        
    except Exception as e:
        # Log error and return a generic error response
        print(f"Error processing chat for session {session_id}: {str(e)}")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}") 