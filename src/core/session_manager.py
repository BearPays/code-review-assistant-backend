from typing import Dict, Optional
import uuid

from src.schemas.chat_schemas import SessionData

# In-memory storage for sessions
_sessions: Dict[str, SessionData] = {}

def get_session(session_id: str) -> Optional[SessionData]:
    """Retrieves a session by its ID."""
    return _sessions.get(session_id)

def create_session_id() -> str:
    """Generates a new unique session ID."""
    return str(uuid.uuid4())

def store_session(session_id: str, session_data: SessionData):
    """Stores session data."""
    _sessions[session_id] = session_data

def update_session_history(session_id: str, user_query: str, ai_answer: str):
    """Adds user query and AI answer to the session's chat history."""
    session = get_session(session_id)
    if session:
        session.chat_history.append({"role": "user", "content": user_query})
        session.chat_history.append({"role": "assistant", "content": ai_answer})
        # Optional: Limit history size
        MAX_HISTORY = 10
        if len(session.chat_history) > MAX_HISTORY * 2:
            session.chat_history = session.chat_history[-(MAX_HISTORY * 2):]
        store_session(session_id, session) # Re-store the updated session
    else:
        print(f"Warning: Attempted to update history for non-existent session: {session_id}")

def set_initial_review_generated(session_id: str):
    """Marks the initial review as generated for a session."""
    session = get_session(session_id)
    if session:
        session.initial_review_generated = True
        store_session(session_id, session) 