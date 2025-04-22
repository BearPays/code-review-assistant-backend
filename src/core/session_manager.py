from typing import Dict, Optional
import uuid

from src.schemas.chat import SessionData
from src.core import rag_utils

class SessionManager:
    def __init__(self):
        # In-memory storage for sessions
        self._sessions: Dict[str, SessionData] = {}
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)
    
    def create_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())
    
    def create_session(self, session_id: str, pr_id: str, mode: str = "co_reviewer") -> SessionData:
        """Create a new session with the given parameters."""
        try:
            # Load the project index for this PR
            index_data = rag_utils.load_project_index(pr_id)
            
            # Create new session data
            session_data = SessionData(
                pr_id=pr_id,
                mode=mode,
                chat_history=[],
                initial_review_generated=False,
                query_engines=index_data["query_engines"],
                collections=index_data["collections"]
            )
            
            # Store the session
            self._sessions[session_id] = session_data
            
            return session_data
            
        except Exception as e:
            print(f"Error creating session: {e}")
            raise e
    
    def store_session(self, session_id: str, session_data: SessionData):
        """Store a session in memory."""
        self._sessions[session_id] = session_data
    
    def update_session_history(self, session_id: str, user_query: str, ai_answer: str):
        """Update the chat history for a session."""
        session = self.get_session(session_id)
        if session:
            # Add user message
            session.chat_history.append({
                "role": "user",
                "content": user_query
            })
            
            # Add AI message
            session.chat_history.append({
                "role": "assistant",
                "content": ai_answer
            })
            
            # Store updated session
            self.store_session(session_id, session)
            
    def set_initial_review_generated(self, session_id: str):
        """Mark that the initial review has been generated for a session."""
        session = self.get_session(session_id)
        if session:
            session.initial_review_generated = True
            self.store_session(session_id, session) 