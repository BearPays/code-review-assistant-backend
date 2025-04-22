from pydantic import BaseModel
from typing import Dict, List, Optional, Literal, Any

# === Session Data Model ===
class SessionData(BaseModel):
    pr_id: str
    mode: Literal["co_reviewer", "interactive_assistant"]
    chat_history: List[Dict] = []
    initial_review_generated: bool = False
    query_engines: Dict = {}
    collections: List[str] = []

# === API Request/Response Models ===
class ChatRequest(BaseModel):
    query: str
    pr_id: str
    mode: Literal["co_reviewer", "interactive_assistant"]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict]
    collections_used: List[str]
    mode: str
    pr_id: str
    tools_used: List[str] = []
    metadata: Dict[str, Any] = {} 