from typing import Dict, Any
from fastapi import HTTPException
import traceback
from src.agent.base import BaseAgent
from src.schemas.chat import ChatResponse
from src.agent.tools import AVAILABLE_TOOLS
from src.core.session_manager import SessionManager
from llama_index.llms.openai import OpenAI

class AgentService:
    def __init__(self):
        self.llm = OpenAI(model="gpt-4")
        self.agent = BaseAgent(self.llm)
        self.session_manager = SessionManager()
        
        # Register all available tools
        for tool in AVAILABLE_TOOLS:
            self.agent.register_tool(tool)
    
    async def process_request(self, request: str, session_id: str, pr_id: str, mode: str = "co_reviewer") -> ChatResponse:
        """Process a user request through the agent."""
        try:
            # Get or create session
            session = None
            if session_id:
                session = self.session_manager.get_session(session_id)
                
            if not session:
                # Create new session if needed
                session_id = self.session_manager.create_session_id()
                session = self.session_manager.create_session(session_id, pr_id, mode)
                
            # Handle initial review for co_reviewer mode
            is_initial_request = not session.initial_review_generated and session.mode == "co_reviewer"
            
            actual_request = request
            if is_initial_request:
                # For initial co_reviewer requests, use a standard prompt
                actual_request = "Generate a comprehensive initial code review summary for this PR."
                # Mark as generated
                self.session_manager.set_initial_review_generated(session_id)
            
            # Process request through agent
            response = await self.agent.process_request(
                request=actual_request,
                session_data={
                    "session_id": session_id,
                    "pr_id": pr_id,
                    "chat_history": session.chat_history,
                    "session": session,  # Pass the entire session object
                    "mode": mode,
                    "is_initial_request": is_initial_request
                }
            )
            
            return response
            
        except HTTPException as e:
            raise e
        except Exception as e:
            print(f"Error in process_request: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {str(e)}"
            ) 