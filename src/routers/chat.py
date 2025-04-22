from fastapi import APIRouter, HTTPException
from src.services.agent_service import AgentService
from src.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()
agent_service = AgentService()

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Main chat endpoint that processes requests through the agent service.
    Accepts ChatRequest and returns ChatResponse.
    """
    try:
        # Process the request through the agent service
        agent_response = await agent_service.process_request(
            request=request.query,
            session_id=request.session_id,
            pr_id=request.pr_id,
            mode=request.mode
        )
        
        # Map AgentResponse to ChatResponse format
        collections_used = []
        sources = []
        
        # Extract collections used and sources from metadata if available
        if agent_response.metadata:
            collections_used = agent_response.metadata.get("collections_used", [])
            sources = agent_response.metadata.get("sources", [])
            
        return ChatResponse(
            session_id=request.session_id or "new_session",
            answer=agent_response.answer,
            sources=sources,
            collections_used=collections_used,
            mode=request.mode,
            pr_id=request.pr_id,
            tools_used=agent_response.tools_used,
            metadata=agent_response.metadata
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in handle_chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        ) 