from fastapi import APIRouter, HTTPException

from src.schemas.chat_schemas import ChatRequest, ChatResponse
from src.services import chat_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """Receives chat requests, processes them via the service, and returns the response."""
    try:
        # Delegate the core logic to the service layer
        response_data = await chat_service.process_chat_request(request)
        return ChatResponse(**response_data)
    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        # Catch any other unexpected errors from the service layer
        print(f"Unhandled error in chat endpoint: {e}")
        # Optionally log the full traceback here
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}") 