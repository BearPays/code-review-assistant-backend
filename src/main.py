from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from src.routers import chat

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Code Review RAG API", description="API for Code Review AI Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(chat.router)

# Health check endpoint (optional, but good practice)
@app.get("/")
def read_root():
    return {"message": "Code Review RAG API is running."}


if __name__ == "__main__":
    # Make sure reload is False if you want sessions to persist across saves in dev
    # Set reload=True for development if you want the server to restart on code changes
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)