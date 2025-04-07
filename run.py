#!/usr/bin/env python3
"""
Utility script to run the FastAPI application.
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or export it.")
        print("You may encounter errors when making queries.")
    
    # Run the FastAPI application
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True) 