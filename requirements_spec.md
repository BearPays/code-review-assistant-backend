
# Backend Spec

This document defines the backend-specific requirements for supporting the interactive AI code review system. The system is intended to analyse 2 different types of AI assistance during code review of a single Pull request (PR).

## Overview

The backend is responsible for managing session state, mode handling, RAG indexing/loading, and communication with the frontend and OpenAI API.

## Functional Requirements

### FR1: Core Chat Interface (Backend Scope)

- Accept chat messages from the frontend.
- Respond with AI-generated replies based on user query and context.
- Store and retrieve chat history for each session.

### FR2: LLM Integration

- Use the OpenAI GPT-4 API to generate completions.
- Use a consistent prompt schema that supports both code review and chat-style interactions.

### FR3: RAG Infrastructure Integration

- Integrate LlamaIndex to perform Retrieval-Augmented Generation (RAG).
- Support chain prompting workflows (multi-step reasoning).
- Support two RAG indexes:
    - `index_pr1`
    - `index_pr2`
- Each index must include:
    - `collection_pr`: Pull request data
    - `collection_code`: Source code diffs/files
    - `collection_reqs`: Feature/requirement descriptions

### FR4: Context Data Loading & Indexing

The RAG is indexed my the researchers through an indexing script before running the program.

### FR5: Mode A - Co-reviewer

- When a session is started in Mode A:
    - Automatically trigger generation of an initial review summary using RAG + GPT.
    - Send the summary to the frontend before the first user message.
- Support follow-up chat using the same context.

### FR6: Mode B - Interactive Assistant

- When a session is started in Mode B:
    - Wait for a user query before sending any AI-generated message.
    - Respond using GPT-4 + appropriate RAG context (based on PR ID).

### FR7: Mode Selection/Configuration

- Receive the interaction mode (”interactive_assistant" or “co_reviewer”) from the frontend.
- Maintain this mode per session.

### FR8: User PR Selection

- Accept PR selection from the frontend (`pr_id`).
- Use this to load the appropriate RAG index.

### FR9: Session Management

- Create a session object per chat that stores:
    - PR ID
    - Mode
    - Context (preloaded RAG sources)
    - Chat history
    - Initial summary state (whether generated or not)

### FR10: Basic Logging

- Store logs per session, including:
    - User query
    - AI response
    - Timestamps (optional)
    - PR ID and mode metadata

### API Queries

Flow: Code Review Mode (mode=co_reviewer)
	1.	Frontend makes first request →
{ query: "Start review", mode: "co_reviewer", pr_id: "project_1" }
	2.	Backend loads and indexes:
	•	PR data
	•	related code files
	•	related requirements
	3.	Backend generates:
	•	structured multi-aspect code review (to lead the developer in the review)
	•	stores it in session.review_generated = true
	4.	Backend manages session state and history
E.g. query = "What are the main security changes?" → looks into chat history + relevant source.

⸻

💬 Flow: Chat Mode (mode=interactive_assistant)
	1.	Every query can hit RAG (with routing to PR/Code/Req source as needed).
	2.	Frontend sends minimal data.
	3.	Backend manages session state + optionally injects system prompt with session context.