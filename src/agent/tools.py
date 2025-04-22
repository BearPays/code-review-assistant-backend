from typing import Dict, Any, List
from src.agent.base import Tool
from src.core import rag_utils

class RAGSearchTool(Tool):
    """Tool for searching through RAG collections."""
    def __init__(self):
        super().__init__(
            name="rag_search",
            description="Search through RAG collections for relevant information about PRs, code, or requirements",
            parameters={
                "query": {"type": "string", "description": "The search query"},
                "collections": {"type": "array", "description": "List of collections to search"},
                "focus": {"type": "string", "description": "What to focus on in the search"}
            }
        )
    
    async def execute(self, query: str, collections: List[str], focus: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = session_data.get("session_id")
        session = session_data.get("session")
        mode = session_data.get("mode", "co_reviewer")
        
        if not session:
            raise ValueError(f"Session data not provided")
            
        # Enhance the query with mode context for better results
        enhanced_query = query
        if mode == "co_reviewer":
            if session_data.get("is_initial_request", False):
                enhanced_query = f"Comprehensive PR review focus - {query}"
                # For initial reviews, ensure we're adding detail to the focus
                if "general" in focus.lower():
                    focus = "PR structure, key changes, potential issues, and code quality aspects"
            else:
                enhanced_query = f"Detailed code review - {query}"
        else:  # interactive_assistant
            enhanced_query = f"Specific query - {query}"
                
        responses = []
        for collection in collections:
            if collection in session.collections:
                response = rag_utils.query_collection(
                    session.query_engines,
                    collection,
                    enhanced_query,
                    focus
                )
                if response:
                    responses.append(response)
                
        return {
            "responses": responses,
            "collections_searched": collections,
            "mode": mode
        }

class CollectionPlannerTool(Tool):
    """Tool for planning which collections to query."""
    def __init__(self):
        super().__init__(
            name="collection_planner",
            description="Determine which collections to query based on the user request",
            parameters={
                "query": {"type": "string", "description": "The search query"}
            }
        )
    
    async def execute(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        session = session_data.get("session")
        pr_id = session_data.get("pr_id")
        mode = session_data.get("mode", "co_reviewer")
        
        if not session or not pr_id:
            raise ValueError("Session data or PR ID not provided")
            
        # Add mode context to the query for better planning
        enhanced_query = query
        if mode == "co_reviewer":
            if session_data.get("is_initial_request", False):
                enhanced_query = f"Initial PR review: {query} - Need comprehensive overview."
            else:
                enhanced_query = f"Co-reviewer follow up: {query} - Focus on detailed code review aspects."
        else:  # interactive_assistant
            enhanced_query = f"Interactive assistance: {query} - Focus on specific answers."
            
        # Get collection plan
        plan = rag_utils.get_collection_plan(
            query=enhanced_query,
            available_collections=session.collections,
            pr_id=pr_id
        )
        
        return {
            "plan": plan,
            "collections": plan.get("collections", []),
            "mode": mode
        }

class PRSummaryTool(Tool):
    """Tool for generating PR summaries."""
    def __init__(self):
        super().__init__(
            name="pr_summary",
            description="Generate a summary of a pull request",
            parameters={
                "mode": {"type": "string", "description": "Summary mode (initial or follow-up)"}
            }
        )
    
    async def execute(self, mode: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        session = session_data.get("session")
        pr_id = session_data.get("pr_id")
        agent_mode = session_data.get("mode", "co_reviewer")
        is_initial_request = session_data.get("is_initial_request", False)
        
        if not session or not pr_id:
            raise ValueError("Session data or PR ID not provided")
            
        # Adjust query based on the mode
        summary_query = "Generate PR summary information"
        summary_focus = "PR metadata and description"
        
        if agent_mode == "co_reviewer":
            if is_initial_request or mode == "initial":
                summary_query = "Generate a comprehensive initial code review summary"
                summary_focus = "PR structure, key changes, and potential issues"
            else:
                summary_query = "Update PR summary with additional details"
                summary_focus = "PR context for follow-up questions"
        else:  # interactive_assistant
            summary_query = "Get key PR information for reference"
            summary_focus = "PR facts and context for specific questions"
            
        # Use existing RAG functionality to get PR info
        responses = []
        for collection in session.collections:
            response = rag_utils.query_collection(
                session.query_engines,
                collection,
                summary_query,
                summary_focus
            )
            if response:
                responses.append(response)
                
        # Generate summary using existing synthesis
        summary = rag_utils.synthesize_co_reviewer_response(
            summary_query,
            responses,
            session.chat_history,
            agent_mode,
            is_initial_review=(mode == "initial" or is_initial_request)
        )
        
        return {
            "summary": summary,
            "pr_id": pr_id,
            "mode": agent_mode
        }

class FileAnalysisTool(Tool):
    """Tool for analyzing specific files."""
    def __init__(self):
        super().__init__(
            name="file_analysis",
            description="Analyze a specific file in the PR",
            parameters={
                "file_path": {"type": "string", "description": "Path to the file"},
                "analysis_type": {"type": "string", "description": "Type of analysis to perform"}
            }
        )
    
    async def execute(self, file_path: str, analysis_type: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        session = session_data.get("session")
        mode = session_data.get("mode", "co_reviewer")
        
        if not session:
            raise ValueError("Session data not provided")
            
        # Adjust query based on mode
        query_prefix = ""
        if mode == "co_reviewer":
            query_prefix = "Review and analyze file in detail: "
            # If analysis type doesn't specify a focus, add mode-specific defaults
            if "general" in analysis_type.lower():
                if mode == "co_reviewer":
                    analysis_type = "code quality, best practices, potential bugs, and improvement suggestions"
        else:  # interactive_assistant
            query_prefix = "Find specific information about file: "
            
        # Search for the specific file
        responses = []
        for collection in session.collections:
            if collection.endswith("_source_code"):
                response = rag_utils.query_collection(
                    session.query_engines,
                    collection,
                    f"{query_prefix}{file_path}",
                    f"Focus on {analysis_type} aspects of the file"
                )
                if response:
                    responses.append(response)
                    
        return {
            "file_path": file_path,
            "analysis": responses,
            "analysis_type": analysis_type,
            "mode": mode
        }

class ResponseSynthesisTool(Tool):
    """Tool for synthesizing final responses from multiple sources."""
    def __init__(self):
        super().__init__(
            name="response_synthesis",
            description="Synthesize a final response based on multiple information sources",
            parameters={
                "query": {"type": "string", "description": "Original user query"},
                "responses": {"type": "array", "description": "List of RAG responses"}
            }
        )
    
    async def execute(self, query: str, responses: List[Dict], session_data: Dict[str, Any]) -> Dict[str, Any]:
        session = session_data.get("session")
        
        if not session:
            raise ValueError("Session data not provided")
            
        # Use different synthesis method based on session mode
        mode = session.mode
        is_initial_request = session_data.get("is_initial_request", False)
        
        if mode == "co_reviewer":
            result = rag_utils.synthesize_co_reviewer_response(
                query=query,
                responses=responses,
                chat_history=session.chat_history,
                mode=mode,
                is_initial_review=is_initial_request
            )
        else:  # interactive_assistant
            result = rag_utils.synthesize_interactive_response(
                query=query,
                responses=responses,
                chat_history=session.chat_history,
                mode=mode
            )
            
        return {
            "synthesized_answer": result.get("answer"),
            "sources": result.get("sources", []),
            "collections_used": result.get("collections_used", []),
            "mode": mode
        }

# Register all tools
AVAILABLE_TOOLS = [
    RAGSearchTool(),
    CollectionPlannerTool(),
    PRSummaryTool(),
    FileAnalysisTool(),
    ResponseSynthesisTool()
] 