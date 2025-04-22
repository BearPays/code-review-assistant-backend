from typing import List, Dict, Any, Literal
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from src.schemas.chat import ChatResponse
import json
import re

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with the given parameters.
        This method should be implemented by subclasses."""
        raise NotImplementedError("Tool execution not implemented")

class BaseAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.tools: List[Tool] = []
        self.system_prompts = {
            "co_reviewer": self._get_co_reviewer_prompt(),
            "interactive_assistant": self._get_interactive_assistant_prompt()
        }
        
    def _get_co_reviewer_prompt(self) -> str:
        """Get the system prompt for co-reviewer mode."""
        return """You are an intelligent Code Review Assistant Agent in CO-REVIEWER mode. Your task is to help with code reviews by:
1. Understanding the user's request in the context of a specific Pull Request
2. Selecting appropriate tools to gather information about the PR
3. Synthesizing responses that provide insightful code review feedback

In CO-REVIEWER mode:
- You are expected to be proactive about suggesting improvements
- You should focus on code quality, best practices, and potential issues
- Your responses should be structured and thorough
- First messages should provide a comprehensive initial review

Available Tools:
{tools_list}

When responding:
1. First analyze the PR and the user's request
2. Determine which tools would be helpful
3. Use the tools to gather specific information about the code
4. Synthesize a helpful, detailed response with concrete suggestions

Always explain your reasoning and be transparent about what information you're using.

Tool Selection Format:
If you need to use tools, respond in this JSON format:
{{
  "reasoning": "Your reasoning for tool selection",
  "tools": [
    {{
      "name": "tool_name",
      "parameters": {{
        "param1": "value1",
        "param2": "value2"
      }}
    }}
  ]
}}
"""

    def _get_interactive_assistant_prompt(self) -> str:
        """Get the system prompt for interactive assistant mode."""
        return """You are an intelligent Code Review Assistant Agent in INTERACTIVE ASSISTANT mode. Your task is to help with code reviews by:
1. Answering the user's specific questions about code
2. Selecting appropriate tools to gather relevant information
3. Synthesizing concise, focused responses that directly address queries

In INTERACTIVE ASSISTANT mode:
- You respond to user queries rather than proactively reviewing code
- You provide targeted, specific answers to questions
- Your responses should be concise and focused on the exact question asked
- You only provide information that was explicitly requested

Available Tools:
{tools_list}

When responding:
1. First analyze the specific query
2. Determine which tools would be helpful to answer it
3. Use the tools to gather precisely the information needed
4. Synthesize a direct, focused response that answers only what was asked

Always explain your reasoning and be transparent about what information you're using.

Tool Selection Format:
If you need to use tools, respond in this JSON format:
{{
  "reasoning": "Your reasoning for tool selection",
  "tools": [
    {{
      "name": "tool_name",
      "parameters": {{
        "param1": "value1",
        "param2": "value2"
      }}
    }}
  ]
}}
"""

    def register_tool(self, tool: Tool):
        """Register a new tool with the agent."""
        self.tools.append(tool)
        
    def _format_tools_prompt(self, mode: str) -> str:
        """Format the tools list for the system prompt."""
        tools_formatted = []
        
        for tool in self.tools:
            # Format parameters
            params_formatted = []
            for param_name, param_details in tool.parameters.items():
                param_type = param_details.get("type", "any")
                param_desc = param_details.get("description", "")
                params_formatted.append(f"  - {param_name} ({param_type}): {param_desc}")
                
            # Add the formatted tool
            tools_formatted.append(
                f"- {tool.name}: {tool.description}\n"
                f"  Parameters:\n"
                f"{chr(10).join(params_formatted)}"
            )
            
        system_prompt = self.system_prompts.get(mode, self.system_prompts["co_reviewer"])
        return system_prompt.format(tools_list="\n\n".join(tools_formatted))
        
    async def process_request(self, request: str, session_data: Dict[str, Any]) -> ChatResponse:
        """Process a user request using the agent."""
        # Get mode from session data, default to co_reviewer
        mode = session_data.get("mode", "co_reviewer")
        is_initial_request = session_data.get("is_initial_request", False)
        
        # Format the prompt with tools and request
        prompt = f"""User Request: {request}

Current Session Data:
- PR ID: {session_data.get('pr_id', 'Unknown')}
- Session ID: {session_data.get('session_id', 'Unknown')}
- Chat History: {len(session_data.get('chat_history', [])) // 2} exchanges
- Mode: {mode}
- Is Initial Request: {is_initial_request}

Available Tools:
{self._format_tools_prompt(mode)}

Please analyze the request and determine which tools to use. Explain your reasoning.
Respond in the JSON format specified if you need to use tools."""

        # Get initial analysis from LLM
        analysis = self.llm.complete(prompt)
        tools_used = []
        
        try:
            # Parse the analysis to see if it contains tool selections
            tool_selections = self._extract_json(str(analysis))
            
            if tool_selections and "tools" in tool_selections:
                # Execute the selected tools
                tools_results = []
                for tool_selection in tool_selections["tools"]:
                    tool_name = tool_selection.get("name")
                    tool_params = tool_selection.get("parameters", {})
                    
                    # Find the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool:
                        tools_used.append(tool_name)
                        # Execute the tool
                        result = await tool.execute(**tool_params, session_data=session_data)
                        tools_results.append({
                            "tool": tool_name,
                            "result": result
                        })
                
                # Synthesize final response
                if tools_results:
                    final_response = await self._synthesize_response(request, tools_results, session_data)
                    return final_response
        except Exception as e:
            print(f"Error executing tools: {e}")
            # Continue with default response if tool execution fails
        
        # Default response if no tools were used or execution failed
        return ChatResponse(
            session_id=session_data.get("session_id", "new_session"),
            answer=str(analysis),
            sources=[],
            collections_used=[],
            mode=mode,
            pr_id=session_data.get("pr_id", "unknown"),
            tools_used=tools_used,
            metadata={}
        )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from the text."""
        try:
            # Try to parse the whole response as JSON first
            try:
                return json.loads(text)
            except:
                pass
            
            # Find JSON blocks, checking for code blocks
            json_blocks = []
            
            # Look for JSON blocks in code fence format: ```json ... ```
            code_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            code_matches = re.findall(code_pattern, text)
            
            for match in code_matches:
                try:
                    json_obj = json.loads(match)
                    return json_obj
                except:
                    pass
            
            # Final fallback: basic JSON extraction between { and }
            json_start = text.find("{")
            json_end = text.rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                try:
                    json_text = text[json_start:json_end+1]
                    return json.loads(json_text)
                except:
                    # Try cleaning up the text by removing line breaks etc
                    json_text = re.sub(r'[\n\r\t]', '', json_text)
                    try:
                        return json.loads(json_text)
                    except:
                        pass
                    
        except Exception as e:
            print(f"Error extracting JSON: {e}")
        
        return {}
    
    async def _synthesize_response(self, request: str, tools_results: List[Dict], session_data: Dict) -> ChatResponse:
        """Synthesize a final response from the tools results."""
        try:
            # Extract tools used and their results
            tools_used = [result["tool"] for result in tools_results]
            sources = []
            collections_used = []
            responses = []
            
            for result in tools_results:
                if "sources" in result:
                    sources.extend(result["sources"])
                if "collections_used" in result:
                    collections_used.extend(result["collections_used"])
                if "result" in result and "responses" in result["result"]:
                    responses.extend(result["result"]["responses"])
            
            # Generate a synthesized answer using the LLM
            mode = session_data.get("mode", "co_reviewer")
            is_initial_request = session_data.get("is_initial_request", False)
            
            # Format the prompt based on mode and request type
            if mode == "co_reviewer" and is_initial_request:
                prompt = f"""Based on the following information about the PR, provide a comprehensive initial code review summary:

{chr(10).join(responses)}

Structure your response as follows:
1. Overview: Summarize the PR's purpose and main changes
2. Key Changes: List and describe the major file changes
3. Potential Areas for Focus: Suggest areas that need attention
4. Next Steps: Recommend what additional information would be helpful

If you don't have enough information for a specific section, say so explicitly rather than making assumptions."""
            else:
                prompt = f"""User Request: {request}

Based on the following information gathered:
{chr(10).join(responses)}

Please provide a detailed response that addresses the user's request. If certain information is missing, acknowledge that and focus on what you can determine from the available data."""

            # Get the synthesized answer from the LLM
            synthesized_answer = str(self.llm.complete(prompt))
            
            # Create the response
            response = ChatResponse(
                session_id=session_data["session_id"],
                answer=synthesized_answer,
                sources=sources,
                collections_used=list(set(collections_used)),  # Remove duplicates
                mode=mode,
                pr_id=session_data["pr_id"],
                tools_used=tools_used,
                metadata={
                    "is_initial_request": is_initial_request
                }
            )
            
            return response
            
        except Exception as e:
            print(f"Error in _synthesize_response: {e}")
            raise e 