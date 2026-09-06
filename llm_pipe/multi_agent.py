from langchain.schema import SystemMessage 
from langchain.schema import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from llm_pipe.multi_agent_tools import agents, query_ollama
from llm_pipe.multi_agent_tools import (
    get_answer_apple,
    get_answer_nvidia,
    get_answer_document,
    youtube_url,
    web_search,
)
from config import app_config
import re
import json

def llm_route_question(question: str, chat_history: list = None) -> str:
    """
    Use LLM to determine which tool to use based on the question content and chat history
    """
    # Format chat history for context
    history_context = ""
    if chat_history and len(chat_history) > 0:
        history_context = "\n\nRecent conversation history:\n"
        # Get last 3 exchanges for context (to avoid token limits)
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        
        for i, entry in enumerate(recent_history):
            if entry["role"] == "user":
                history_context += f"User: {entry['content']}\n"
            elif entry["role"] == "assistant":
                history_context += f"Assistant: {entry['content'][:200]}...\n"  # Truncate long responses
    
    routing_prompt = f"""You are a routing assistant that determines which tool should be used to answer a user's question.

Available tools:
1. get_answer_apple - Use for questions specifically about Apple company, its financials, operations, or business details
2. get_answer_nvidia - Use for questions specifically about NVIDIA company, its financials, operations, or business details
3. get_answer_document - Use for questions about Amazon or Tesla, the other companies covered by the financial-report corpus
4. youtube_url - Use when user asks for YouTube videos, video content, tutorials, or wants to watch something
5. web_search - Use for general questions, current events, news, or information not specific to the document corpus

Rules:
- If the user asks for videos, YouTube content, or anything related to watching/viewing, use youtube_url
- If the user asks specifically about Apple or NVIDIA company information, use the respective company tool
- If the user asks about Amazon or Tesla company information, use get_answer_document
- For general questions or current information, use web_search
- Pay attention to conversation history to understand context and follow-up questions
- If a follow-up question refers to a previously discussed company (like "which country gave highest sales" after discussing Apple), use that company's tool
- Only respond with the exact tool name, nothing else

{history_context}

Current question: {question}

Tool to use:"""

    try:
        response = query_ollama(routing_prompt)
        response = response.strip().lower()
        
        # Map the response to exact function names
        if 'get_answer_apple' in response or 'apple' in response:
            return 'get_answer_apple'
        elif 'get_answer_nvidia' in response or 'nvidia' in response:
            return 'get_answer_nvidia'
        elif 'get_answer_document' in response or any(
            company in response for company in app_config.COMPANY_PDF_MAPPING
        ):
            return 'get_answer_document'
        elif 'youtube_url' in response or 'youtube' in response:
            return 'youtube_url'
        elif 'web_search' in response or 'web' in response:
            return 'web_search'
        else:
            # If LLM response is unclear, fall back to web_search
            print(f"[WARNING] Unclear LLM routing response: {response}. Defaulting to web_search")
            return 'web_search'
            
    except Exception as e:
        print(f"[ERROR] LLM routing failed: {str(e)}. Defaulting to web_search")
        return 'web_search'

class BaseLLM():
    def __init__(self, message: str, history):
        self.user_input = message
        self.history = history
        
        # Map function names to actual functions
        self.function_map = {
            "get_answer_apple": get_answer_apple,
            "get_answer_nvidia": get_answer_nvidia,
            "get_answer_document": get_answer_document,
            "youtube_url": youtube_url,
            "web_search": web_search,
        }

    def generate_answer(self):
        try:
            # Use LLM to route the question to appropriate tool with chat history
            selected_function = llm_route_question(self.user_input, self.history)
            
            # Print the routing decision to terminal
            print(f"\n{'='*50}")
            print(f"[ROUTING] Question: {self.user_input}")
            print(f"[ROUTING] Chat History Length: {len(self.history)} messages")
            print(f"[ROUTING] Selected Tool: {selected_function}")
            print(f"{'='*50}")
            
            # Get the function
            if selected_function in self.function_map:
                func = self.function_map[selected_function]
                
                # Print tool execution to terminal
                print(f"[EXECUTION] Calling {selected_function}...")
                
                # Call the function directly
                answer = func(self.user_input)
                
                print(f"[EXECUTION] {selected_function} completed successfully")
                print(f"[RESPONSE] Length: {len(answer)} characters")
                
                return {
                    "output": answer,
                    "intermediate_steps": f"LLM routed to tool: {selected_function} (with chat history context)"
                }
            else:
                # Fallback to general response
                print(f"[FALLBACK] Using direct Ollama response")
                prompt = f"""Answer the following question correctly and concisely:

Question: {self.user_input}

Answer:"""
                
                response = query_ollama(prompt)
                return {
                    "output": response,
                    "intermediate_steps": "Direct response from Ollama (no tool routing)"
                }
                
        except Exception as e:
            print(f"[ERROR] Exception in generate_answer: {str(e)}")
            return {
                "output": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "intermediate_steps": f"Error occurred: {str(e)}"
            }

def invoke(message, history):
    try:
        print(f"\n[INVOKE] New request received: {message}")
        
        # Format history for the agent
        formatted_history = []
        for user_msg, assistant_msg in history:
            if user_msg:
                formatted_history.append({"role": "user", "content": user_msg})
            if assistant_msg:
                formatted_history.append({"role": "assistant", "content": assistant_msg})

        # Create agent instance
        agent = BaseLLM(message=message, history=formatted_history)
        
        # Generate response
        output = agent.generate_answer()
        main_output = str(output['output'])
        intermediate_steps = str(output['intermediate_steps'])

        # Update history
        updated_history = history + [(message, main_output)]
        
        print(f"[INVOKE] Response generated successfully")
        return updated_history, intermediate_steps
        
    except Exception as e:
        print(f"[INVOKE ERROR] {str(e)}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        updated_history = history + [(message, error_msg)]
        return updated_history, f"Error in invoke function: {str(e)}"