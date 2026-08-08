import os
import gradio as gr
from typing import Dict, List, Tuple
import re
import requests
from gtts import gTTS
import tempfile
import json
import platform
from PIL import Image
from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import init_pinecone_and_embeddings, process_query
from llm_pipe.Ingestion_Retrieval.youtube_qa_agent import handle_url_submit, answer_question, summarize_transcript
from llm_pipe.sql_agent import process_query_sql, display_columns, refresh_models, get_available_models
from llm_pipe.multi_agent import invoke
from config import app_config


# Configuration
PINECONE_API_KEY = app_config.PINECONE_API_KEY
INDEX_NAME = app_config.INDEX_NAME
MODELS_DIR = app_config.MODELS_DIR
EMBEDDING_MODEL = app_config.EMBEDDING_MODEL

# Configure Ollama API - use localhost as Ollama should be running locally
OLLAMA_API_BASE = app_config.OLLAMA_API_BASE
OLLAMA_MODEL = app_config.OLLAMA_MODEL

# Detect OS for better error messages
is_windows = platform.system() == "Windows"
is_wsl = "microsoft" in platform.uname().release.lower() or "wsl" in platform.uname().release.lower()

# Company to PDF mapping - maps company names to their respective PDF files
COMPANY_PDF_MAPPING = app_config.COMPANY_PDF_MAPPING


def check_ollama_available():
    """Check if Ollama is available by sending a request to list models"""
    try:
        print(f"Trying to connect to Ollama at {OLLAMA_API_BASE}...")
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            if OLLAMA_MODEL in available_models:
                print(f"Ollama is available with {OLLAMA_MODEL}")
                return True
            else:
                print(f"Warning: {OLLAMA_MODEL} not found in Ollama. Available models: {available_models}")
                return False
        else:
            print(f"Ollama API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        return False

# Initialize Pinecone and embedding models using the imported function
try:
    pc, index, embeddings_model = init_pinecone_and_embeddings()
except Exception as exc:
    print(f"[WARN] Pinecone initialization failed: {exc}")
    pc = index = embeddings_model = None

ollama_available = check_ollama_available()

def format_documents_with_page_info(results):
    """
    Format retrieved documents for inclusion in the context, 
    including clear page number formatting for the LLM to recognize.
    """
    if not results.get("results"):
        return "No relevant information found."
    
    formatted_text = ""
    
    # Extract the document-page mapping for reference
    doc_page_mapping = results.get("doc_page_mapping", {})
    
    # Process results by document and page
    for doc_id, pages in results.get("results", {}).items():
        formatted_text += f"\n=== Document: {doc_id} ===\n"
        
        for page_num, matches in pages.items():
            formatted_text += f"\n[Page {page_num}]\n"
            
            for match in matches:
                content_type = "Table" if match.get("table_info") else "Text"
                formatted_text += f"--- {content_type}{match.get('table_info', '')} (Relevance score: {match.get('score', 0):.2f}) ---\n"
                formatted_text += match.get("text", "").strip() + "\n\n"
    
    return formatted_text or "No relevant information found."

def generate_answer_with_ollama(context: str, query: str, doc_page_mapping: Dict) -> str:
    """Generate an answer using Ollama's Phi-3 model based on context and query,
    with instructions to include page references."""
    
    # Format the doc-page mapping for inclusion in the prompt
    page_references = ""
    for doc_id, pages in doc_page_mapping.items():
        page_references += f"- Document '{doc_id}' has relevant information on pages: {', '.join(pages)}\n"
    
    prompt = f"""You are a helpful assistant that answers questions about financial reports and SEC filings.
Answer the user's question based solely on the provided context.

IMPORTANT INSTRUCTIONS:
1. Always cite the specific page numbers when referencing information in your answer.
2. Format page citations as [Page X] within your answer text.
3. If information comes from multiple pages, cite each relevant page.
4. If you don't know the answer based on the context, just say so.
5. Don't make up information.

The following pages contain relevant information for this query:
{page_references}

Context:
{context}

User Question:
{query}

Answer (remember to cite specific page numbers in your response):"""
    
    # Make API call to Ollama
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
            timeout=180  # Increase timeout to 180 seconds
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Error generating response.")
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Function for citation
JSON_DIR = os.path.join(os.path.dirname(__file__), "..", "jsons")
def get_all_cited_images(doc_page_mapping):

    images = []
    for doc_id, pages in doc_page_mapping.items():
        json_filename = f"{doc_id}_images.json"
        json_path = os.path.join(JSON_DIR, json_filename)
        if os.path.exists(json_path):
            print(f"Loading JSON: {json_path}")
            with open(json_path, "r") as f:
                image_map = json.load(f)
                for page in pages:
                    img = image_map.get(str(page))
                    if img:
                        print(f"Found image path: {img}")    
                        images.append(img)
    return images

def process_query_and_generate(company: str, query: str) -> Tuple[str, Dict[str, List[str]], List[str]]:
    """Process a query for a specific company and generate answer with page references."""
    if not query.strip():
        return "Please enter a query."
    
    if index is None or embeddings_model is None:
        return "Document search is unavailable because the Pinecone index could not be initialized.", {}, []

    # Use the imported process_query function to get search results
    results = process_query(index, embeddings_model, company, query)
    
    # Format retrieved documents with clear page information
    context = format_documents_with_page_info(results)
    
    # Get document-page mapping for reference in the prompt
    doc_page_mapping = results.get("doc_page_mapping", {})
    
    # Generate answer using Phi-3 via Ollama with instructions to include page references
    if "No relevant information found" in context:
        answer = "I couldn't find relevant information to answer your question in the selected document."
    else:
        try:
            answer = generate_answer_with_ollama(context, query, doc_page_mapping)
            
            # Post-process answer to highlight page citations
            answer = re.sub(r'\[Page \d+\]', '', answer).strip()
        except Exception as e:
            answer = f"Error: Unable to connect to Ollama. Please make sure Ollama is installed and running with the phi3:mini model."
    
    image_paths = []
    for path in get_all_cited_images(doc_page_mapping):
        if os.path.exists(path):
            image_paths.append(path)
    return answer, doc_page_mapping, image_paths


# Gradio GUI
def create_interface():
    """Create unified Gradio interface with two tabs: Document QA + YouTube QA."""
    # Create list of PDF options for dropdown
    pdf_options = []
    for company, pdfs in COMPANY_PDF_MAPPING.items():
        for pdf in pdfs:
            pdf_options.append(f"{company.capitalize()}: {pdf}")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tabs():
            # === Tab 1: Multi-Agent Assistant ===
            with gr.Tab("🤖 Multi-Agent Assistant"):
                gr.Markdown("#  Multi-Agent Assistant")
                gr.Markdown("Ask questions about EnerSys, Apple, NVIDIA, search for YouTube videos, or get general information.")
                
                # Chat interface
                chat_history = gr.Chatbot(
                    label="Multi Agent Chatbot",
                    height=350,
                    show_label=True
                )

                # User input
                with gr.Row():
                    user_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about company financials, YouTube videos, or general information...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # Clear button
                clear_btn = gr.Button("Clear Chat", variant="secondary")

                # Intermediate steps display (collapsed by default)
                with gr.Accordion("Agent's Thought Process", open=False):
                    intermediate_steps_box = gr.Textbox(
                        label="Routing & Debug Information",
                        lines=3,
                        interactive=False
                    )

                # Example questions
                gr.Markdown("### Example Questions:")
                gr.Markdown("""
                **Company Specific:**
                - "What was EnerSys' quarterly dividend in fiscal year 2023?"
                - "What was the effective income tax rate in fiscal 2023 of EnerSys?"
                - "Tell me about NVIDIA's AI business growth"
                - "Apple's reportable segments consist of which countries?"
                
                **YouTube Videos:**
                - "YouTube videos related to Apple"
                - "Show me videos about NVIDIA graphics cards"
                - "Find YouTube tutorials on financial analysis"
                
                **General Search:**
                - "Search for recent news about electric vehicle batteries"
                - "Current stock market trends"
                - "Latest AI developments"
                """)

                # Connect the send button to the invoke function
                def handle_send(message, history):
                    if not message.strip():
                        return history, ""
                    return invoke(message, history)

                send_btn.click(
                    fn=handle_send,
                    inputs=[user_input, chat_history],
                    outputs=[chat_history, intermediate_steps_box]
                ).then(
                    lambda: "",  # Clear input after sending
                    outputs=[user_input]
                )

                # Handle Enter key
                user_input.submit(
                    fn=handle_send,
                    inputs=[user_input, chat_history],
                    outputs=[chat_history, intermediate_steps_box]
                ).then(
                    lambda: "",  # Clear input after sending
                    outputs=[user_input]
                )

                # Clear chat functionality
                clear_btn.click(
                    lambda: ([], ""),
                    outputs=[chat_history, intermediate_steps_box]
                )

            # === Tab 2: Document QA System ===
            with gr.Tab("📄 Document QA System"):
                gr.Markdown("# 📊 Financial Reports QA\n**Extract valuable insights from SEC filings and annual reports**")

                if not ollama_available:
                    gr.Markdown("⚠️ **Ollama is not available.** Ensure it is running and the Phi-3 model is pulled.")

                with gr.Row():  # Full row split in two halves
                    with gr.Column(scale=1, min_width=600):  # Left side
                        with gr.Column():
                            gr.Markdown("### 📁 Select Report & Ask Your Question")

                            pdf_dropdown = gr.Dropdown(
                                choices=pdf_options,
                                label="Select Financial Report",
                                value=pdf_options[0] if pdf_options else None,
                                interactive=True
                            )

                            query_input = gr.Textbox(
                                lines=3,
                                label="Your Question",
                                placeholder="Example: What were the total R&D expenses for 2022?"
                            )

                            submit_btn = gr.Button("Get Answer", variant="primary")

                        with gr.Column():
                            gr.Markdown("### 🧠 Answer from the AI")

                            answer_output = gr.Textbox(
                                label="Answer",
                                lines=10,
                                show_copy_button=True
                            )

                            speak_btn = gr.Audio(label="Click Play to Hear", type="filepath")

                    with gr.Column(scale=1, min_width=600):  # Right side
                        with gr.Column():
                            gr.Markdown("### 📄 Pages Cited in Answer")

                            citation_gallery = gr.Gallery(
                            label="Source Pages",
                            show_label=True,
                            columns=1,
                            height="600px",
                            object_fit="contain",
                            preview=True
                            )


                # Handle submission
                def on_submit(pdf_selection, query):
                    if not pdf_selection:
                        return "Please select a financial report first.", None, []

                    company = pdf_selection.split(":")[0].lower()
                    answer, doc_page_mapping, image_paths = process_query_and_generate(company, query)

                    # Load all cited page images using the proper helper
                    image_objects = []
                    for path in image_paths:
                        if os.path.exists(path):
                            try:
                                image_objects.append(Image.open(path))
                            except Exception as e:
                                print(f"Warning: Failed to open image {path}: {e}")

                    # Convert answer to speech using gTTS
                    try:
                        if answer and answer.strip():
                            tts = gTTS(answer)
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                            tts.save(temp_file.name)
                            audio_path = temp_file.name
                        else:
                            audio_path = None
                    except Exception as exc:
                        print(f"Warning: Failed to create speech output: {exc}")
                        audio_path = None

                    return answer, audio_path, image_objects

                submit_btn.click(
                    fn=on_submit,
                    inputs=[pdf_dropdown, query_input],
                    outputs=[answer_output, speak_btn, citation_gallery]
                )

            # === Tab 3: YouTube QA Agent ===
            with gr.Tab("🎥 YouTube QA Agent"):
                gr.Markdown("## 🤖 YouTube QA Agent (Powered by Ollama)\nAsk questions and summarize any YouTube video using an LLM.")

                # Section 1: URL & Model Selection
                with gr.Group():
                    gr.Markdown("### 🔗 Load YouTube Transcript")

                    with gr.Row():
                        with gr.Column(scale=4, min_width=400):
                            url_input = gr.Textbox(label="📺 YouTube URL", placeholder="Paste a YouTube video link...", lines=2)
                        with gr.Column(scale=2, min_width=400):
                            model_selector = gr.Dropdown(
                                choices=["phi3:mini", "mistral", "llama3", "gemma"],
                                label="🧠 Ollama Model",
                                value="phi3:mini",
                                interactive=True
                            )
                            fetch_button = gr.Button("📥 Fetch Transcript", variant="primary")

                    status_output = gr.Textbox(label="📡 Transcript Status", interactive=False, lines=2)

                fetch_button.click(fn=handle_url_submit, inputs=[url_input], outputs=status_output)

                gr.Markdown("---")

                # Section 2: Ask Questions
                with gr.Group():
                    gr.Markdown("### ❓ Ask a Question About the Video")

                    with gr.Column():
                        question_input = gr.Textbox(label="💬 Your Question", placeholder="What is the video about?", lines=2)
                        ask_button = gr.Button("🧠 Ask", variant="secondary", scale=1)

                    answer_output = gr.Textbox(label="📝 Answer", lines=5, interactive=False)

                ask_button.click(fn=answer_question, inputs=[question_input, url_input, model_selector], outputs=answer_output)

                gr.Markdown("---")

                # Section 3: Summarize
                with gr.Group():
                    gr.Markdown("### 📄 Summarize the Video")

                    with gr.Row():
                        summarize_button = gr.Button("📝 Summarize Video", variant="secondary")

                    summary_output = gr.Textbox(label="🧾 Summary", lines=5, interactive=False)

                summarize_button.click(fn=summarize_transcript, inputs=[url_input, model_selector], outputs=summary_output)

            # === Tab 4: SQL Agent Tab ===
            with gr.Tab("📄 SQL Agent Tab"):
                gr.Markdown("# 🔍 Enhanced SQL RAG System with Ollama Integration")
                gr.Markdown("Upload a CSV file and ask questions about your data in natural language. Enhanced with better error handling and column matching.")
                
                with gr.Row():
                    csv_file = gr.File(label="📁 Upload CSV File", file_types=[".csv"])
                
                with gr.Row():
                    upload_status = gr.Textbox(label="📊 Upload Status", interactive=False)
                    columns_display = gr.Textbox(label="📋 Available Columns", lines=4, interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        use_ollama = gr.Checkbox(label="🤖 Use Ollama for SQL Generation", value=ollama_available)
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            label="🔧 Select Ollama Model",
                            choices=get_available_models(),
                            value=get_available_models()[0] if get_available_models() else "",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
                
                with gr.Row():
                    sql_query_input = gr.Textbox(  # Renamed to avoid confusion
                        label="💬 Ask a question about your data",
                        placeholder="e.g., 'What is the average price?' or 'Show me the top 5 highest sales records'",
                        lines=2
                    )
                
                with gr.Row():
                    sql_submit_btn = gr.Button("🚀 Submit Query", variant="primary", size="lg")  # Renamed to avoid confusion
                
                with gr.Row():
                    sql_status_output = gr.Textbox(label="📈 Query Status", interactive=False)  # Renamed to avoid confusion
                
                with gr.Row():
                    sql_output = gr.Textbox(label="💾 Generated SQL Query", lines=2, interactive=False)
                
                with gr.Row():
                    result_output = gr.Dataframe(label="📊 Results", interactive=False)
                
                # Event handlers
                csv_file.change(
                    fn=display_columns,
                    inputs=[csv_file],
                    outputs=[upload_status, columns_display]
                )
                
                refresh_btn.click(
                    fn=refresh_models,
                    outputs=[model_dropdown]
                )
                
                sql_submit_btn.click(  # Use the renamed button
                    fn=process_query_sql,  # Use the SQL-specific function
                    inputs=[csv_file, sql_query_input, model_dropdown, use_ollama],  # Use renamed input
                    outputs=[sql_status_output, sql_output, result_output]  # Use renamed output
                )


    return demo


# Launch the Gradio app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)