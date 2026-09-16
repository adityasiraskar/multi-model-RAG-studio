import os
import gradio as gr
from typing import Dict, List, Tuple
import re
from gtts import gTTS
import tempfile
import json
from PIL import Image

from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import (
    init_pinecone_and_embeddings,
    process_query
)

from llm_pipe.Ingestion_Retrieval.youtube_qa_agent import (
    handle_url_submit,
    answer_question,
    summarize_transcript
)

from llm_pipe.sql_agent import (
    process_query_sql,
    display_columns,
    refresh_models,
    get_available_models
)

from llm_pipe.multi_agent import invoke
from llm_pipe.document_qa import generate_answer_with_openai
from config import app_config


# ============================================================
# Configuration
# ============================================================

PINECONE_API_KEY = app_config.PINECONE_API_KEY
INDEX_NAME = app_config.INDEX_NAME
MODELS_DIR = app_config.MODELS_DIR
EMBEDDING_MODEL = app_config.EMBEDDING_MODEL

COMPANY_PDF_MAPPING = app_config.COMPANY_PDF_MAPPING


# ============================================================
# Document QA Backend
# ============================================================

pc = None
index = None
embeddings_model = None


def _ensure_document_backend():
    """
    Initialize Pinecone and embedding model only when required.
    """

    global pc, index, embeddings_model

    if index is not None and embeddings_model is not None:
        return True

    try:
        pc, index, embeddings_model = init_pinecone_and_embeddings()

    except Exception as exc:
        print(f"[WARN] Pinecone initialization failed: {exc}")

        pc = None
        index = None
        embeddings_model = None

    return index is not None and embeddings_model is not None


# ============================================================
# Format Retrieved Documents
# ============================================================

def format_documents_with_page_info(results):
    """
    Format retrieved documents for inclusion in the LLM context,
    including page number information.
    """

    if not results.get("results"):
        return "No relevant information found."

    formatted_text = ""

    for doc_id, pages in results.get("results", {}).items():

        formatted_text += f"\n=== Document: {doc_id} ===\n"

        for page_num, matches in pages.items():

            formatted_text += f"\n[Page {page_num}]\n"

            for match in matches:

                content_type = (
                    "Table"
                    if match.get("table_info")
                    else "Text"
                )

                formatted_text += (
                    f"--- {content_type}"
                    f"{match.get('table_info', '')} "
                    f"(Relevance score: "
                    f"{match.get('score', 0):.2f}) ---\n"
                )

                formatted_text += (
                    match.get("text", "").strip()
                    + "\n\n"
                )

    return formatted_text or "No relevant information found."


# ============================================================
# Citation / Images
# ============================================================

JSON_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "jsons"
)


def get_all_cited_images(doc_page_mapping):

    images = []

    for doc_id, pages in doc_page_mapping.items():

        json_filename = f"{doc_id}_images.json"

        json_path = os.path.join(
            JSON_DIR,
            json_filename
        )

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


# ============================================================
# Document QA + OpenAI GPT-4o-mini
# ============================================================

def process_query_and_generate(
    company: str,
    query: str
) -> Tuple[str, Dict[str, List[str]], List[str]]:

    """
    Retrieve relevant document chunks from Pinecone
    and generate the final answer using OpenAI GPT-4o-mini.

    Ollama is NOT used in this function.
    """

    if not query.strip():
        return "Please enter a query.", {}, []

    # Initialize Pinecone + embeddings
    if not _ensure_document_backend():

        return (
            "Document search is unavailable because "
            "the Pinecone index could not be initialized.",
            {},
            []
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    results = process_query(
        index,
        embeddings_model,
        company,
        query
    )

    # --------------------------------------------------------
    # Prepare context
    # --------------------------------------------------------

    context = format_documents_with_page_info(results)

    doc_page_mapping = results.get(
        "doc_page_mapping",
        {}
    )

    # --------------------------------------------------------
    # Generate answer using OpenAI GPT-4o-mini
    # --------------------------------------------------------

    if "No relevant information found" in context:

        answer = (
            "I couldn't find relevant information to answer "
            "your question in the selected document."
        )

    else:

        try:

            answer = generate_answer_with_openai(
                context,
                query,
                doc_page_mapping
            )

            # Remove page markers from final answer
            answer = re.sub(
                r"\[Page \d+\]",
                "",
                answer
            ).strip()

        except Exception as exc:

            print(
                f"[ERROR] OpenAI generation failed: {exc}"
            )

            answer = (
                "Error: Unable to generate an answer using "
                "OpenAI GPT-4o-mini."
            )

    # --------------------------------------------------------
    # Get cited images
    # --------------------------------------------------------

    image_paths = []

    for path in get_all_cited_images(
        doc_page_mapping
    ):

        if os.path.exists(path):
            image_paths.append(path)

    return (
        answer,
        doc_page_mapping,
        image_paths
    )


# ============================================================
# Gradio Interface
# ============================================================

def create_interface():

    """
    Create unified Gradio interface.
    """

    # --------------------------------------------------------
    # Create PDF dropdown options
    # --------------------------------------------------------

    pdf_options = []

    for company, pdfs in COMPANY_PDF_MAPPING.items():

        for pdf in pdfs:

            pdf_options.append(
                f"{company.capitalize()}: {pdf}"
            )

    # ========================================================
    # Main Gradio App
    # ========================================================

    with gr.Blocks(
        theme=gr.themes.Soft()
    ) as demo:

        with gr.Tabs():

            # =================================================
            # TAB 1: Multi-Agent Assistant
            # =================================================

            with gr.Tab("🤖 Multi-Agent Assistant"):

                gr.Markdown(
                    "# Multi-Agent Assistant"
                )

                gr.Markdown(
                    "Ask questions about Amazon, Apple, NVIDIA, "
                    "Tesla, search for YouTube videos, or get "
                    "general information."
                )

                chat_history = gr.Chatbot(
                    label="Multi Agent Chatbot",
                    height=350,
                    show_label=True,
                    type="messages"
                )

                with gr.Row():

                    user_input = gr.Textbox(
                        label="Your Question",
                        placeholder=(
                            "Ask about company financials, "
                            "YouTube videos, or general "
                            "information..."
                        ),
                        lines=2,
                        scale=4
                    )

                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1
                    )

                clear_btn = gr.Button(
                    "Clear Chat",
                    variant="secondary"
                )

                with gr.Accordion(
                    "Agent's Thought Process",
                    open=False
                ):

                    intermediate_steps_box = gr.Textbox(
                        label="Routing & Debug Information",
                        lines=3,
                        interactive=False
                    )

                gr.Markdown(
                    """
### Example Questions

**Company Specific:**

- "What were Amazon's North America, International, and AWS net sales in 2022?"
- "What was Tesla's total revenue in 2022?"
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
"""
                )

                def handle_send(
                    message,
                    history
                ):

                    if not message.strip():
                        return history, ""

                    return invoke(
                        message,
                        history
                    )

                send_btn.click(
                    fn=handle_send,
                    inputs=[
                        user_input,
                        chat_history
                    ],
                    outputs=[
                        chat_history,
                        intermediate_steps_box
                    ]
                ).then(
                    lambda: "",
                    outputs=[user_input]
                )

                user_input.submit(
                    fn=handle_send,
                    inputs=[
                        user_input,
                        chat_history
                    ],
                    outputs=[
                        chat_history,
                        intermediate_steps_box
                    ]
                ).then(
                    lambda: "",
                    outputs=[user_input]
                )

                clear_btn.click(
                    lambda: ([], ""),
                    outputs=[
                        chat_history,
                        intermediate_steps_box
                    ]
                )

            # =================================================
            # TAB 2: Document QA
            # =================================================

            with gr.Tab("📄 Document QA System"):

                gr.Markdown(
                    "# 📊 Financial Reports QA\n"
                    "**Extract valuable insights from SEC filings "
                    "and annual reports using OpenAI GPT-4o-mini**"
                )

                with gr.Row():

                    # ------------------------------------------------
                    # Left Column
                    # ------------------------------------------------

                    with gr.Column(
                        scale=1,
                        min_width=600
                    ):

                        gr.Markdown(
                            "### 📁 Select Report & Ask Your Question"
                        )

                        pdf_dropdown = gr.Dropdown(
                            choices=pdf_options,
                            label="Select Financial Report",
                            value=(
                                pdf_options[0]
                                if pdf_options
                                else None
                            ),
                            interactive=True
                        )

                        query_input = gr.Textbox(
                            lines=3,
                            label="Your Question",
                            placeholder=(
                                "Example: What were the total "
                                "R&D expenses for 2022?"
                            )
                        )

                        submit_btn = gr.Button(
                            "Get Answer",
                            variant="primary"
                        )

                        gr.Markdown(
                            "### 🧠 Answer from GPT-4o-mini"
                        )

                        answer_output = gr.Textbox(
                            label="Answer",
                            lines=10,
                            show_copy_button=True
                        )

                        speak_btn = gr.Audio(
                            label="Click Play to Hear",
                            type="filepath"
                        )

                    # ------------------------------------------------
                    # Right Column
                    # ------------------------------------------------

                    with gr.Column(
                        scale=1,
                        min_width=600
                    ):

                        gr.Markdown(
                            "### 📄 Pages Cited in Answer"
                        )

                        citation_gallery = gr.Gallery(
                            label="Source Pages",
                            show_label=True,
                            columns=1,
                            height="600px",
                            object_fit="contain",
                            preview=True
                        )

                # =================================================
                # Document QA Submit Handler
                # =================================================

                def on_submit(
                    pdf_selection,
                    query
                ):

                    if not pdf_selection:

                        return (
                            "Please select a financial report first.",
                            None,
                            []
                        )

                    if not query.strip():

                        return (
                            "Please enter a question.",
                            None,
                            []
                        )

                    # Extract company name
                    company = (
                        pdf_selection
                        .split(":")[0]
                        .lower()
                    )

                    # Retrieve + Generate
                    (
                        answer,
                        doc_page_mapping,
                        image_paths
                    ) = process_query_and_generate(
                        company,
                        query
                    )

                    # ------------------------------------------------
                    # Load cited page images
                    # ------------------------------------------------

                    image_objects = []

                    for path in image_paths:

                        if os.path.exists(path):

                            try:

                                image_objects.append(
                                    Image.open(path)
                                )

                            except Exception as exc:

                                print(
                                    "[WARN] Failed to open "
                                    f"image {path}: {exc}"
                                )

                    # ------------------------------------------------
                    # Text-to-Speech
                    # ------------------------------------------------

                    audio_path = None

                    try:

                        if answer and answer.strip():

                            tts = gTTS(answer)

                            temp_file = (
                                tempfile.NamedTemporaryFile(
                                    delete=False,
                                    suffix=".mp3"
                                )
                            )

                            tts.save(
                                temp_file.name
                            )

                            audio_path = (
                                temp_file.name
                            )

                    except Exception as exc:

                        print(
                            "[WARN] Failed to create "
                            f"speech output: {exc}"
                        )

                    return (
                        answer,
                        audio_path,
                        image_objects
                    )

                submit_btn.click(
                    fn=on_submit,
                    inputs=[
                        pdf_dropdown,
                        query_input
                    ],
                    outputs=[
                        answer_output,
                        speak_btn,
                        citation_gallery
                    ]
                )

            # =================================================
            # TAB 3: YouTube QA Agent
            # =================================================

            with gr.Tab("🎥 YouTube QA Agent"):

                gr.Markdown(
                    "## 🤖 YouTube QA Agent (Powered by Ollama)\n"
                    "Ask questions and summarize any YouTube video "
                    "using an LLM."
                )

                with gr.Group():

                    gr.Markdown(
                        "### 🔗 Load YouTube Transcript"
                    )

                    with gr.Row():

                        with gr.Column(
                            scale=4,
                            min_width=400
                        ):

                            url_input = gr.Textbox(
                                label="📺 YouTube URL",
                                placeholder=(
                                    "Paste a YouTube video link..."
                                ),
                                lines=2
                            )

                        with gr.Column(
                            scale=2,
                            min_width=400
                        ):

                            model_selector = gr.Dropdown(
                                choices=[
                                    "phi3:mini",
                                    "mistral",
                                    "llama3",
                                    "gemma"
                                ],
                                label="🧠 Ollama Model",
                                value="phi3:mini",
                                interactive=True
                            )

                    fetch_button = gr.Button(
                        "📥 Fetch Transcript",
                        variant="primary"
                    )

                    status_output = gr.Textbox(
                        label="📡 Transcript Status",
                        interactive=False,
                        lines=2
                    )

                fetch_button.click(
                    fn=handle_url_submit,
                    inputs=[url_input],
                    outputs=status_output
                )

                gr.Markdown("---")

                with gr.Group():

                    gr.Markdown(
                        "### ❓ Ask a Question About the Video"
                    )

                    question_input = gr.Textbox(
                        label="💬 Your Question",
                        placeholder=(
                            "What is the video about?"
                        ),
                        lines=2
                    )

                    ask_button = gr.Button(
                        "🧠 Ask",
                        variant="secondary"
                    )

                    youtube_answer_output = gr.Textbox(
                        label="📝 Answer",
                        lines=5,
                        interactive=False
                    )

                ask_button.click(
                    fn=answer_question,
                    inputs=[
                        question_input,
                        url_input,
                        model_selector
                    ],
                    outputs=youtube_answer_output
                )

                gr.Markdown("---")

                with gr.Group():

                    gr.Markdown(
                        "### 📄 Summarize the Video"
                    )

                    summarize_button = gr.Button(
                        "📝 Summarize Video",
                        variant="secondary"
                    )

                    summary_output = gr.Textbox(
                        label="🧾 Summary",
                        lines=5,
                        interactive=False
                    )

                summarize_button.click(
                    fn=summarize_transcript,
                    inputs=[
                        url_input,
                        model_selector
                    ],
                    outputs=summary_output
                )

            # =================================================
            # TAB 4: SQL Agent
            # =================================================

            with gr.Tab("📄 SQL Agent Tab"):

                gr.Markdown(
                    "# 🔍 Enhanced SQL RAG System with Ollama Integration"
                )

                gr.Markdown(
                    "Upload a CSV file and ask questions about "
                    "your data in natural language."
                )

                with gr.Row():

                    csv_file = gr.File(
                        label="📁 Upload CSV File",
                        file_types=[".csv"]
                    )

                with gr.Row():

                    upload_status = gr.Textbox(
                        label="📊 Upload Status",
                        interactive=False
                    )

                    columns_display = gr.Textbox(
                        label="📋 Available Columns",
                        lines=4,
                        interactive=False
                    )

                with gr.Row():

                    use_ollama = gr.Checkbox(
                        label="🤖 Use Ollama for SQL Generation",
                        value=True
                    )

                    available_models = get_available_models()

                    model_dropdown = gr.Dropdown(
                        label="🔧 Select Ollama Model",
                        choices=available_models,
                        value=(
                            available_models[0]
                            if available_models
                            else None
                        ),
                        interactive=True
                    )

                    refresh_btn = gr.Button(
                        "🔄 Refresh Models",
                        size="sm"
                    )

                with gr.Row():

                    sql_query_input = gr.Textbox(
                        label="💬 Ask a question about your data",
                        placeholder=(
                            "e.g., 'What is the average price?' "
                            "or 'Show me the top 5 highest sales records'"
                        ),
                        lines=2
                    )

                sql_submit_btn = gr.Button(
                    "🚀 Submit Query",
                    variant="primary",
                    size="lg"
                )

                sql_status_output = gr.Textbox(
                    label="📈 Query Status",
                    interactive=False
                )

                sql_output = gr.Textbox(
                    label="💾 Generated SQL Query",
                    lines=2,
                    interactive=False
                )

                result_output = gr.Dataframe(
                    label="📊 Results",
                    interactive=False
                )

                # ------------------------------------------------
                # SQL Event Handlers
                # ------------------------------------------------

                csv_file.change(
                    fn=display_columns,
                    inputs=[csv_file],
                    outputs=[
                        upload_status,
                        columns_display
                    ]
                )

                refresh_btn.click(
                    fn=refresh_models,
                    outputs=[model_dropdown]
                )

                sql_submit_btn.click(
                    fn=process_query_sql,
                    inputs=[
                        csv_file,
                        sql_query_input,
                        model_dropdown,
                        use_ollama
                    ],
                    outputs=[
                        sql_status_output,
                        sql_output,
                        result_output
                    ]
                )

    return demo


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    demo = create_interface()

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(
            os.getenv("PORT", "7860")
        ),
        share=False,
        show_error=True
    )
