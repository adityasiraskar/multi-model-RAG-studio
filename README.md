# Multi-Modal RAG System

## 🎯 System Overview

This is a comprehensive Multi-Modal Retrieval-Augmented Generation (RAG) system that combines multi-agent conversations, document question-answering, YouTube video analysis, and CSV data querying capabilities. The system leverages open-source models via Ollama and provides an intuitive web interface built with Gradio for users to interact with various data sources.

### 🌟 Key Capabilities

#### 🤖 Multi-Agent Assistant
- **Intelligent Routing**: Automatically routes queries to appropriate specialized agents
- **Document QA**: Extract insights from financial reports and SEC filings (EnerSys, Apple, NVIDIA)
- **YouTube Integration**: Search and analyze YouTube video content with transcript extraction
- **General Search**: Web search capabilities for current information
- **Conversational Interface**: Chat-based interaction with memory and thought process visualization

#### 📄 Document QA System
- **Financial Report Analysis**: Process SEC filings and annual reports with company-specific document mapping
- **Semantic Search**: Vector-based document retrieval using Pinecone
- **Page Citation**: Automatic page number references in answers with visual citations
- **Gallery Display**: Show relevant document pages as images in an organized gallery
- **Text-to-Speech**: Convert answers to audio using Google Text-to-Speech (gTTS)
- **Ollama Integration**: Uses Phi-3 mini model for answer generation with fallback mechanisms

#### 🎥 YouTube QA Agent
- **Transcript Extraction**: Automatic transcript retrieval from YouTube videos
- **Video Summarization**: AI-powered video content summarization using multiple Ollama models
- **Question Answering**: Ask specific questions about video content
- **Multiple Models**: Support for phi3:mini, mistral, llama3, gemma models
- **Model Selection**: Dynamic model selection for different use cases

#### 📊 SQL Agent
- **Natural Language to SQL**: Convert plain English queries to SQL using Ollama or rule-based fallback
- **CSV Processing**: Upload and analyze CSV files with automatic schema detection
- **Smart Column Matching**: Fuzzy matching for column names with enhanced error handling
- **Multiple Encodings**: Support for various CSV encodings (utf-8, latin-1, cp1252, iso-8859-1)
- **Query Validation**: Robust error handling with informative error messages
- **Model Integration**: Configurable Ollama model selection with availability checking

## 🏗️ Architecture

The system follows a modular architecture with four main components accessible through a unified Gradio interface:

<img src="Flow diagram/RAG_SYSTEM.png" alt="Architecture">


## 🔧 Core Components

### 1. Vector Database Integration
- **Pinecone**: Primary vector database for document embeddings
- **Sentence Transformers**: Generate embeddings using all-MiniLM-L6-v2 model
- **Document Mapping**: Company-specific PDF mapping (EnerSys, Apple, NVIDIA)
- **Retrieval System**: Semantic search with relevance scoring and page citations
- **Image Citations**: JSON-based image mapping for visual page references

### 2. Language Model Integration
- **Ollama**: Local LLM server with multiple model support
  - Primary: `phi3:mini` for fast responses
  - Alternative: `mistral`, `llama3`, `gemma` for specialized tasks
- **Health Checking**: Automatic Ollama availability detection
- **Fallback Mechanisms**: Rule-based responses when LLM is unavailable
- **Model Management**: Dynamic model selection and refresh capabilities
- **Timeout Handling**: Graceful handling of slow responses (180s timeout)

### 3. Data Processing Pipeline
- **PDF Processing**: Extract text with page mapping and image citations
- **CSV Processing**: Multi-encoding support with automatic schema detection
- **YouTube Processing**: Transcript extraction using youtube-transcript-api
- **Column Matching**: Fuzzy string matching for SQL query generation

### 4. Web Interface (Gradio)
- **Multi-Tab Interface**: Four specialized tabs for different functionalities
- **Real-time Processing**: Async handling with progress indicators
- **Interactive Elements**: Dropdowns, file uploads, model selection
- **Visual Feedback**: Status messages, error handling, and debug information

## ✨ Features

### 🤖 Multi-Agent Assistant
- **Conversational Interface**: Chat-based interaction with message history
- **Intelligent Routing**: Automatic query classification and agent selection
- **Debug Visibility**: Thought process and routing information display
- **Example Queries**: Pre-defined examples for different query types

### 📄 Document QA System
- **Company Selection**: Dropdown for EnerSys, Apple, NVIDIA documents
- **Answer Generation**: Context-aware responses with page citations
- **Visual Citations**: Gallery display of referenced document pages
- **Audio Output**: Text-to-speech functionality for accessibility
- **Page Reference**: Automatic [Page X] citations in responses

### 🎥 YouTube QA Agent
- **URL Processing**: Direct YouTube URL input with transcript fetching
- **Model Selection**: Choose between phi3:mini, mistral, llama3, gemma
- **Question Answering**: Interactive Q&A about video content
- **Video Summarization**: One-click video summary generation
- **Status Tracking**: Real-time status updates for transcript processing

### 📊 SQL Agent
- **File Upload**: Drag-and-drop CSV file support
- **Column Display**: Automatic column listing with data type information
- **Query Generation**: Natural language to SQL conversion
- **Model Toggle**: Enable/disable Ollama integration with fallback
- **Result Display**: Tabular result presentation with error handling

## 🔄 System Workflow

### 1. Multi-Agent Assistant Workflow
```
User Query → Query Analysis → Route to Appropriate Agent → Execute → Unified Response
```

### Document QA Workflow
```
User Upload PDF → Process & Chunk → Generate Embeddings → Store in Pinecone
                                                               ↓
User Query → Semantic Search → Retrieve Relevant Chunks → Generate Answer → Display with Citations
```

### YouTube QA Workflow
```
YouTube URL → Extract Transcript → Store Locally → Process with Ollama → Generate Responses
```

### SQL Agent Workflow
```
CSV Upload → Create SQLite DB → Natural Language Query → Generate SQL → Execute → Return Results
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama installed and running locally on port 11434
- Pinecone account and API key
- Tavily API key

### Required Ollama Models
```bash
# Install recommended models
ollama pull phi3:mini      # Primary model (fast, efficient)
ollama pull mistral        # Alternative model
ollama pull llama3         # Advanced model
ollama pull gemma          # Google's model
```

### Python Dependencies
```bash
# Core dependencies
pip install gradio>=4.0.0
pip install pinecone-client
pip install sentence-transformers
pip install pandas
pip install requests
pip install python-dotenv
pip install Pillow

# Audio and video processing
pip install gtts
pip install youtube-transcript-api

# Optional dependencies
pip install sqlite3  # Usually included with Python
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the `llm_pipe/Ingestion_Retrieval` directory:
```env
api_key=your_pinecone_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### Configuration Settings
The system uses the following default settings in `config.py`:
```python
PINECONE_API_KEY: str
INDEX_NAME: str = "document-analysis"
MODELS_DIR: str = "models"
EMBEDDING_MODEL: str = "all-mpnet-base-v2"
OLLAMA_API_BASE: str = "http://localhost:11434/api"
OLLAMA_MODEL: str = "phi3:mini"
```

### Company PDF Mapping
The system is configured with the following document mapping in `config.py`:
```python
COMPANY_PDF_MAPPING = {
    "enersys": ["EnerSys-2023-10K.pdf", "EnerSys-2017-10K.pdf"],
    "amazon": ["Amazon10k2022.pdf"],
    "apple": ["Apple_10-K-2021.pdf"],
    "nvidia": ["Nvidia.pdf"],
    "tesla": ["Tesla.pdf"],
    "lockheed": ["Lockheed_martin_10k.pdf"],
    "advent": ["Advent_Technologies_2022_10K.pdf"],
    "transdigm": ["TransDigm-2022-10K.pdf"]
}
```

### Directory Structure
```
project_root/
├── llm_pipe/
│   ├── Ingestion_Retrieval/
│   │   ├── pinecone_integration.py
│   │   ├── pinecone_retrieval.py
│   │   ├── document_chunker.py
│   │   ├── pdf_parser.py
│   │   ├── youtube_qa_agent.py
│   │   ├── __init__.py
│   │   └── .env
│   ├── gradio_app.py
│   ├── multi_agent.py
│   ├── multi_agent_tools.py
│   └── sql_agent.py
├── jsons/                 # Image citation mappings
├── pdfs/                  # Company PDF documents
├── page_images/          # Extracted page images
├── reports/              # Generated reports
├── notebooks/            # Jupyter notebooks
├── Flow diagram/         # System architecture diagrams
├── config.py             # Application configuration
├── requirements.txt      # Core dependencies
├── full-requirements.txt # All dependencies
├── docker-compose.yml    # Docker configuration
└── Dockerfile           # Docker build file
```

## 📖 Usage Guide

### Running the System

1. **First, ingest documents into Pinecone**:
```bash
python -m llm_pipe.Ingestion_Retrieval.pinecone_integration
```

2. **Then, start the Gradio interface**:
```bash
python -m llm_pipe.gradio_app
```

### 🤖 Multi-Agent Assistant
1. **Ask Questions**: Type questions about companies, YouTube content, or general topics
2. **View Routing**: Check the "Agent's Thought Process" accordion for debug info
3. **Clear Chat**: Use the clear button to reset conversation history
4. **Example Queries**: Use provided examples as templates

### 📄 Document QA System
1. **Select Company**: Choose from EnerSys, Apple, or NVIDIA documents
2. **Enter Query**: Ask specific questions about financial data or business operations
3. **View Results**: Get AI-generated answers with page citations
4. **Visual Citations**: Review source pages in the gallery
5. **Audio Playback**: Listen to the generated response

### 🎥 YouTube QA Agent
1. **Enter URL**: Paste YouTube video URL
2. **Select Model**: Choose appropriate Ollama model for your use case
3. **Fetch Transcript**: Extract video transcript
4. **Ask Questions**: Query specific aspects of the video content
5. **Summarize**: Generate concise video summaries

### 📊 SQL Agent
1. **Upload CSV**: Select CSV file (supports multiple encodings)
2. **Review Columns**: Check available columns and data types
3. **Configure Model**: Enable/disable Ollama integration
4. **Natural Query**: Ask questions in plain English
5. **View Results**: Examine generated SQL and query results

## 🔌 API Integration

### Ollama API Endpoints
- **Health Check**: `GET http://localhost:11434/tags`
- **Generate**: `POST http://localhost:11434/generate`
- **Model List**: `GET http://localhost:11434/tags`

### Key Functions
```python
# Document QA
process_query_and_generate(company, query)

# YouTube processing
handle_url_submit(url)
answer_question(question, url, model)
summarize_transcript(url, model)

# SQL processing
process_query_sql(csv_file, query, model, use_ollama)

# Multi-agent routing
invoke(message, history)
```

## 🐛 Troubleshooting

### Common Issues

#### Ollama Connection Problems
```bash
# Check Ollama status
curl http://localhost:11434/tags

# Restart Ollama service
ollama serve
```

**Solution**: Ensure Ollama is running and accessible on port 11434

#### Pinecone Authentication
```
Error: Pinecone API key not found
```
**Solution**: Verify `.env` file contains valid `PINECONE_API_KEY`

#### CSV Processing Issues
```
UnicodeDecodeError: 'utf-8' codec can't decode
```
**Solution**: System automatically tries multiple encodings (latin-1, cp1252, iso-8859-1)

#### Model Availability
```
Warning: phi3:mini not found in Ollama
```
**Solution**: Install required models using `ollama pull phi3:mini`

#### Memory Issues
```
CUDA out of memory
```
**Solution**: Use smaller models or CPU-based inference

### Performance Optimization

#### Model Selection Guidelines
- **phi3:mini**: Best for speed and efficiency (recommended default)
- **mistral**: Good balance of speed and accuracy
- **llama3**: Higher accuracy, slower response
- **gemma**: Google's optimized model

#### Vector Database Optimization
- Use appropriate index dimensions for your embedding model
- Implement batch processing for large document sets
- Regular index maintenance and optimization

#### Query Performance
- Implement query result caching where possible
- Use connection pooling for database operations
- Optimize chunk sizes for document retrieval

## 🚀 Future Enhancements

### Planned Features
- **Multi-language Support**: Support for non-English documents and queries
- **Advanced Analytics**: Query performance metrics and usage analytics
- **Batch Processing**: Handle multiple files simultaneously
- **API Integration**: RESTful API for external applications
- **User Management**: Authentication and session management

### Technical Improvements
- **Caching Layer**: Redis integration for faster response times
- **Model Fine-tuning**: Domain-specific model adaptations
- **Hybrid Search**: Combine semantic and keyword search
- **Container Deployment**: Docker and Kubernetes support
- **Monitoring**: Health checks and performance monitoring

### UI/UX Enhancements
- **Dark Mode**: Theme switching capabilities
- **Mobile Responsiveness**: Better mobile device support
- **Export Functions**: Save results to various formats
- **Advanced Filters**: Query filtering and sorting options

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4-core processor
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Network**: Internet connection for Pinecone and YouTube APIs

### Recommended Requirements
- **CPU**: 8-core processor with GPU support
- **RAM**: 16GB or higher
- **Storage**: SSD with 20GB+ free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster model inference)

---

*This documentation provides a comprehensive overview of the Multi-Modal RAG system. For specific implementation details and advanced configuration options, refer to the source code and configuration files.*
