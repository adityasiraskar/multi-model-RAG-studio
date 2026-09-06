# Multi-Model RAG Studio

A Gradio application for question answering over four financial-report PDFs, with Pinecone retrieval, multi-agent routing, YouTube QA, CSV-to-SQL analysis, and a DeepEval regression suite.

## Current Architecture

The production code lives in `llm_pipe/`:

- `llm_pipe/gradio_app.py`: Gradio application and document-QA UI.
- `llm_pipe/Ingestion_Retrieval/pinecone_integration.py`: PDF parsing, chunking, embedding, and Pinecone ingestion.
- `llm_pipe/Ingestion_Retrieval/pinecone_retrieval.py`: namespace search and page-aware result formatting.
- `llm_pipe/multi_agent.py`: conversational routing.
- `llm_pipe/multi_agent_tools.py`: document, YouTube, and web tools.
- `llm_pipe/sql_agent.py`: CSV upload and natural-language SQL.
- `llm_pipe/document_qa.py`: shared OpenAI document-answer generator.

The evaluation adapters live in `src/`:

- `src/reranker.py`: exposes the existing Pinecone retrieval as evaluator-compatible documents.
- `src/rag_pipeline.py`: shared `retrieve -> generate` pipeline contract.
- `src/generator.py`: adapter used by the evaluation suite.

The evaluation suite lives in `evals/` and reads datasets from `golden/`.

## Four-PDF Corpus

Only these PDFs are supported and ingested:

| Namespace | File |
|---|---|
| `amazon` | `pdfs/Amazon10k2022.pdf` |
| `apple` | `pdfs/Apple_10-K-2021.pdf` |
| `nvidia` | `pdfs/Nvidia.pdf` |
| `tesla` | `pdfs/Tesla.pdf` |

The mapping is defined once in `config.py` and reused by ingestion, retrieval, routing, and the UI.

## Model Roles

The current code uses:

- **OpenAI**: document-QA generation and DeepEval judging, configured with `OPENAI_API_KEY` and `OPENAI_MODEL`.
- **Ollama**: multi-agent routing, SQL generation when enabled, YouTube question answering, and YouTube summarization. The default model is `phi3:mini`.
- **Pinecone**: vector storage and retrieval.
- **Sentence Transformers**: local embeddings using `all-mpnet-base-v2`.

Do not put API keys in Python files. API keys belong only in the untracked `.env` file.

## Requirements

- Windows PowerShell or a compatible shell.
- Python 3.13 recommended.
- An Ollama installation for agent, SQL, and YouTube features.
- A Pinecone account and API key for PDF ingestion and retrieval.
- An OpenAI API key for document generation and DeepEval judging.
- Internet access for package installation, model download, Pinecone, OpenAI, and YouTube transcript requests.

All Python dependencies are in `requirements.txt`. There is no separate `full-requirements.txt`.

## Setup on Windows

Run commands from the repository root:

```powershell
cd D:\sabudh_project\rag_with_eval\multi-model-RAG-studio
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Create `.env` beside `config.py`:

```env
PINECONE_API_KEY=replace_with_your_pinecone_key
OPENAI_API_KEY=replace_with_your_openai_key
OPENAI_MODEL=gpt-4o-mini
TAVILY_API_KEY=replace_if_web_search_is_used
OLLAMA_API_BASE=http://localhost:11434/api
OLLAMA_MODEL=phi3:mini
```

Never commit `.env`. Rotate any key that has previously been pasted into source code, chat, logs, or screenshots.

## Start Ollama

Check whether Ollama is already running:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/tags
```

If it is not running, start it once in a separate terminal:

```powershell
ollama pull phi3:mini
ollama serve
```

If `ollama serve` reports that port `11434` is already in use, Ollama is already running. Do not start a second instance.

## Ingest the PDFs

Ingestion creates or uses the Pinecone index configured as `document-analysis`, and creates these namespaces:

- `amazon`
- `apple`
- `nvidia`
- `tesla`

Run:

```powershell
python -m llm_pipe.Ingestion_Retrieval.pinecone_integration
```

The first run downloads and caches the embedding model under `models/embedding_model`. It also creates page-image and image-map artifacts used by the UI. Re-run ingestion when a PDF or chunking configuration changes.

## Run the Gradio Application

```powershell
python -m llm_pipe.gradio_app
```

The UI provides four tabs:

1. **Multi-Agent Assistant**: routes company, YouTube, and general queries.
2. **Document QA**: answers questions about Amazon, Apple, NVIDIA, and Tesla with page citations and optional audio.
3. **YouTube QA**: fetches transcripts, answers questions, and summarizes videos.
4. **SQL Agent**: uploads CSV files, generates SQL with Ollama or fallback rules, and executes SQLite queries.

## Run the Evaluation Suite

The suite builds one shared pipeline and evaluates retrieval, generation, end-to-end RAG quality, application quality, safety, latency, cost estimates, and reliability.

First verify the environment:

```powershell
python -m compileall -q llm_pipe src evals
python -c "import deepeval, langchain, openai; print('evaluation dependencies: ok')"
python -m evals.run_suite --help
```

Run a candidate snapshot:

```powershell
python -m evals.run_suite --out baselines/candidate.json --label "four-pdf-current" --quiet
```

Create a baseline snapshot:

```powershell
python -m evals.run_suite --baseline --label "baseline" --quiet
```

Compare baseline and candidate:

```powershell
python -m evals.compare `
  --baseline baselines/baseline.json `
  --candidate baselines/candidate.json
```

DeepEval uses OpenAI as the judge. The evaluation run therefore requires a valid `OPENAI_API_KEY`, and judge calls may incur OpenAI charges. Pinecone and the configured application generator must also be available.

## Golden Datasets

The `golden/` directory contains the evaluation inputs:

- `correctness_golden.json`
- `faithfullness_golden.json`
- `retrievel_golden.json`
- `scope_golden.json`
- `leakege_golden.json`
- `toxicity_golden.json`

The filenames retain their existing spelling for compatibility with the evaluation modules.

## Useful Entry Points

```powershell
# Ingest PDFs
python -m llm_pipe.Ingestion_Retrieval.pinecone_integration

# Run application
python -m llm_pipe.gradio_app

# Run individual evaluations
python -m evals.eval_retriever
python -m evals.eval_generator
python -m evals.eval_rag_pipeline
python -m evals.eval_application
python -m evals.eval_safety
python -m evals.eval_ops

# Compare snapshots
python -m evals.compare --baseline baselines/baseline.json --candidate baselines/candidate.json
```

## Troubleshooting

### `ModuleNotFoundError: deepeval`

Use the project interpreter explicitly:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import deepeval; print('deepeval: ok')"
```

### Dependency conflicts or a broken virtual environment

Recreate the environment. This does not modify source files:

```powershell
Remove-Item .\.venv -Recurse -Force
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### OpenAI HTTP 401

Confirm that `.env` contains a valid OpenAI key beginning with `sk-`, and make sure an old `OPENAI_API_KEY` process environment variable is not overriding it. Restart the terminal after changing environment variables.

### Pinecone authentication failure

Check `PINECONE_API_KEY` in `.env`, confirm the Pinecone index name is `document-analysis`, and run ingestion before querying documents.

### Ollama port conflict

A bind error on `127.0.0.1:11434` usually means Ollama is already running. Check it with:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/tags
```

### No relevant retrieval results

Confirm that ingestion completed for all four namespaces and that the company name appears in the query. The evaluation retriever maps queries to `amazon`, `apple`, `nvidia`, or `tesla`.

## Run with Docker

Docker uses the same four-PDF application entrypoint as local execution. The
container listens on port `7860`; Ollama and Pinecone remain external services.

1. Make sure `.env` exists in the repository root and contains valid
   `PINECONE_API_KEY` and `OPENAI_API_KEY` values.
2. Build the image:

```powershell
docker build -t multi-model-rag-studio:latest .
```

3. Run the container:

```powershell
docker run --rm --name multi-model-rag-studio `
  --env-file .env `
  -p 7860:7860 `
  multi-model-rag-studio:latest
```

Open `http://localhost:7860` in a browser. The container does not start Ollama;
the application must be able to reach the configured `OLLAMA_API_BASE`. On
Windows, host services may require `http://host.docker.internal:11434/api`
instead of `http://localhost:11434/api` inside the container.

Alternatively, use Compose:

```powershell
docker compose up --build
```

Stop it with:

```powershell
docker compose down
```

The `.dockerignore` file excludes `.env`, virtual environments, generated
artifacts, and local caches from the image build context.

## Security

- Keep `.env` out of version control.
- Do not hard-code Pinecone, OpenAI, Tavily, or other credentials.
- Rotate credentials that have been exposed.
- Review generated snapshots before sharing them because evaluation logs can contain prompts, model outputs, and metadata.
