# DeepReader

A semantic document Q&A API — upload a PDF and ask questions about it.
The API retrieves relevant context from your document and generates
grounded answers using an LLM.

## What it does

- Upload any PDF document via `/upload`
- Ask questions about the document via `/ask`
- Answers are generated using retrieved context, not hallucination

## Stack

- Python, FastAPI
- LangChain, ChromaDB
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- OpenRouter API (LLM)
- Docker, Render

## Endpoints

- `POST /upload` — accepts a PDF, chunks and embeds it into ChromaDB
- `POST /ask` — accepts a question, retrieves relevant chunks, returns LLM answer

## Run locally

```bash
git clone https://github.com/abubakkersiddiqq/deep-reader
cd deep-reader
cp .env.example .env  # add your OpenRouter API key
docker build -t deep-reader .
docker run -p 8000:8000 deep-reader
```

## Test the API

Visit `https://deepreader.onrender.com/docs` for interactive Swagger UI.

> 🚧 **Note:** The live link may be down. This project is hosted on Render's
> free tier (750hr/month limit). If the demo doesn't load, it's likely
> suspended. Feel free to run it locally using the instructions below.

## Live

[🔗 DeepReader on Render](https://deepreader.onrender.com)

> ⚠️ First request may take 30–60 seconds (free tier cold start)
