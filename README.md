# DeepReader

A semantic document Q&A API - upload a PDF and ask questions about it.
The API retrieves relevant context from your document and generates
grounded answers using an LLM.

## What it does

- Upload any PDF document via `/upload`
- Ask questions about the document via `/ask`
- Answers are generated using retrieved context, not hallucination

## Stack

- Python, FastAPI
- LangChain, ChromaDB
- FastEmbed Embeddings (BAAI/bge-small-en-v1.5)
- OpenRouter API (LLM)
- Docker, Hugging Face Spaces

## Endpoints

- `POST /upload` - accepts a PDF, chunks and embeds it into ChromaDB
- `POST /ask` - accepts a question, retrieves relevant chunks, returns LLM answer

## Live

[DeepReader on Hugging Face Spaces](https://abubakker66-deepreader.hf.space)

No frontend - interact with the API directly via Swagger UI at:
`https://abubakker66-deepreader.hf.space/docs`

## Run locally

```bash
git clone https://github.com/abubakkersiddiqq/deep-reader
cd deep-reader
cp .env.example .env  # add your OpenRouter API key
docker build -t deep-reader .
docker run -p 8000:8000 deep-reader
```

Local Swagger UI available at: `http://localhost:8000/docs`
