from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Request
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag import init_llm, init_prompt, get_answer, index_document
from pydantic import BaseModel

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

embedding= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class Question(BaseModel):
    question: str
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    db= Chroma(persist_directory="./chroma_db",embedding_function=embedding)
    
    app.state.vector_db = db 
    app.state.llm = init_llm(api_key= api_key)   
    app.state.prompt = init_prompt()
    
    yield
    
    
app = FastAPI(lifespan= lifespan)


@app.get('/')
def health_status():
    return {'status': 'ok'}

@app.post("/upload")
async def create_upload_file(request: Request, file: UploadFile):

    contents = await file.read()
    with open(f"temp_{file.filename}", "wb") as f:
        f.write(contents)
    vectorize_docs = index_document(f"temp_{file.filename}")
    request.app.state.vector_db = vectorize_docs
    return {"message": "success"}
    
        
@app.post("/ask")
async def receive_answers(question: Question, request: Request):
    question = question.question
    
    if request.app.state.vector_db is None:
        return {"error": "No document uploaded yet. Please use /upload first."}

    response = get_answer(
        vector_db= request.app.state.vector_db,
        llm= request.app.state.llm,
        prompt= request.app.state.prompt,
        question= question
    )
    return {"Bot_response": response}