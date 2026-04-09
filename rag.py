from collections import Counter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def load(file_path) -> list:
    
    def detect_repeated_lines(docs, threshold=0.6):
        """
        Lines appearing in more than `threshold` % of pages 
        are likely headers/footers
        """
        all_lines = []
        for doc in docs:
            lines = [line.strip() for line in doc.page_content.split('\n') 
                    if line.strip()]
            all_lines.extend(lines)
        
        total_pages = len(docs)
        line_counts = Counter(all_lines)
        
        # Flag lines appearing in 60%+ of pages
        repeated = {
            line for line, count in line_counts.items() 
            if count / total_pages >= threshold and len(line) > 3
        }
        
        return repeated

    def clean_docs(docs):
        repeated_lines = detect_repeated_lines(docs)
        
        print("Detected repeating lines (will be removed):")
        for line in repeated_lines:
            print(f"  → '{line}'")
        
        for doc in docs:
            lines = doc.page_content.split('\n')
            cleaned = [l for l in lines if l.strip() not in repeated_lines]
            doc.page_content = '\n'.join(cleaned)
        
        return docs
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    docs = clean_docs(docs)
    return docs

def chunk(docs: list) -> list:
    chunk_size=600
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.10) )
    chunks = splitter.split_documents(docs)
    return chunks

def vectorize(chunks: list):
    embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    return vector_db


def simple_retriever(vector_db, query):
    
    retriever = vector_db.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 4,
        'fetch_k': 25,       
        'lambda_mult': 0.5  
        }
    )

    return retriever
        
def init_llm(api_key):
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free"
    )
    return llm
    
def init_prompt():
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question between 4-5 sentences."
        "\n\n"
        "The output format should be in neat with break for each sentence"
        "User '▪' for each points"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt

# MAIN FUNCTIONS

def index_document(pdf_path):
    docs = load(file_path= pdf_path)
    chunks = chunk(docs= docs)
    vector_db = vectorize(chunks= chunks)
    
    return vector_db

def get_answer(vector_db, llm, prompt, question):
    
    retriever = vector_db.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 4,
        'fetch_k': 25,       
        'lambda_mult': 0.5  
        }
    )
    
    question_answer_chain = create_stuff_documents_chain(llm= llm, prompt= prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question})
    bot_response = response["answer"]
    return bot_response

