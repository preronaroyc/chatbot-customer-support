# Import necessary librariesimport os
import os
from dotenv import load_dotenv
import shutil
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser  
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  
api_key = os.getenv("GEMINI_API_KEY")

# Define Base Directory for Robust Path Handling
BASE_DIR = Path.cwd()
CHROMA_PATH = BASE_DIR / "chroma_db" / "FAQ"
DOCUMENTS_PATH = BASE_DIR / "documents" / "FAQ.pdf"

# Global variable to cache the chain
_rag_chain = None              

# Function to the PDF, create embeddings, and save them to disk
def ingest_documents():
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(f"Source document not found at {DOCUMENTS_PATH}")

    print("Database not found. Starting ingestion (One-time process)...")
    
    loader = PyPDFLoader(str(DOCUMENTS_PATH))  
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          
        chunk_overlap=120,       
        separators=["\n\n", "\n", ".", "?", "!", " "]  
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
        
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="FAQ",
        persist_directory=str(CHROMA_PATH)
    )
    print(f"Ingestion complete. Database saved to {CHROMA_PATH}")


# Function to load the existing vector DB and build the chain
def get_rag_chain():
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    # Check if DB exists; if not, create it ONCE.
    if not CHROMA_PATH.exists():
        ingest_documents()
        
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_db = Chroma(
        collection_name="FAQ",
        persist_directory=str(CHROMA_PATH) ,
        embedding_function=embeddings,
    )

    retriever = vector_db.as_retriever(
        search_type="mmr",  
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )

    # prompts
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a customer support assistant for an India-based small business jewelry shop called Jazzy Things. "
                "Answer the user's question using ONLY the provided context. If the answer is not in the context, say you don't know and suggest to contact through instagram message. "
                "Be concise, accurate, and use clear, friendly language."
            ),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Answer:"
            ),
        ]
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=1024,
    )
    
    
    # Build the RAG graph: retrieve -> prompt -> LLM -> text
    _rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return _rag_chain


# Public function to call from Router
def query_rag(question: str):
    chain = get_rag_chain()
    return chain.invoke(question)

if __name__ == "__main__":
    # Test locally
    print(query_rag("What are your shipping charges?"))
