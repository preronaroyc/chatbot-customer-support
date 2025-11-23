# Import necessary libraries
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
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

# Function to load pdf and split into chunks
def load_and_split(pdf_path: str):
    loader = PyPDFLoader(pdf_path)  
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          
        chunk_overlap=120,       
        separators=["\n\n", "\n", ".", "?", "!", " "]  
    )
    chunks = splitter.split_documents(docs)

    return chunks

# Function to update the vector store with embeddings
def ingest(pdf_path: str, reset: bool):

    chunks = load_and_split(pdf_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2" ,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    if reset:
        db_dir = "./chroma_db/FAQ"
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)

    vector_db = Chroma(
        collection_name="FAQ",
        persist_directory="./chroma_db/FAQ" ,
        embedding_function=embeddings,
    )
    
    vector_db.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks into vector db")

# Function to build retrieval and chat chain
def build_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2" ,
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_db = Chroma(
        collection_name="FAQ",
        persist_directory="./chroma_db/FAQ" ,
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
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# function to ask a question
def ask_question(question: str):
    chain = build_rag_chain()
    return chain.invoke(question,
                        config={
            "project_name": "jazzy-things-rag"
            })

# Main function
def main():
    pdf_path = "./documents/FAQ.pdf"
    if Path(pdf_path).exists():
        ingest(pdf_path, reset = True)
    else:
        print(f"WARNING: '{pdf_path}' not found. Skipping ingestion. Place your FAQ PDF and rerun.")

    while True:
        q = input("Type your customer question (e.g., 'What are your shipping charges?'), or 'exit' to quit.").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            answer = ask_question(q)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()


