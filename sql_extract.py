# Load API key
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter 
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Define Base Directory for Robust Path Handling
BASE_DIR = Path.cwd()
CHROMA_STORE_PATH = BASE_DIR / "chroma_db" / "store_db"

# Caching variables
_db_instance = None
_few_shot_prompt = None

# Function to connect to database
def connect_to_db():
    global _db_instance
    if _db_instance is not None:
        return _db_instance
    
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")  
    db_name = os.getenv("DB_NAME")

    from langchain_community.utilities.sql_database import SQLDatabase
    _db_instance = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    return _db_instance

# Function to get example few shot prompts
def get_examples():
    examples = [
    {
    "input": "Show me all available products.",
    "query": "SELECT * FROM product WHERE status =1;"
    },
    {
    "input": "What products do you have in the Earrings category?",
    "query": "SELECT name, price, stock_quantity FROM product WHERE category = 'Earrings' AND status = 1;"
    },
    {
    "input": "Do you have product Cherry danglers in stock?",
    "query": "SELECT name, stock_quantity FROM product WHERE name = ‘Cherry danglers’ AND stock_quantity > 0 AND status = 1;"
    },
    {
    "input": "Show me products under Rs.200.",
    "query": "SELECT name, category, price FROM product WHERE price < 200 AND status = 1;"
    },
    {
    "input": "What is the price of the product named 'Rose-white pearl'?",
    "query": "SELECT price FROM product WHERE name = 'Rose-white pearl';"
    },
    {
        "input": "Find my account details, my email is upad@gmail.com.",
        "query": "SELECT first_name, last_name, email, phone, address FROM customer WHERE email = 'upad@gmail.com';"
    },
    {
        "input": "Is my account active?  My email is upad@gmail.com",
        "query": "SELECT status FROM customer WHERE email = 'upad@gmail.com';"
    },
    {
        "input": "What is the status of my order? My order number is 3.",
        "query": "SELECT order_status, order_date, total_amount FROM order_details WHERE id = 3;"
    },
    {
        "input": "How many products are out of stock?",
        "query": "SELECT COUNT(*) as out_of_stock_count FROM product WHERE stock_quantity = 0;"
    },
    {
        "input": "What categories do you have?",
        "query": "SELECT DISTINCT category FROM product WHERE status = 1;"
    }
    ]

    # Structure the few shot prompt
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}\nSQLQuery:"),
            ("ai", "{query}"),
        ]
    )
    
    return examples, example_prompt

# Function to ingest few shot examples
def ingest_examples():
    print("SQL Example DB not found. creating index...")
    examples, _ = get_examples()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # We use SemanticSimilarityExampleSelector to ingest
    SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        Chroma,
        k=4,
        input_keys=["input"],
        vectorstore_kwargs={
            "collection_name": "store_db",
            "persist_directory": str(CHROMA_STORE_PATH),
            "embedding_function": embeddings
        }
    )
    print("SQL Example DB created")
    
# Function to dynamically select relevant examples as few shot prompt
def get_few_shot_prompt_template():
    global _few_shot_prompt
    if _few_shot_prompt is not None:
        return _few_shot_prompt
    
    # Check/Create DB
    if not CHROMA_STORE_PATH.exists():
        ingest_examples()
        
    _, example_prompt = get_examples()
    
    embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2" 
        ) 

    vector_db = Chroma(
        collection_name="store_db",
        persist_directory=str(CHROMA_STORE_PATH),
        embedding_function=embeddings,
    )
    
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_db,
        k=2,
        input_keys=["input"],
    )
    
    _few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input", "top_k"],
    )
    
    return _few_shot_prompt

# Public function to call from Router
def query_db(question: str):
    db = connect_to_db()
    few_shot_prompt = get_few_shot_prompt_template()
    
    sql_query_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", """Given an input question, create a syntactically correct SQL query.

        Use the following table information:
        {table_info}

        IMPORTANT: Output ONLY the SQL query without any prefixes, explanations, or markdown formatting.
        Do NOT include 'SQLQuery:', '``````', or any other text.
        
        Below are a number of examples of questions and their corresponding SQL queries for your reference.
        """),
        few_shot_prompt,
        ("human", 
        "Question:\n{{question}}\n\n Answer:"),
        ]
    )
    
    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and the result obtained from the query. Reframe the result obtained from the query into a natural language form and answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        temperature=0,
        google_api_key=api_key 
    )

    generate_query = create_sql_query_chain(llm, db, prompt=sql_query_prompt)
    chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | QuerySQLDatabaseTool(db=db)
    )
    | answer_prompt | llm | StrOutputParser()
    )
    
    return chain.invoke({"question": question})

if __name__ == "__main__":
    # Test locally
    print(query_db("How many categories do you have?"))