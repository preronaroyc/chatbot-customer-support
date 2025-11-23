# Load API key
import os
from dotenv import load_dotenv
from langchain_classic.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter 
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Function to connect to database
def connect_to_db():
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")  
    db_name = os.getenv("DB_NAME")

    from langchain_community.utilities.sql_database import SQLDatabase
    # db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=1,include_tables=['customers','orders'],custom_table_info={'customers':"customer"})
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    return db

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


# Function to dynamically select relevant examples as few shot prompt
def select_examples(examples, example_prompt):
    embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2" 
        )

    vector_db = Chroma(
            collection_name="store_db",
            persist_directory="./chroma_db/store_db" ,
            embedding_function=embeddings,
        )

    vector_db.delete_collection()

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        vector_db,
        k=2,
        input_keys=["input"],
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input","top_k"],
    )
    
    return few_shot_prompt

# Function to get answer
def get_answer(question, db, few_shot_prompt):
    
    final_prompt = ChatPromptTemplate.from_messages(
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
        "Question:\n{input}\n\n"
        "Answer:"),
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

    generate_query = create_sql_query_chain(llm, db, prompt = final_prompt)
    chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | QuerySQLDataBaseTool(db=db)
    )
    | answer_prompt | llm | StrOutputParser()
    )
    return chain.invoke(question)

# Main function
def main():
    # Connect to database
    db = connect_to_db()
    
    examples, example_prompt = get_examples()
    # Get dynamically selected few shot prompts
    few_shot_prompt = select_examples(examples, example_prompt)
    
    # Get output answer
    while True:
        q = input("Type your customer question (e.g., 'How many categories do you have to choose from?'), or 'exit' to quit.").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            answer = get_answer(q, db, few_shot_prompt)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
