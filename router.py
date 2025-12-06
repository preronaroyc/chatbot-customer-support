# Import necessary libraries
import os
from dotenv import load_dotenv
import numpy as np
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.index.local import LocalIndex

# Import downstream pipelines
from rag import query_rag
from sql_extract import query_db

load_dotenv()  
api_key = os.getenv("GEMINI_API_KEY") 


# Define routes with example utterances
faq_route = Route(
    name="faq",
    utterances=[
    "Can I place an order through Instagram or WhatsApp instead of the website?",
    "Do you ship internationally or only within India?",
    "What are the standard shipping charges for an order?",
    "Do you offer free shipping above a certain order amount?",
    "Within how many days can I request a return or exchange?",
    "What photos or proof do I need to provide to request a return?",
    "Are earrings eligible for return or exchange?",
    "Are sale or clearance items returnable if there is a problem?",
    "How long does it take to get my refund after returning an item?",
    "What are your working days and hours for customer support?",
    "How do I care for my jewelry?",
    "Are your products handmade or machine made?"
]
)

database_route = Route(
    name="database",
    utterances=[
        "Show me earrings under 500",
        "What  necklaces are available?",
        "Do you have  bracelets in stock?",
        "What is the price of 'Cherry danglers' earrings?",
        "List all necklaces",
        "What is the minimum price of a necklace?",
        "What categories do you have in your collection?",
        "Find necklaces with price more than 700",
        "Do you have bangles?",
        "Is my order shipped?",
        "How many earrings are in stock?",
        "What is the status of my order?"  
    ]
)

ALL_ROUTES = [faq_route, database_route]

print("Initializing Router Encoder...")
# Using a thread-safe configuration for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

try:
    print("Initializing Semantic Router Index...")
    index = LocalIndex()
    semantic_router = SemanticRouter(
        encoder=encoder,
        routes=ALL_ROUTES,
        index=index,
    )
    # Warmup query to check if index is truly ready
    semantic_router("test")
    ROUTER_MODE = "LIBRARY"
    print("Semantic Router initialized successfully.")

except Exception as e:
    print(f"Semantic Router library failed ({e}). Switching to Manual Fallback Router.")
    ROUTER_MODE = "MANUAL"
    
    # Pre-compute embeddings for manual routing to ensure speed
    print("Pre-computing route embeddings for fallback...")
    route_embeddings = {}
    for route in ALL_ROUTES:
        embeddings = encoder(route.utterances)
        route_embeddings[route.name] = np.array(embeddings)
    print("Manual Fallback Router ready")

# Function to manually calculate cosine similarity if the library index fails
def get_manual_route(query: str, threshold: float = 0.3):
    query_emb = np.array(encoder([query])[0])
    
    best_score = -1
    best_route_name = None

    for route_name, embeddings in route_embeddings.items():
        # Calculate cosine similarity: (A . B) / (|A| * |B|)
        # Since embeddings are usually normalized, just dot product works often, 
        # but we do full cosine to be safe.
        scores = np.dot(embeddings, query_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        max_score = np.max(scores)
        
        if max_score > best_score:
            best_score = max_score
            best_route_name = route_name

    if best_score >= threshold:
        return best_route_name
    return None

# Function that calls the pipelines
def process_query(query: str):
    try:
        route_name = None
        
        # Attempt Routing
        if ROUTER_MODE == "LIBRARY":
            try:
                choice = semantic_router(query)
                if choice and choice.name:
                    route_name = choice.name
            except Exception:
                # If library crashes mid-query, retry with manual
                route_name = get_manual_route(query)
        else:
            route_name = get_manual_route(query)

        # Dispatch
        if route_name == "database":
            print(f"Routing to DATABASE: {query}")
            return query_db(query)
        
        elif route_name == "faq":
            print(f"Routing to FAQ: {query}")
            return query_rag(query)
            
        else:
            print(f"Routing to Default (FAQ): {query}")
            return query_rag(query)
            
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"
    
if __name__ == "__main__":
    # Test routing
    print(process_query("What is the price of Cherry danglers?"))
    print(process_query("How do I return an item?"))

