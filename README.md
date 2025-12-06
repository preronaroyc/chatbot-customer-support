## Customer Support Chatbot

A smart customer support chat assistant for Jazzy Things, an online small business jewelry store. This chatbot intelligently routes user queries to either a RAG pipeline and a Text-to-SQL pipeline depending on the user query.

### Features

- Intelligent Routing: Uses semantic similarity to determine if a user is asking about general policies or specific database data.
- RAG (Retrieval-Augmented Generation): Answers policy questions using a PDF knowledge base.
- Text-to-SQL: Converts natural language questions into SQL queries to fetch real-time data from a MySQL database.
- Streamlit UI: A clean, chat-based interface for easy interaction.
- Docker: For easy deployment.

### Tech Stack

- Frontend: Streamlit
- LLM Integration: LangChain, Gemini 2.5 Flash
- Vector Database: ChromaDB
- Embeddings: HuggingFace Embedding Model ("all-MiniLM-L6-v2")
- Database: MySQL
- Routing: Semantic Router & Cosine Similarity Router
