import streamlit as st
import time
from router import process_query

# Page Configuration
st.set_page_config(
    page_title="Customer Support Chat",
    layout="centered"
)

# Styling
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stTextInput input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("💎 Customer Support Chatbot")
st.write("Welcome to Jazzy Things Customer Support! How may we help you today?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call the router to get the response
            full_response = process_query(prompt)
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"I apologize, but I encountered an error: {e}"
            message_placeholder.markdown(full_response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})