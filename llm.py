import streamlit as st
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

@st.cache_resource
def initialize_llm():
    """Initialize and cache the Groq language model."""
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")