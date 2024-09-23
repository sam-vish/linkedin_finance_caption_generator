import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

@st.cache_resource
def initialize_embeddings():
    """Initialize and cache the HuggingFace embeddings."""
    return HuggingFaceEmbeddings()

@st.cache_data
def create_faiss_index():
    """Create and save a FAISS index from the LinkedIn posts."""
    try:
        loader = TextLoader("LinkedIn_Posts.md")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, initialize_embeddings())
        vectorstore.save_local("faiss_index")
        st.success("FAISS index created and saved successfully.")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None

@st.cache_data
def load_faiss_index():
    """Load the FAISS index or create a new one if it doesn't exist."""
    try:
        return FAISS.load_local("faiss_index", initialize_embeddings(), allow_dangerous_deserialization=True)
    except FileNotFoundError:
        st.warning("FAISS index not found. Creating a new one...")
        return create_faiss_index()
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

@st.cache_data
def retrieve_relevant_examples(query, k=2):
    """Retrieve relevant examples from the FAISS index."""
    loaded_vectorstore = load_faiss_index()
    if loaded_vectorstore is None:
        return "Unable to retrieve examples due to indexing issues."
    relevant_docs = loaded_vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in relevant_docs])