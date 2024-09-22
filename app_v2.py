import os
import streamlit as st
from langchain.agents import Tool
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.tools import DuckDuckGoSearchRun

# Set up environment variables
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Create a custom tool that doesn't require human input
class AutoResponseTool:
    def run(self, query):
        return "Proceed with the available information."

@st.cache_resource
def initialize_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings()

@st.cache_data
def create_faiss_index():
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
    loaded_vectorstore = load_faiss_index()
    if loaded_vectorstore is None:
        return "Unable to retrieve examples due to indexing issues."
    relevant_docs = loaded_vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in relevant_docs])

def main():
    st.title("LinkedIn Financial Post Generator")

    finance_topic = st.text_input("Enter the finance topic you want to research and write about:", 
                                  "How to invest in real estate in India")

    if st.button("Generate LinkedIn Post"):
        with st.spinner("Researching and writing your LinkedIn post..."):
            # Initialize tools and agents
            llm = initialize_llm()
            search_tool = DuckDuckGoSearchRun()
            auto_response_tool = AutoResponseTool()

            researcher = {
                "role": 'Financial Researcher',
                "goal": f'Research the given finance topic: "{finance_topic}".',
                "backstory": """You are a knowledgeable financial researcher with expertise in various financial topics.
                You excel at gathering accurate and comprehensive information on finance-related subjects, including specific markets and investment strategies.""",
                "tools": [
                    Tool(
                        name="Internet Search",
                        func=search_tool.run,
                        description="Useful for finding the latest information on finance and investment strategies."
                    ),
                    Tool(
                        name="Auto Response",
                        func=auto_response_tool.run,
                        description="Provides an automatic response to continue with available information."
                    )
                ],
            }

            writer = {
                "role": 'LinkedIn Financial Influencer',
                "goal": f'Write a short, engaging, and easy-to-understand LinkedIn post on {finance_topic}.',
                "backstory": """You are a popular financial influencer on LinkedIn, known for explaining complex financial concepts in simple terms.
                Your posts are concise, use everyday language, and incorporate numbers and simple analogies to make financial topics accessible to everyone.""",
                "tools": [
                    Tool(
                        name="Auto Response",
                        func=auto_response_tool.run,
                        description="Provides an automatic response to continue with available information."
                    )
                ],
            }

            research_task = f"""Conduct thorough research on the topic: "{finance_topic}".
            Focus on key aspects such as:
            1. Current market trends
            2. Investment strategies specific to this topic
            3. Potential risks and rewards
            4. Legal or regulatory considerations
            5. Expert opinions and case studies
            6. Relevant statistics and data

            Provide a comprehensive research report on the given finance topic, including actionable insights and specific data points."""

            write_post_task = f"""Based on the research, create an engaging LinkedIn post about {finance_topic}.
            Your post should closely mimic the style, tone, and format of the following examples:

            {retrieve_relevant_examples(finance_topic)}

            Specifically, your post should:
            1. Start with a bold, attention-grabbing statement or question related to {finance_topic}.
            2. Use short, punchy paragraphs, often just one or two sentences each.
            3. Include 2-3 relevant statistics or numbers, presented in a memorable way (e.g., "125% returns").
            4. Incorporate bullet points or numbered lists for easy readability.
            5. Use some text formatting like **bold** or CAPS for emphasis (but don't overdo it).
            6. Include a simple, relatable analogy or real-life example to explain a complex concept.
            7. Provide one or two actionable tips or insights.
            8. End with a thought-provoking question or call-to-action.
            9. Keep the overall tone conversational and slightly provocative, as if you're challenging conventional wisdom.
            10. Aim for around 200-300 words total.
            11. Don't use emojis keep it simple, formal and clean
            12. End with a question to encourage engagement
            13. The language should be human-like, not AI-generated

            Remember, the goal is to sound like a knowledgeable friend giving advice, not a textbook or formal report. Make it engaging, informative, and easy to read quickly on a LinkedIn feed."""

            research_result = llm.predict(f"As a financial researcher, {research_task}")
            linkedin_post = llm.predict(f"As a LinkedIn financial influencer, use this research to {write_post_task}\n\nResearch:\n{research_result}")

        st.subheader("Research Result:")
        st.text_area("Research", research_result, height=300)

        st.subheader("LinkedIn Post:")
        st.text_area("Post", linkedin_post, height=300)

if __name__ == "__main__":
    main()