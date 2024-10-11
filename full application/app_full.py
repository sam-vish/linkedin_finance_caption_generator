import os
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

langsmith_key = os.getenv('LANGSMITH_API_KEY')

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LinkedinCaptionGenerator"

# Set up environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

class AutoResponseTool:
    def run(self, query):
        return "Proceed with the available information."

@st.cache_resource
def initialize_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.2-90b-text-preview")

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings()

@st.cache_resource
def load_or_create_faiss_index():
    """Load existing FAISS index or create a new one if it doesn't exist."""
    embeddings = initialize_embeddings()
    
    try:
        # Try to load existing index
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.success("Loaded existing FAISS index.")
        return vectorstore
    except (FileNotFoundError, RuntimeError):
        try:
            # Create new index if loading fails
            loader = TextLoader("LinkedIn_Posts.md")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local("faiss_index")
            st.success("Created and saved new FAISS index.")
            return vectorstore
        except Exception as e:
            st.error(f"Error creating FAISS index: {e}")
            return None

@st.cache_data
def retrieve_relevant_examples(_query, k=2):
    """Retrieve relevant examples from the FAISS index."""
    vectorstore = load_or_create_faiss_index()
    if vectorstore is None:
        return "Unable to retrieve examples due to indexing issues."
    relevant_docs = vectorstore.similarity_search(_query, k=k)
    return "\n\n".join([doc.page_content for doc in relevant_docs])

def main():
    """Main function to run the Streamlit app."""
    st.title("LinkedIn Financial Post Generator")

    finance_topic = st.text_input("Enter the finance topic you want to research and write about:", 
                                  "How to invest in Tata motors")

    if st.button("Generate LinkedIn Post"):
        # Initialize tools and agents
        llm = initialize_llm()
        search_tool = DuckDuckGoSearchRun()
        auto_response_tool = AutoResponseTool()

        tools = [
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
        ]

        # Get the prompt to use
        prompt = hub.pull("hwchase17/react")

        # Create the ReAct agent
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Research task
        research_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""You are a knowledgeable financial researcher with expertise in various financial topics.
            You excel at gathering accurate and comprehensive information on finance-related subjects, including specific markets and investment strategies.

            Conduct thorough research on the topic: "{topic}".
            Focus on key aspects such as:
            1. Current market trends
            2. Investment strategies specific to this topic
            3. Potential risks and rewards
            4. Legal or regulatory considerations
            5. Expert opinions and case studies
            6. Relevant statistics and data

            Provide a comprehensive research report on the given finance topic, including actionable insights and specific data points."""
        )

        research_chain = research_prompt | llm | StrOutputParser()

        # Writing task
        write_prompt = PromptTemplate(
            input_variables=["topic", "research", "examples"],
            template="""You are a popular financial influencer on LinkedIn, known for explaining complex financial concepts in simple terms.
            Your posts are concise, use everyday language, and incorporate numbers and simple analogies to make financial topics accessible to everyone.

            Based on the research, create an engaging LinkedIn post about {topic}.
            Your post MUST closely mimic the style, tone, and format of the following examples. Pay careful attention to these examples and ensure your post matches their format exactly:

            {examples}

            Your post MUST adhere to these strict guidelines:
            1. Start with a bold, attention-grabbing statement or question related to {topic}.
            2. Use short, punchy paragraphs, often just one or two sentences each.
            3. Include 2-3 relevant statistics or numbers, presented in a memorable way (e.g., "125% returns").
            4. Incorporate bullet points or numbered lists for easy readability.
            5. Use some text formatting like **bold** for emphasis (but don't overdo it).
            6. Include a simple, relatable analogy or real-life example to explain a complex concept.
            7. Provide one or two actionable tips or insights.
            8. End with a thought-provoking question or call-to-action.
            9. Keep the overall tone conversational and slightly provocative.
            10. Aim for around 200-300 words total.
            11. DO NOT use any emojis. Keep it simple, formal, and clean.
            12. The language should be human-like, not AI-generated.

            Research:
            {research}

            Remember, the goal is to sound like a knowledgeable friend giving advice, not a textbook or formal report. Make it engaging, informative, and easy to read quickly on a LinkedIn feed."""
        )

        write_chain = write_prompt | llm | StrOutputParser()

        with st.spinner("Researching and generating LinkedIn post..."):
            # Use the agent to perform research
            research_result = agent_executor.invoke({"input": f"Research the latest information on {finance_topic}"})
            
            # Retrieve relevant examples from FAISS index
            relevant_examples = retrieve_relevant_examples(finance_topic)

            # Generate the LinkedIn post using the research and relevant examples
            linkedin_post = write_chain.invoke({"topic": finance_topic, "research": research_result['output'], "examples": relevant_examples})

        # Display research results
        st.subheader("Research Result:")
        st.text_area("Research", research_result['output'], height=300)

        # Display LinkedIn post
        st.subheader("LinkedIn Post:")
        st.text_area("Post", linkedin_post, height=300)

if __name__ == "__main__":
    main()
