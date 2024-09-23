import streamlit as st
from linkedin_post_generator import generate_linkedin_post

def main():
    """Main function to run the Streamlit app."""
    st.title("LinkedIn Financial Post Generator")

    finance_topic = st.text_input("Enter the finance topic you want to research and write about:", 
                                  "How to invest in stocks")

    if st.button("Generate LinkedIn Post"):
        with st.spinner("Generating LinkedIn post..."):
            research_result, linkedin_post = generate_linkedin_post(finance_topic)
        
        # Display research results
        st.subheader("Research Result:")
        st.text_area("Research", research_result, height=300)

        # Display LinkedIn post
        st.subheader("LinkedIn Post:")
        st.text_area("Post", linkedin_post, height=300)

if __name__ == "__main__":
    main()