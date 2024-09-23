# LinkedIn Financial Post Generator

## Overview
The LinkedIn Financial Post Generator is a Streamlit application designed to help users generate engaging LinkedIn posts on various finance topics. It utilizes a language model to conduct research and create posts that are informative and easy to understand.

## Features
- Input a finance topic to generate a LinkedIn post.
- Conducts thorough research on the specified topic.
- Generates a well-structured LinkedIn post following best practices for engagement.

## Requirements
- Python 3.7+
- Streamlit
- Langchain
- dotenv
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/linkedin-financial-post-generator.git
   cd linkedin-financial-post-generator
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory and add your `GROQ_API_KEY`:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run linkedin_agent/main.py
   ```

2. Open your browser and navigate to `http://localhost:8501`.

3. Enter a finance topic and click "Generate LinkedIn Post" to see the results.

## File Structure
linkedin_agent/
│
├── main.py # Main Streamlit application
├── linkedin_post_generator.py # Logic for generating LinkedIn posts
├── llm.py # Language model initialization
├── embeddings.py # Embedding and indexing logic
├── tools.py # Tool definitions for the application
├── config.py # Configuration and environment variable loading
└── .env # Environment variables (not included in version control)
## File Structure

```
linkedin_agent/
│
├── main.py                  # Main Streamlit application
├── linkedin_post_generator.py # Logic for generating LinkedIn posts
├── llm.py                   # Language model initialization
├── embeddings.py            # Embedding and indexing logic
├── tools.py                 # Tool definitions for the application
├── config.py                # Configuration and environment variable loading
└── .env                     # Environment variables (not included in version control)
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [Langchain](https://langchain.com/)
- [dotenv](https://pypi.org/project/python-dotenv/)
