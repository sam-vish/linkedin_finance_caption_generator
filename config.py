import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")