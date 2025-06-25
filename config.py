import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if it exists
# Determine the project root dynamically, assuming config.py is at the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# --- Hugging Face Hub Configuration ---
# It's recommended to set HF_TOKEN as an environment variable (e.g., in a .env file)
# and load it using `python-dotenv`.
HF_TOKEN = os.getenv("HF_TOKEN", None) 

# IMPORTANT: Replace "your-hf-username/your-rag-dataset" with your actual Hugging Face Dataset ID.
# This is where your RAG vector store will be uploaded and persisted.
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "gm42/rag") 
HF_VECTOR_STORE_SUBDIR = "esi_simplevector" # Subdirectory within the HF dataset to store the vector store

# New: Hugging Face Dataset ID for user memories (chat history and metadata)
HF_USER_MEMORIES_DATASET_ID = os.getenv("HF_USER_MEMORIES_DATASET_ID", "gm42/user_memories")

# --- RAG Document Processing Configuration ---
CHUNK_SIZE = 512 # Size of text chunks for processing by the node parser
CHUNK_OVERLAP = 20 # Overlap between text chunks to maintain context

# Paths for source data and scraped web content
# These paths are relative to the PROJECT_ROOT
SOURCE_DATA_DIR_RELATIVE = "ragdb/articles" # Directory for local documents (e.g., PDFs, text files)
SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, SOURCE_DATA_DIR_RELATIVE)

WEB_MARKDOWN_PATH_RELATIVE = "ragdb/web_markdown" # Directory where scraped web content (markdown) is stored
WEB_MARKDOWN_PATH = os.path.join(PROJECT_ROOT, WEB_MARKDOWN_PATH_RELATIVE)

# New: Path for additional source data to be indexed and uploaded
ADDITIONAL_SOURCE_DATA_DIR_RELATIVE = "ragdb/source_data"
ADDITIONAL_SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, ADDITIONAL_SOURCE_DATA_DIR_RELATIVE)
HF_ADDITIONAL_SOURCE_DATA_UPLOAD_PATH = "source_data" # Target subfolder on Hugging Face

WEBPAGES_FILE_RELATIVE = "ragdb/webpages.txt" # File containing URLs to scrape, one per line
WEBPAGES_FILE = os.path.join(PROJECT_ROOT, WEBPAGES_FILE_RELATIVE)

# --- Agent Configuration ---
MAX_SEARCH_RESULTS = 5 # Default number of search results for tools

# --- Other API Keys (if needed by other modules in your project) ---
# These are included as common examples, ensure they are set as environment variables
# or directly in your .env file.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None) # For GoogleGenAIEmbedding
