import os
import sys
import asyncio
from urllib.parse import urlparse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from huggingface_hub import HfApi, create_repo
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables from a .env file if it exists
load_dotenv()

# Add the project root to sys.path to allow importing config.py
# This assumes config.py is in the directory above ragdb/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

# --- DEBUGGING: Print paths to verify ---
print(f"DEBUG: Current script directory: {current_dir}")
print(f"DEBUG: Calculated project root: {project_root}")
print(f"DEBUG: sys.path after modification: {sys.path}")
# --- END DEBUGGING ---

# PROJECT_ROOT and other configurations are now imported from config.py
from config import (
    PROJECT_ROOT,
    HF_DATASET_ID,
    HF_VECTOR_STORE_SUBDIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SOURCE_DATA_DIR_RELATIVE,
    SOURCE_DATA_DIR,
    WEB_MARKDOWN_PATH_RELATIVE,
    WEB_MARKDOWN_PATH,
    ADDITIONAL_SOURCE_DATA_DIR_RELATIVE, # New import
    ADDITIONAL_SOURCE_DATA_DIR,         # New import
    HF_ADDITIONAL_SOURCE_DATA_UPLOAD_PATH, # New import
    WEBPAGES_FILE_RELATIVE,
    WEBPAGES_FILE
)

# Ensure HF_TOKEN is set for writing to Hugging Face Hub
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Error: HF_TOKEN environment variable is not set. Please set it to upload to Hugging Face Hub.")
    sys.exit(1) # Exit if token is not set, as upload will fail

print(f"Target Hugging Face Dataset for RAG persistence: {HF_DATASET_ID}/{HF_VECTOR_STORE_SUBDIR}")

# Ensure local directories exist
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)
os.makedirs(WEB_MARKDOWN_PATH, exist_ok=True)
os.makedirs(ADDITIONAL_SOURCE_DATA_DIR, exist_ok=True) # Ensure new source_data directory exists

URLS_TO_SCRAPE = []
try:
    with open(WEBPAGES_FILE, 'r') as file:
        # Strip whitespace/newlines from each line
        URLS_TO_SCRAPE = [line.strip() for line in file if line.strip()]
    if not URLS_TO_SCRAPE:
        print(f"Warning: {WEBPAGES_FILE_RELATIVE} is empty. No webpages will be scraped.")
except FileNotFoundError:
    print(f"Warning: Could not find {WEBPAGES_FILE_RELATIVE}. Please create this file in the project root directory and add URLs to scrape, one per line. No webpages will be scraped.")
except Exception as e:
    print(f"Error reading {WEBPAGES_FILE_RELATIVE}: {e}. No webpages will be scraped.")

# url_to_filename and scrape_websites functions have been moved to ragdb.web_scraper
from ragdb.web_scraper import scrape_websites
# Import document processing function
from ragdb.document_processor import load_and_process_documents

# --- Main Script ---
async def main():
    # --- Initialize Embedding Model ---
    embedding_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

    print(f"Configuring RAG to persist to Hugging Face Dataset: {HF_DATASET_ID}, path in repo: {HF_VECTOR_STORE_SUBDIR}")

    # Create the Hugging Face repository if it doesn't exist
    try:
        print(f"Ensuring Hugging Face dataset '{HF_DATASET_ID}' exists...")
        create_repo(repo_id=HF_DATASET_ID, repo_type="dataset", token=hf_token, exist_ok=True)
        print(f"Hugging Face dataset '{HF_DATASET_ID}' is ready.")
    except Exception as e:
        print(f"Error creating/checking Hugging Face repository: {e}")
        print("Please ensure your HF_TOKEN is valid and you have permissions to create datasets.")
        return # Exit if repo cannot be created/accessed

    # 1. Scrape websites
    await scrape_websites(URLS_TO_SCRAPE, WEB_MARKDOWN_PATH)

    # 2. Load and process documents from existing sources (articles and web_markdown)
    all_documents, node_parser = load_and_process_documents()

    # New: Load documents from the additional source_data directory
    print(f"LOG: doc_processor.Reading from directory: {ADDITIONAL_SOURCE_DATA_DIR}")
    try:
        additional_documents = SimpleDirectoryReader(
            input_dir=ADDITIONAL_SOURCE_DATA_DIR,
            recursive=True,
            exclude_hidden=True
        ).load_data()
        if additional_documents:
            print(f"LOG: doc_processor.Loaded {len(additional_documents)} documents from {ADDITIONAL_SOURCE_DATA_DIR}")
            all_documents.extend(additional_documents) # Add them to the main list
        else:
            print(f"LOG: doc_processor.Directory is empty, skipping: {ADDITIONAL_SOURCE_DATA_DIR}")
    except Exception as e:
        print(f"Error loading documents from {ADDITIONAL_SOURCE_DATA_DIR}: {e}")


    # Initialize HfApi for uploads
    api = HfApi(token=hf_token)

    if not all_documents:
        print("make_rag.py: No documents loaded for indexing. Skipping vector store creation.")
    else:
        # 3. Initialize SimpleVectorStore and Storage Context for local persistence
        print("make_rag.py: Initializing SimpleVectorStore for local persistence...")
        vector_store = SimpleVectorStore()
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 5. Create VectorStoreIndex (This performs parsing, embedding, and indexing)
        print(f"Creating index for {len(all_documents)} documents... (This may take a while)")
        index = VectorStoreIndex.from_documents(
            all_documents,
            storage_context=storage_context,
            embed_model=embedding_model,
            node_parser=node_parser,
            show_progress=True,
        )

        # 6. Persist the index locally to a temporary directory
        local_persist_dir = tempfile.mkdtemp()
        print(f"Persisting index locally to temporary directory: {local_persist_dir}...")
        try:
            index.storage_context.persist(persist_dir=local_persist_dir)
            print("Local persistence successful.")

            # 7. Upload the persisted vector store data to Hugging Face Dataset
            print(f"Uploading persisted vector store to Hugging Face Dataset: {HF_DATASET_ID}, path in repo: {HF_VECTOR_STORE_SUBDIR}...")
            api.upload_folder(
                folder_path=local_persist_dir,
                repo_id=HF_DATASET_ID,
                path_in_repo=HF_VECTOR_STORE_SUBDIR,
                repo_type="dataset",
                commit_message="Upload RAG vector store"
            )
            print(f"Successfully uploaded vector store to Hugging Face Dataset: {HF_DATASET_ID}/{HF_VECTOR_STORE_SUBDIR}")

        except Exception as e:
            print(f"An error occurred during local persistence or vector store upload: {e}")
        finally:
            if os.path.exists(local_persist_dir):
                print(f"Cleaning up temporary local persistence directory: {local_persist_dir}")
                shutil.rmtree(local_persist_dir)

    # Upload source data directories regardless of whether an index was created
    # Upload ragdb/articles to rag/articles
    print(f"Uploading local documents from '{SOURCE_DATA_DIR}' to '{HF_DATASET_ID}/articles'...")
    try:
        api.upload_folder(
            folder_path=SOURCE_DATA_DIR,
            repo_id=HF_DATASET_ID,
            path_in_repo="articles", # User specified 'articles' subfolder
            repo_type="dataset",
            commit_message="Upload local articles for RAG",
            # allow_patterns=["*.pdf", "*.txt", "*.md"] # Optional: restrict file types
        )
        print(f"Successfully uploaded '{SOURCE_DATA_DIR}' to '{HF_DATASET_ID}/articles'.")
    except Exception as e:
        print(f"Error uploading local articles: {e}")

    # Upload ragdb/web_markdown to rag/web_markdown
    print(f"Uploading scraped web markdown from '{WEB_MARKDOWN_PATH}' to '{HF_DATASET_ID}/web_markdown'...")
    try:
        api.upload_folder(
            folder_path=WEB_MARKDOWN_PATH,
            repo_id=HF_DATASET_ID,
            path_in_repo="web_markdown", # User specified 'web_markdown' subfolder
            repo_type="dataset",
            commit_message="Upload scraped web markdown for RAG",
            # allow_patterns=["*.md"] # Optional: restrict file types
        )
        print(f"Successfully uploaded '{WEB_MARKDOWN_PATH}' to '{HF_DATASET_ID}/web_markdown'.")
    except Exception as e:
        print(f"Error uploading web markdown: {e}")

    # New: Upload ragdb/source_data to gm42/rag/source_data
    print(f"Uploading additional source data from '{ADDITIONAL_SOURCE_DATA_DIR}' to '{HF_DATASET_ID}/{HF_ADDITIONAL_SOURCE_DATA_UPLOAD_PATH}'...")
    try:
        api.upload_folder(
            folder_path=ADDITIONAL_SOURCE_DATA_DIR,
            repo_id=HF_DATASET_ID,
            path_in_repo=HF_ADDITIONAL_SOURCE_DATA_UPLOAD_PATH,
            repo_type="dataset",
            commit_message="Upload additional source data for RAG",
            # allow_patterns=["*.pdf", "*.txt", "*.md"] # Optional: restrict file types
        )
        print(f"Successfully uploaded '{ADDITIONAL_SOURCE_DATA_DIR}' to '{HF_DATASET_ID}/{HF_ADDITIONAL_SOURCE_DATA_UPLOAD_PATH}'.")
    except Exception as e:
        print(f"Error uploading additional source data: {e}")

    print("RAG database creation script finished.")


if __name__ == "__main__":
   asyncio.run(main())
