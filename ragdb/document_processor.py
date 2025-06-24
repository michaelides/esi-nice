import os
from typing import List, Tuple, Dict, Any
from llama_index.core import SimpleDirectoryReader, Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter, NodeParser # NodeParser for type hint

# Config imports
from config import PROJECT_ROOT, SOURCE_DATA_DIR, WEB_MARKDOWN_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_process_documents() -> Tuple[List[LlamaDocument], NodeParser]:
    """
    Loads documents from specified directories, processes them, and prepares a node parser.
    Returns a tuple containing:
        - all_documents (List[LlamaDocument]): The loaded documents.
        - node_parser (NodeParser): The initialized SentenceSplitter.
    """
    print(f"LOG: doc_processor.Loading documents...")

    try:
        os.makedirs(SOURCE_DATA_DIR, exist_ok=True)
        os.makedirs(WEB_MARKDOWN_PATH, exist_ok=True)
    except OSError as e:
        print(f"ERROR: doc_processor.Could not create source/markdown directories. Error: {e}")
        # If these critical dirs can't be made, it's probably best to stop.
        # However, the function could also return empty list and allow make_rag to handle.
        # For now, let it proceed and potentially find no documents.
        pass # Or raise e if this should be a fatal error for the script

    input_dirs = [SOURCE_DATA_DIR, WEB_MARKDOWN_PATH]
    required_exts = [".pdf", ".txt", ".md", ".csv"]
    all_documents: List[LlamaDocument] = []

    def get_relative_path_metadata(filename: str) -> Dict[str, Any]:
        """Generates metadata including the file path relative to PROJECT_ROOT."""
        try:
            abs_project_root = os.path.abspath(PROJECT_ROOT)
            abs_filename = os.path.abspath(filename)

            if os.path.commonpath([abs_project_root, abs_filename]) == abs_project_root:
                relative_path = os.path.relpath(abs_filename, start=abs_project_root)
                return {"file_path": relative_path}
            else:
                print(f"WARNING: doc_processor.File {abs_filename} is outside project root {abs_project_root}. Storing absolute path.")
                return {"file_path": abs_filename}
        except ValueError as ve:
            print(f"WARNING: doc_processor.ValueError generating relative path for {filename}: {ve}. Storing absolute path.")
            return {"file_path": os.path.abspath(filename)}
        except Exception as e: # Catch any other error during path manipulation
            print(f"ERROR: doc_processor.Error in get_relative_path_metadata for {filename}: {e}. Storing absolute path.")
            return {"file_path": os.path.abspath(filename)}

    for input_dir_path_str in input_dirs: # Renamed to avoid conflict with input_dir variable in SimpleDirectoryReader
        abs_input_dir = os.path.abspath(input_dir_path_str)
        if not os.path.isdir(abs_input_dir): # Check if it's a directory
            print(f"WARNING: doc_processor.Path '{abs_input_dir}' is not a directory or does not exist. Skipping.")
            continue
        if not os.listdir(abs_input_dir): # Check if directory is empty
            print(f"LOG: doc_processor.Directory is empty, skipping: {abs_input_dir}")
            continue

        try:
            print(f"LOG: doc_processor.Reading from directory: {abs_input_dir}")
            reader = SimpleDirectoryReader(
                input_dir=abs_input_dir,
                required_exts=required_exts,
                recursive=True,
                file_metadata=get_relative_path_metadata
            )
            docs = reader.load_data(show_progress=True)
            if docs:
                print(f"LOG: doc_processor.Loaded {len(docs)} documents from {abs_input_dir}")
                all_documents.extend(docs)
            else:
                 print(f"LOG: doc_processor.No documents with extensions {required_exts} found in {abs_input_dir}")
        except ValueError as e:
            print(f"WARNING: doc_processor.ValueError reading from directory {abs_input_dir} (check path validity): {e}")
        except Exception as e:
            print(f"ERROR: doc_processor.Unexpected error loading documents from {abs_input_dir}: {e}")
            # SimpleDirectoryReader is quite robust and usually logs errors for individual files.
            # This catch is for more general failures during the load_data call for a whole directory.

    print(f"LOG: doc_processor.Total documents loaded: {len(all_documents)}")

    # Initialize Text Splitter (Node Parser) - this is unlikely to fail with default settings
    try:
        node_parser: NodeParser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"LOG: doc_processor.Node parser initialized (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    except Exception as e:
        print(f"CRITICAL_ERROR: doc_processor.Failed to initialize SentenceSplitter: {e}")
        # This is critical, RAG cannot proceed without a node_parser.
        # Re-raise or handle by returning a specific error indicator.
        # For now, if this fails, make_rag.py will likely fail when node_parser is None or invalid.
        # A more robust solution would be to raise a custom exception or ensure make_rag.py checks.
        raise # Re-raise for now, as it's critical.

    if not all_documents:
        print("WARNING: doc_processor.No documents were loaded. Returning empty list and parser.")
        # The calling function in make_rag.py already handles the case of no documents.

    return all_documents, node_parser

print("DEBUG: ragdb/document_processor.py processed with error handling improvements.")
