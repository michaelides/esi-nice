import streamlit as st
import os
import re
import json
from typing import List, Dict, Any, Optional, Callable
import html # Import html for escaping HTML content
import pandas as pd
from PyPDF2 import PdfReader # Added for PDF processing in stui.py
from docx import Document # Added for DOCX processing in stui.py
import io # Added for BytesIO in stui.py
import pyreadstat # Ensure pyreadstat is imported if read_spss is used

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Import UI_ACCESSIBLE_WORKSPACE from tools.py
from tools import UI_ACCESSIBLE_WORKSPACE

# Initialize session state for clipboard functionality
if 'text_to_copy_payload' not in st.session_state:
    st.session_state.text_to_copy_payload = None
if 'clipboard_triggered_for_id' not in st.session_state:
    st.session_state.clipboard_triggered_for_id = None

st.set_page_config(
    page_title="ESI - ESI Scholarly Instructor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_chat():
    """Display the chat messages from the session state, handling file downloads and image display."""
    CODE_DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
    RAG_SOURCE_MARKER = "---RAG_SOURCE---"
    
    # Use the imported UI_ACCESSIBLE_WORKSPACE directly
    os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)

    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            
            text_to_display = content
            rag_sources_data = []
            code_download_filename = None
            code_download_filepath_relative = None
            code_is_image = False

            if message["role"] == "assistant":
                # --- 1. Extract RAG sources using regex ---
                # Regex to find RAG_SOURCE_MARKER followed by a JSON object
                # The JSON object is captured in group 1
                rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER)}({{\"type\":.*?}})", re.DOTALL)
                
                # Find all matches
                all_rag_matches = list(rag_source_pattern.finditer(text_to_display))
                
                # Extract JSON data and remove markers from text_to_display
                processed_text_after_rag = text_to_display
                for match in reversed(all_rag_matches): # Process in reverse to avoid index issues
                    json_str = match.group(1)
                    try:
                        rag_data = json.loads(json_str)
                        rag_sources_data.append(rag_data)
                        # print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}") # Removed verbose log
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")
                    
                    # Remove the entire matched marker and JSON from the text
                    processed_text_after_rag = processed_text_after_rag[:match.start()] + processed_text_after_rag[match.end():]
                
                text_to_display = processed_text_after_rag.strip()

                # --- 2. Extract Code Interpreter download marker ---
                code_marker_match = re.search(rf"^{re.escape(CODE_DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    text_to_display = text_to_display[:code_marker_match.start()].strip() + text_to_display[code_marker_match.end():].strip()
                    
                    # print(f"Found code download marker. Filename: {extracted_filename}") # Removed verbose log
                    code_download_filename = extracted_filename
                    # Use UI_ACCESSIBLE_WORKSPACE for relative path construction
                    code_download_filepath_relative = os.path.relpath(os.path.join(UI_ACCESSIBLE_WORKSPACE, extracted_filename), PROJECT_ROOT)

                    code_download_filepath_absolute = os.path.join(PROJECT_ROOT, code_download_filepath_relative)

                    if extracted_filename and os.path.exists(code_download_filepath_absolute):
                        # print(f"Code download file exists at: {code_download_filepath_absolute}") # Removed verbose log
                        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
                        if os.path.splitext(code_download_filename)[1].lower() in image_extensions:
                            code_is_image = True
                            # print(f"Detected image file from code interpreter: {code_download_filename}") # Removed verbose log
                    else:
                        print(f"Code download file '{extracted_filename}' NOT found at '{code_download_filepath_absolute}'.")
                        text_to_display += f"\n\n*(Warning: The file '{extracted_filename}' mentioned for download could not be found.)*"

            # --- 3. Display main text content ---
            if text_to_display:
                st.markdown(text_to_display)

            # --- 4. Display RAG sources (PDFs and Web Links) - Deduplicated ---
            displayed_rag_identifiers = set()
            # Sort rag_sources_data to ensure consistent display order if multiple sources
            rag_sources_data.sort(key=lambda x: x.get('citation_number', float('inf')) if x.get('type') == 'pdf' else x.get('url', x.get('title', '')))

            for rag_idx, rag_data in enumerate(rag_sources_data):
                source_type = rag_data.get("type")
                identifier = None
                display_item = False

                if source_type == "pdf":
                    pdf_name = rag_data.get("name", "source.pdf")
                    pdf_source_path = rag_data.get("path") # This will be either http:// or file://
                    citation_num = rag_data.get('citation_number')
                    citation_prefix = f"[{citation_num}] " if citation_num else ""
                    
                    identifier = pdf_source_path # Use the path/URL as identifier for deduplication

                    if identifier and identifier not in displayed_rag_identifiers:
                        if pdf_source_path and pdf_source_path.startswith("http"):
                            # It's a Hugging Face URL, display as a link
                            st.markdown(f"Source: {citation_prefix}[{pdf_name}]({pdf_source_path})")
                            # print(f"Added link for RAG PDF (URL): {citation_prefix}{pdf_name}") # Removed verbose log
                            display_item = True
                        elif pdf_source_path and pdf_source_path.startswith("file://"):
                            # It's a local file path with 'file://' prefix
                            local_file_path = pdf_source_path[len("file://"):] # Remove 'file://'
                            # The path from tools.py is already absolute, so no need to join with PROJECT_ROOT
                            pdf_absolute_path = local_file_path 

                            if os.path.exists(pdf_absolute_path):
                                try:
                                    button_label = f"{citation_prefix}Download PDF: {pdf_name}"
                                    with open(pdf_absolute_path, "rb") as fp:
                                        st.download_button(
                                            label=button_label,
                                            data=fp,
                                            file_name=pdf_name,
                                            mime="application/pdf",
                                            key=f"rag_pdf_{msg_idx}_{rag_idx}_{pdf_name}"
                                        )
                                    # print(f"Added download button for RAG PDF (local file://): {button_label} (Path: {pdf_absolute_path})") # Removed verbose log
                                    display_item = True
                                except Exception as e:
                                    st.error(f"Error creating download button for {pdf_name}: {e}")
                                    print(f"Error for RAG PDF '{pdf_name}': {e}")
                            else:
                                st.warning(f"Referenced PDF '{pdf_name}' not found locally at '{pdf_absolute_path}'.")
                                print(f"Warning: Referenced PDF '{pdf_name}' not found at expected absolute path: {pdf_absolute_path}")
                        else:
                            # Unexpected path format
                            st.warning(f"Referenced PDF '{pdf_name}' has an unsupported path format: '{pdf_source_path}'.")
                            print(f"Warning: Referenced PDF '{pdf_name}' has an unsupported path format: '{pdf_source_path}'.")
                
                elif source_type == "web":
                    url = rag_data.get("url")
                    title = rag_data.get("title", url)
                    identifier = url
                    if identifier and identifier not in displayed_rag_identifiers:
                        if url:
                            st.markdown(f"Source: [{title}]({url})")
                            # print(f"Added link for RAG web source: {title} (URL: {url})") # Removed verbose log
                            display_item = True
                
                if display_item and identifier:
                    displayed_rag_identifiers.add(identifier)
                    st.divider()

            # --- 5. Display Code Interpreter output (Image or Download Button) ---
            code_download_absolute_filepath = os.path.join(PROJECT_ROOT, code_download_filepath_relative) if code_download_filepath_relative else None

            if code_is_image and code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath):
                try:
                    st.image(code_download_absolute_filepath, caption=code_download_filename, use_container_width=True)
                    # print(f"Successfully displayed image from code interpreter: {code_download_filename}") # Removed verbose log
                except Exception as e:
                    st.error(f"Error displaying image {code_download_filename}: {e}")
                    code_is_image = False
            
            if code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath) and not code_is_image:
                try:
                    with open(code_download_absolute_filepath, "rb") as fp:
                        st.download_button(
                            label=f"Download {code_download_filename}",
                            data=fp,
                            file_name=code_download_filename,
                            mime="application/octet-stream",
                            key=f"code_dl_{msg_idx}_{code_download_filename}"
                        )
                    # print(f"Successfully added download button for code interpreter file: {code_download_filename}") # Removed verbose log
                except Exception as e:
                    st.error(f"Error creating download button for {code_download_filename}: {e}")

            # --- Add Copy to Clipboard and Regenerate Buttons ---
            is_last_assistant_message = (message["role"] == "assistant" and msg_idx == len(st.session_state.messages) - 1)
            
            can_regenerate = False
            if is_last_assistant_message:
                if len(st.session_state.messages) == 1:
                    can_regenerate = True
                elif len(st.session_state.messages) > 1 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    can_regenerate = True

            # The hidden div and content_div_id are no longer needed.

            if can_regenerate:
                col_copy, col_regen, _ = st.columns([0.05, 0.05, 0.9])
            else:
                col_copy, _ = st.columns([0.05, 0.95])

            # Define callback for the copy button
            def _copy_button_callback(text_payload, message_id):
                st.session_state.text_to_copy_payload = text_payload
                st.session_state.clipboard_triggered_for_id = message_id
                # print(f"Copy button clicked for msg {message_id}. Payload set in session state.") # Removed verbose log

            with col_copy:
                # Replace markdown button with st.button
                # text_to_display and msg_idx are from the parent scope of display_chat's loop
                col_copy.button(
                    "📋",
                    key=f"copy_btn_{msg_idx}",
                    help="Copy message to clipboard",
                    on_click=_copy_button_callback,
                    args=(text_to_display, msg_idx) # Corrected: Use text_to_display here
                )
            
            # JS injection for clipboard based on session state
            if st.session_state.get('clipboard_triggered_for_id') == msg_idx:
                text_to_copy_js = st.session_state.get('text_to_copy_payload', "")
                # Using json.dumps to safely escape the text for JavaScript
                escaped_text_for_js = json.dumps(escaped_text_for_js)

                javascript_to_run = f"""
                <script>
                    (function() {{
                        window.focus(); // Attempt to focus the current window/iframe
                        const textToCopy = {escaped_text_for_js};
                        navigator.clipboard.writeText(textToCopy).then(function() {{
                            console.log('Async: Copying to clipboard was successful!', textToCopy);
                            // Toast will be shown from Python side
                        }}, function(err) {{
                            console.error('Async: Could not copy text. Error object:', err); console.error('Error name:', err.name); console.error('Error message:', err.message);
                            alert('Failed to copy text. Check console for errors.');
                        }});
                    }})();
                </script>
                """
                st.components.v1.html(javascript_to_run, height=0, width=0)

                # Display toast message
                st.toast(f"Content from message {msg_idx + 1} copied!", icon="📋")

                # Reset the trigger and payload
                st.session_state.clipboard_triggered_for_id = None
                st.session_state.text_to_copy_payload = None

            if can_regenerate:
                with col_regen:
                    if st.button("🔄", key=f"regenerate_{msg_idx}", help="Regenerate Response"):
                        st.session_state.do_regenerate = True
                        st.rerun()

def remove_uploaded_file(file_name: str, file_type: str):
    """Removes an uploaded file from session state and from the workspace."""
    if file_type == "document":
        if file_name in st.session_state.uploaded_documents:
            del st.session_state.uploaded_documents[file_name]
            st.toast(f"Document '{file_name}' removed.", icon="🗑️")
    elif file_type == "dataframe":
        if file_name in st.session_state.uploaded_dataframes:
            del st.session_state.uploaded_dataframes[file_name]
            st.toast(f"Dataset '{file_name}' removed.", icon="🗑️")
    
    # Attempt to delete the physical file from the workspace
    file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
    if os.path.exists(file_path_in_workspace):
        try:
            os.remove(file_path_in_workspace)
            # print(f"Successfully deleted physical file: {file_path_in_workspace}") # Removed verbose log
        except Exception as e:
            print(f"Error deleting physical file '{file_path_in_workspace}': {e}")
            st.error(f"Error deleting physical file '{file_name}': {e}")
    
    st.rerun()


def process_uploaded_file(uploaded_file):
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    # Save the raw file to the UI_ACCESSIBLE_WORKSPACE first
    try:
        os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)
        file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
        with open(file_path_in_workspace, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{file_name}' saved to workspace.")
    except Exception as e:
        st.error(f"Error saving file '{file_name}' to workspace: {e}")
        return None, None # Indicate failure

    # --- ADDED: Warning for research data files ---
    data_file_extensions = [".csv", ".xlsx", ".sav", ".rdata", ".rds"]
    if file_extension in data_file_extensions:
        st.warning(
            "**Important:** If this file contains research data, please ensure you have "
            "obtained all necessary ethical approvals for its use and upload. "
            "Do not upload sensitive or confidential data without proper authorization."
        )
    # --- END ADDED SECTION ---

    # Now, process the file content and store in session state for agent tools
    if file_extension in [".pdf", ".docx", ".md", ".txt"]: # Added .txt
        text_content = ""
        try:
            if file_extension == ".pdf":
                reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                for page in reader.pages:
                    text_content += page.extract_text() or ""
            elif file_extension == ".docx":
                document = Document(io.BytesIO(uploaded_file.getvalue()))
                for para in document.paragraphs:
                    text_content += para.text + "\n"
            elif file_extension in [".md", ".txt"]: # Handle .md and .txt as plain text
                text_content = uploaded_file.getvalue().decode("utf-8")
            
            st.session_state.uploaded_documents[file_name] = text_content
            st.success(f"Document '{file_name}' processed for agent access.")
            return "document", file_name
        except Exception as e:
            # Corrected typo from file_file_name to file_name
            st.error(f"Error processing document '{file_name}' for agent access: {e}")
            return None, None
    
    elif file_extension in [".csv", ".xlsx", ".sav"]:
        df = None
        try:
            if file_extension == ".csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == ".xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_extension == ".sav":
                # pandas.read_spss requires pyreadstat
                try:
                    # Use the file path from the saved file in the workspace
                    df = pd.read_spss(file_path_in_workspace)
                except ImportError:
                    st.error("`pyreadstat` library not found. Please install it (`pip install pyreadstat`) to read .sav files.")
                    return None, None
            
            if df is not None:
                st.session_state.uploaded_dataframes[file_name] = df
                st.success(f"Dataset '{file_name}' processed for agent access.")
                return "dataframe", file_name
            else:
                st.error(f"Could not load dataframe from '{file_name}'.")
                return None, None
        except Exception as e:
            st.error(f"Error processing dataset '{file_name}' for agent access: {e}")
            return None, None
    
    elif file_extension in [".rdata", ".rds"]:
        st.warning(f"File type '{file_extension}' for '{file_name}' is not directly supported for processing in Python. Please convert it to CSV or XLSX.")
        return None, None
    else:
        st.warning(f"Unsupported file type: {file_extension} for '{file_name}'. File saved to workspace but not processed for agent tools.")
        return None, None

def create_interface(
    reset_callback: Callable,
    new_chat_callback: Callable,
    delete_chat_callback: Callable,
    rename_chat_callback: Callable,
    chat_metadata: Dict[str, str],
    current_chat_id: str,
    switch_chat_callback: Callable,
    get_discussion_markdown_callback: Callable,
    get_discussion_docx_callback: Callable,
    suggested_prompts_list: Optional[List[str]],
    handle_user_input_callback: Callable,
    long_term_memory_enabled: bool, # New parameter
    forget_me_callback: Callable, # New parameter
    set_long_term_memory_callback: Callable # New parameter
):
    """Create the Streamlit UI for the chat interface."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your dissertation research")

    # Initialize editing state if not present
    if 'editing_chat_id' not in st.session_state:
        st.session_state.editing_chat_id = None

    with st.sidebar:
        with st.expander("**Chat History**", expanded=False, icon = ":material/forum:"):
            if not long_term_memory_enabled:
                st.warning("Long-term memory is currently **disabled**. Your chat history will not be saved and will be lost when you close this tab or refresh the page.")
                st.info("To enable long-term memory, check the option in 'LLM Settings'.")
                # Only show "New Chat" button, no list of past chats
                if st.button("➕ New Chat (Temporary)", key="new_chat_button_temp", use_container_width=True):
                    st.session_state.editing_chat_id = None
                    new_chat_callback() # This will create a new in-memory session
            else:
                st.info("Conversations are automatically saved and linked to your browser via cookies. Clearing browser data will remove your saved discussions.")
                
                if st.button("➕ New Chat", key="new_chat_button", use_container_width=True):
                    # Clear any active editing state when creating a new chat
                    st.session_state.editing_chat_id = None
                    new_chat_callback()

                # Display existing chats
                if chat_metadata: # Only iterate if there's metadata
                    sorted_chat_items = sorted(chat_metadata.items(), key=lambda item: item[1].lower())
                    
                    for chat_id, chat_name in sorted_chat_items:
                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            if st.session_state.editing_chat_id == chat_id:
                                # Display text input for renaming
                                new_name = st.text_input(
                                    "New name:",
                                    value=chat_name,
                                    key=f"rename_input_{chat_id}",
                                    label_visibility="collapsed",
                                    on_change=lambda current_chat_id_in_loop=chat_id: (
                                        rename_chat_callback(current_chat_id_in_loop, st.session_state[f"rename_input_{current_chat_id_in_loop}"]) if st.session_state[f"rename_input_{current_chat_id_in_loop}"] and st.session_state[f"rename_input_{current_chat_id_in_loop}"] != chat_metadata.get(current_chat_id_in_loop) else None,
                                        setattr(st.session_state, 'editing_chat_id', None) # Clear editing state
                                    )
                                )
                            else:
                                # Display button for switching chat
                                if st.button(chat_name, key=f"chat_select_{chat_id}", use_container_width=True,
                                            type="primary" if chat_id == current_chat_id else "secondary"):
                                    if chat_id != current_chat_id:
                                        # Clear any active editing state when switching chats
                                        st.session_state.editing_chat_id = None
                                        switch_chat_callback(chat_id)
                        with col2:
                            with st.popover("⋮", use_container_width=True):
                                st.write(f"Options for: **{chat_name}**")
                                
                                # Option to download Markdown
                                st.download_button(
                                    label="⬇️ Download (.md)",
                                    data=get_discussion_markdown_callback(chat_id),
                                    file_name=f"{chat_name.replace(' ', '_')}.md",
                                    mime="text/markdown",
                                    key=f"download_listed_md_{chat_id}", # Changed key to be unique
                                    use_container_width=True
                                )

                                # Option to download DOCX
                                st.download_button(
                                    label="⬇️ Download (.docx)",
                                    data=get_discussion_docx_callback(chat_id),
                                    file_name=f"{chat_name.replace(' ', '_')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"download_listed_docx_{chat_id}", # New unique key
                                    use_container_width=True
                                )
                                
                                # Option to rename (sets editing_chat_id and reruns)
                                if st.button("✏️ Rename", key=f"rename_btn_{chat_id}", use_container_width=True):
                                    st.session_state.editing_chat_id = chat_id
                                    st.rerun() # Rerun to show the input field

                                # Option to delete
                                if st.button("♻ Delete", key=f"delete_from_popover_{chat_id}", use_container_width=True):
                                    # Clear any active editing state if the chat being edited is deleted
                                    if st.session_state.editing_chat_id == chat_id:
                                        st.session_state.editing_chat_id = None
                                    delete_chat_callback(chat_id)
                else:
                    st.info("No saved chats yet. Start a new conversation!")

        with st.expander("**Upload files**", expanded=False, icon = ":material/upload_file:"):
            uploaded_file = st.file_uploader(
                "Upload a document or dataset",
                type=["pdf", "docx", "md", "txt", "csv", "xlsx", "sav", "rdata", "rds"], # Added .txt
                accept_multiple_files=False,
                key="file_uploader"
            )

            if uploaded_file is not None:
                # Check if the file has already been processed in this session
                if uploaded_file.name not in st.session_state.uploaded_documents and \
                uploaded_file.name not in st.session_state.uploaded_dataframes:
                    file_type, file_name = process_uploaded_file(uploaded_file)
                    if file_name:
                        # Add a message to the chat history about the upload
                        if file_type == "document":
                            st.session_state.messages.append({"role": "assistant", "content": f"I've received your document: `{file_name}`. You can now ask me to `read_uploaded_document('{file_name}')`."})
                        elif file_type == "dataframe":
                            st.session_state.messages.append({"role": "assistant", "content": f"I've received your dataset: `{file_name}`. You can now ask me to `analyze_uploaded_dataframe('{file_name}')` or use the `code_interpreter` tool for more complex analysis."})
                        st.rerun() # Rerun to display the new assistant message
                else:
                    st.info(f"File '{uploaded_file.name}' has already been uploaded and processed.")
            
            # Removed the "Uploaded Files" subsection to avoid duplication
            # st.subheader("Uploaded Files")
            # if st.session_state.uploaded_documents or st.session_state.uploaded_dataframes:
            #     st.markdown("---")
            #     if st.session_state.uploaded_documents:
            #         st.markdown("##### Documents:")
            #         for doc_name in st.session_state.uploaded_documents.keys():
            #             col1, col2 = st.columns([0.8, 0.2])
            #             with col1:
            #                 st.write(f"- 📄 {doc_name}")
            #             with col2:
            #                 st.button(
            #                     "🗑️",
            #                     key=f"remove_doc_{doc_name}",
            #                     help=f"Remove {doc_name}",
            #                     on_click=remove_uploaded_file,
            #                     args=(doc_name, "document")
            #                 )
            #     if st.session_state.uploaded_dataframes:
            #         st.markdown("##### Datasets:")
            #         for df_name in st.session_state.uploaded_dataframes.keys():
            #             col1, col2 = st.columns([0.8, 0.2])
            #             with col1:
            #                 st.write(f"- 📊 {df_name}")
            #             with col2:
            #                 st.button(
            #                     "🗑️",
            #                     key=f"remove_df_{df_name}",
            #                     help=f"Remove {df_name}",
            #                     on_click=remove_uploaded_file,
            #                     args=(df_name, "dataframe")
            #                 )
            #     st.markdown("---")
            # else:
            #     st.info("No files uploaded yet.")

        with st.expander("**LLM Settings**", expanded=False, icon = ":material/tune:"):
            st.slider(
                "Creativity (Temperature)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get("llm_temperature", 0.7),
                step=0.1,
                key="llm_temperature",
                help="Controls the randomness of the AI's responses. Lower values are more focused, higher values are more creative."
            )
            st.slider(
                "Verbosity",
                min_value=1,
                max_value=5,
                value=st.session_state.get("llm_verbosity", 3),
                step=1,
                key="llm_verbosity",
                help="Controls the detail level of the AI's responses. 1 is concise, 5 is very detailed."
            )
            st.slider(
                "Number of Search Results",
                min_value=3,
                max_value=15,
                value=st.session_state.get("search_results_count", 5),
                step=1,
                key="search_results_count",
                help="Controls the maximum number of search results returned by search tools (DuckDuckGo, Tavily, Stochasticscholar)."
            )
            st.toggle(
                "Enable Long-term Memory (saves chat history)",
                value=st.session_state.get("long_term_memory_enabled", False), # Default to False
                key="long_term_memory_enabled", # Ensure this key matches the one used in app.py
                help="If enabled, your chat history will be saved and loaded across sessions using browser cookies. If disabled, your chats will be forgotten when you close the browser or refresh the page."
            )

           # Implement the forget me button
            if long_term_memory_enabled: # Corrected variable name from 'on' to 'long_term_memory_enabled'
                # Use a popover for confirmation
                with st.popover("Forget Me (Delete All Data)", use_container_width=True):
                    st.warning("This will permanently delete ALL your saved chat histories and remove your user ID cookie from this browser. This action cannot be undone.")
                    st.write("Are you sure you want to proceed?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, Delete All Data", key="confirm_forget_me_yes", type="primary", use_container_width=True):
                            forget_me_callback() # Call the function passed from app.py
                            # No need for st.success() here, as the page will immediately reload.
                    with col_no:
                        if st.button("No, Cancel", key="confirm_forget_me_no", use_container_width=True):
                            st.info("Deletion cancelled.")
                            # Popover will close automatically
            else:
                st.info("Long-term memory is disabled. No chat history is being saved.")



        with st.expander("**About ESI**", expanded=False, icon = ":material/info:"):
            st.info("ESI uses AI to help you navigate the dissertation process. It has access to some of the literature in your reading lists and also uses search tools for web lookups.")
            st.warning("⚠️  Remember: Always consult your dissertation supervisor for final guidance and decisions.")
            st.info("Made for NBS7091A and NBS7095x")

    # Apply CSS globally
    CSS = """
    .stExpander > details {
        border: none;
    }
    """
    st.html(f"<style>{CSS}</style>")

    display_chat()
    display_main_chat_area(suggested_prompts_list, handle_user_input_callback)

def display_main_chat_area(suggested_prompts_list: Optional[List[str]], handle_user_input_callback: Callable):
    """
    Displays the main chat area including suggested prompts and the chat input field.
    """
    if suggested_prompts_list:
        cols = st.columns(len(suggested_prompts_list))
        for i, prompt in enumerate(suggested_prompts_list):
            with cols[i]:
                if st.button(prompt, key=f"suggested_prompt_btn_{i}"):
                    st.session_state.prompt_to_use = prompt
                    st.rerun() 
    else:
        pass

    chat_input_value = st.chat_input("Ask me about dissertations, research methods, academic writing, etc.")
    st.session_state.chat_input_value_from_stui = chat_input_value
