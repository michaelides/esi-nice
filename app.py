import functools
import os
import uuid
import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from nicegui import app, ui, Client
from nicegui.events import UploadEventArguments
from nicegui.storage import ObservableDict

from agent import create_orchestrator_agent, generate_llm_greeting, generate_suggested_prompts
from config import HF_TOKEN, HF_USER_MEMORIES_DATASET_ID

# LlamaIndex imports
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool # Import FunctionTool for dynamic tools

# File processing imports
from docx import Document
from io import BytesIO
import pandas as pd
import pyreadstat # For SPSS processing
from PyPDF2 import PdfReader

# Hugging Face related
try:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
except ImportError:
    print("Hugging Face Hub not installed. User data will not be persisted.")
    fs = None

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
AGENT_INSTANCE_KEY = "esi_orchestrator_agent_instance"

def simple_once_cache(func):
    _cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Use a simple key for global functions that should only run once
        # This assumes these functions don't need different results for different args
        key = func.__name__
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]
    return wrapper

@simple_once_cache
def setup_global_llm_settings_cached() -> tuple[bool, str | None]:
    """
    Initializes global LLM settings and caches the result.
    Returns (True, None) on success, or (False, error_message) on failure.
    """
    from agent import initialize_settings as initialize_agent_settings
    print("Initializing LLM settings...")
    try:
        initialize_agent_settings()
        print("LLM settings initialized.")
        return True, None
    except Exception as e:
        error_message = f"Fatal Error: Could not initialize LLM settings. {e}"
        print(error_message)
        return False, error_message

@functools.lru_cache(maxsize=1)
def setup_agent_cached(max_search_results: int) -> tuple[Any | None, str | None]:
    print(f"Initializing AI agent with max_search_results={max_search_results}...")
    try:
        # Import tools here to avoid circular dependency if agent imports app
        from tools import get_duckduckgo_tool, get_tavily_tool, get_wikipedia_tool, get_semantic_scholar_tool_for_agent, get_web_scraper_tool_for_agent, get_rag_tool_for_agent, get_coder_tools

        # Define runtime versions of tools that need access to app.storage.user
        def read_uploaded_document_tool_fn_runtime(filename: str) -> str:
            uploaded_docs = app.storage.user.get("uploaded_documents", {})
            if filename not in uploaded_docs:
                return f"Error: Document '{filename}' not found. Available: {list(uploaded_docs.keys())}"
            # Assuming content is bytes, decode it for text tools
            return uploaded_docs[filename].decode('utf-8', errors='replace')

        def analyze_dataframe_tool_fn_runtime(filename: str, head_rows: int = 5) -> str:
            uploaded_dfs = app.storage.user.get("uploaded_dataframes", {})
            if filename not in uploaded_dfs:
                return f"Error: DataFrame '{filename}' not found. Available: {list(uploaded_dfs.keys())}"
            
            df_content_bytes = uploaded_dfs[filename]
            df = None
            try:
                # Attempt to infer format or use a common one like CSV
                # A more robust solution would store content_type or infer from filename
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(BytesIO(df_content_bytes))
                elif filename.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(BytesIO(df_content_bytes))
                elif filename.lower().endswith('.sav'):
                    # pyreadstat needs a file path, so save temporarily
                    temp_path = os.path.join(app.storage.general.get("UI_ACCESSIBLE_WORKSPACE"), filename)
                    with open(temp_path, 'wb') as f:
                        f.write(df_content_bytes)
                    df = pyreadstat.read_sav(temp_path)[0]
                    os.remove(temp_path) # Clean up temp file
                else:
                    return f"Error: Unsupported data file format for '{filename}'. Try CSV, Excel, or SPSS (.sav)."

                if df is not None:
                    info_str = f"DataFrame: {filename}\nShape: {df.shape}\nColumns: {', '.join(df.columns)}\nData Types:\n{df.dtypes.to_string()}\n"
                    h_rows = max(0, min(head_rows, len(df)))
                    if h_rows > 0:
                        info_str += f"Head {h_rows} rows:\n{df.head(h_rows).to_string()}\n"
                    info_str += f"Descriptive Statistics:\n{df.describe().to_string()}\n"
                    return info_str
                else:
                    return f"Could not process data file '{filename}'."
            except Exception as e:
                return f"Error analyzing data file '{filename}': {e}"

        doc_tool = FunctionTool.from_defaults(fn=read_uploaded_document_tool_fn_runtime, name="read_uploaded_document", description="Reads the text content of an uploaded document. Input is the filename (e.g., 'my_document.pdf').")
        df_tool = FunctionTool.from_defaults(fn=analyze_dataframe_tool_fn_runtime, name="analyze_uploaded_dataframe", description="Analyzes an uploaded dataset (CSV, Excel, SPSS). Provides shape, columns, data types, head rows, and descriptive statistics. Input is the filename (e.g., 'my_data.csv').")

        agent_instance = create_orchestrator_agent(dynamic_tools=[
            doc_tool,
            df_tool,
            get_duckduckgo_tool(max_results=max_search_results),
            get_tavily_tool(max_results=max_search_results),
            get_wikipedia_tool(),
            get_semantic_scholar_tool_for_agent(max_results=max_search_results),
            get_web_scraper_tool_for_agent(),
            get_rag_tool_for_agent(),
            get_coder_tools() # Coder tools don't need runtime access to app.storage.user directly
        ])
        print("AI agent initialized successfully.")
        return agent_instance, None
    except Exception as e:
        error_message = f"Failed to initialize AI agent: {e}"
        print(error_message)
        return None, error_message

# --- User Session Management ---

def _get_user_id_from_cookie() -> Optional[str]:
    return app.storage.client.get("user_id")

def _set_user_id_cookie(user_id: str):
    app.storage.client["user_id"] = user_id

def _delete_user_id_cookie():
    if "user_id" in app.storage.client:
        del app.storage.client["user_id"]

def _get_ltm_pref_from_cookie() -> bool:
    pref = app.storage.client.get("long_term_memory_pref")
    return True if pref is None else (str(pref).lower() == 'true' or str(pref) == '1')

def _set_ltm_pref_cookie(enabled: bool):
    app.storage.client["long_term_memory_pref"] = str(enabled)

# Removed _load_user_data_from_hf function
# Removed _save_to_hf function
# Removed save_chat_history_to_hf function
# Removed save_chat_metadata_to_hf function

def format_llama_chat_history(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Formats chat history from UI format to LlamaIndex ChatMessage format."""
    llama_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            llama_messages.append(ChatMessage(role=MessageRole.USER, content=content))
        elif role == "assistant":
            llama_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
        elif role == "tool":
            llama_messages.append(ChatMessage(role=MessageRole.TOOL, content=content))
    return llama_messages[-MAX_CHAT_HISTORY_MESSAGES:] # Truncate to avoid exceeding context window

async def get_agent_response_stream(query: str, chat_history: List[ChatMessage]) -> AsyncGenerator[str, None]:
    agent = app.storage.user.get(AGENT_INSTANCE_KEY)
    if not agent:
        yield "Error: AI agent not initialized."
        return

    try:
        # Settings.llm.temperature is set globally during initialize_agent_settings
        # If you want to dynamically change it per chat, you'd need to re-initialize the LLM or agent
        # For now, we assume it's set once.
        
        response_stream = await agent.astream_chat(query, chat_history)
        async for chunk in response_stream.response_gen:
            yield chunk
    except Exception as e:
        print(f"Error getting agent response: {e}")
        yield f"An error occurred while generating a response: {e}"

async def handle_user_input_from_ui(user_input: str):
    s = app.storage.user # Get a fresh copy of user storage
    chat_id = s["current_chat_id"]
    messages = s["messages"][chat_id] # Get the specific chat's messages

    messages.append({"role": "user", "content": user_input})
    save_chat_history_to_hf(s["user_id"], chat_id, messages)
    
    # Clear suggested prompts after user input
    s["suggested_prompts"] = []
    ui.find_by_id("suggested-prompts-container").refresh() # Refresh suggested prompts UI

    # Format history for agent, excluding the latest user message for prompt generation
    history_for_agent = format_llama_chat_history(messages)

    # Stream agent response
    full_response_content = ""
    # Add a placeholder for the assistant's response
    messages.append({"role": "assistant", "content": ""})
    
    async for chunk in get_agent_response_stream(user_input, history_for_agent):
        full_response_content += chunk
        messages[-1]["content"] = full_response_content
        s["messages"][chat_id] = messages # Update observable dict
        # Scroll to bottom and refresh chat UI
        ui.run_javascript(f'document.getElementById("chat-messages-container").scrollTop = document.getElementById("chat-messages-container").scrollHeight;')
        await ui.update(ui.find_by_id("chat-messages-container"))

    save_chat_history_to_hf(s["user_id"], chat_id, messages)

    # Generate new suggested prompts based on updated history
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in messages))
    ui.find_by_id("suggested-prompts-container").refresh() # Refresh suggested prompts UI


async def new_chat_from_ui():
    s = app.storage.user
    new_chat_id = str(uuid.uuid4())
    prefix = "Idea" if s["long_term_memory_enabled"] else "Session"
    
    # Generate a unique name for the new chat
    existing_nums = [int(re.search(r'\d+', name).group()) for name in s["chat_metadata"].values() if re.search(r'\d+', name)]
    next_num = max(existing_nums) + 1 if existing_nums else 1
    new_name = f"{prefix} {next_num}"

    s["chat_metadata"][new_chat_id] = new_name
    s["messages"][new_chat_id] = ObservableDict([{"role": "assistant", "content": get_initial_greeting_text_cached()}])
    s["current_chat_id"] = new_chat_id
    
    save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
    save_chat_history_to_hf(s["user_id"], new_chat_id, s["messages"][new_chat_id])
    
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple()) # Generate initial prompts for new chat
    
    # Refresh relevant UI components
    await ui.update(ui.find_by_id("chat-list-container"), ui.find_by_id("chat-messages-container"), ui.find_by_id("suggested-prompts-container"))
    ui.run_javascript(f'document.getElementById("chat-messages-container").scrollTop = document.getElementById("chat-messages-container").scrollHeight;')


async def delete_chat_from_ui(chat_id: str):
    s = app.storage.user
    if chat_id in s["chat_metadata"]:
        del s["chat_metadata"][chat_id]
        if chat_id in s["messages"]: # Ensure messages for this chat are also deleted
            del s["messages"][chat_id]
        
        save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
        
        # If the deleted chat was the current one, switch to another or create new
        if s["current_chat_id"] == chat_id:
            if s["chat_metadata"]:
                s["current_chat_id"] = list(s["chat_metadata"].keys())[0]
                current_messages = s["messages"].get(s["current_chat_id"], [])
                s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in current_messages))
            else:
                await new_chat_from_ui() # Create a new chat if no others exist
        
        # Refresh relevant UI components
        await ui.update(ui.find_by_id("chat-list-container"), ui.find_by_id("chat-messages-container"), ui.find_by_id("suggested-prompts-container"))
        ui.run_javascript(f'document.getElementById("chat-messages-container").scrollTop = document.getElementById("chat-messages-container").scrollHeight;')


async def rename_chat_from_ui(chat_id: str, new_name: str):
    s = app.storage.user
    if chat_id in s["chat_metadata"]:
        s["chat_metadata"][chat_id] = new_name
        save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
        await ui.update(ui.find_by_id("chat-list-container")) # Only need to refresh chat list for name change


async def switch_chat_from_ui(chat_id: str):
    s = app.storage.user
    s["current_chat_id"] = chat_id
    current_messages = s["messages"].get(chat_id, [])
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in current_messages))
    
    # Refresh relevant UI components
    await ui.update(ui.find_by_id("chat-messages-container"), ui.find_by_id("suggested-prompts-container"))
    ui.run_javascript(f'document.getElementById("chat-messages-container").scrollTop = document.getElementById("chat-messages-container").scrollHeight;')


def get_discussion_markdown_from_ui(chat_id: str) -> str:
    s = app.storage.user
    messages = s["messages"].get(chat_id, [])
    markdown_content = f"# Chat Discussion: {s['chat_metadata'].get(chat_id, 'Untitled Chat')}\n\n"
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        markdown_content += f"## {role}\n{content}\n\n"
    return markdown_content

def get_discussion_docx_from_ui(chat_id: str) -> bytes:
    s = app.storage.user
    messages = s["messages"].get(chat_id, [])
    doc = Document()
    doc.add_heading(f"Chat Discussion: {s['chat_metadata'].get(chat_id, 'Untitled Chat')}", level=1)

    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        doc.add_heading(role, level=3)
        doc.add_paragraph(content)
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

async def forget_me_from_ui():
    # The user_id from cookie and HF deletion are no longer the primary mechanism.
    # NiceGUI's app.storage.user is automatically user-specific.
    # Clearing it and reloading achieves the "forget me" effect.

    # Optional: If there was an old user_id in a cookie, delete it.
    _delete_user_id_cookie()
    # Also delete the LTM preference cookie for a full reset.
    if "long_term_memory_pref" in app.storage.client:
        del app.storage.client["long_term_memory_pref"]
    
    # Clear all data associated with the current user in NiceGUI's persistent storage.
    app.storage.user.clear()
    print(f"Cleared app.storage.user for user ID: {app.storage.user.id}") # Log NiceGUI's internal user ID

    # Note: Manually deleting the specific user's .json file from the filesystem
    # (e.g., .nicegui/storage-user-....json) is an option for more permanent deletion if required,
    # but app.storage.user.clear() handles the functional aspect of data removal for subsequent sessions.
    # For this implementation, we rely on clear() and the app not re-populating from elsewhere.

    ui.notify("All your data for this application has been cleared. Refreshing...", type='warning')
    # Force a full page reload to re-initialize everything from scratch for this user.
    await ui.run_javascript("setTimeout(() => window.location.reload(), 1500);")


async def set_long_term_memory_from_ui(enabled: bool):
    s = app.storage.user
    s["long_term_memory_enabled"] = enabled # Update the value in app.storage.user first

    # Explicitly update the client-side cookie to match the new preference.
    # This ensures that if initialize_user_session_storage with force_reload=True
    # uses _get_ltm_pref_from_cookie (which it does for new/cleared sessions),
    # it will pick up the *newly set* preference.
    app.storage.client["long_term_memory_pref"] = str(enabled)

    ui.notify(f"Long-term memory {'en' if enabled else 'dis'}abled.", type='info')
    
    # Re-initialize user session storage to apply LTM change (e.g., chat naming conventions)
    await initialize_user_session_storage(force_reload=True)
    
    # After force_reload, s["long_term_memory_enabled"] will be correctly set
    # because _get_ltm_pref_from_cookie will read the updated cookie value.

    # Refresh the main UI content to reflect changes in chat list/messages
    ui.find_by_id("chat-list-container").refresh()
    ui.find_by_id("chat-messages-container").refresh()
    ui.find_by_id("suggested-prompts-container").refresh()


async def regenerate_last_response_from_ui():
    s = app.storage.user
    chat_id = s["current_chat_id"]
    messages = s["messages"][chat_id]

    if not messages:
        return # Nothing to regenerate if chat is empty

    # Find the last user message
    last_user_message_idx = -1
    for i in reversed(range(len(messages))):
        if messages[i]["role"] == "user":
            last_user_message_idx = i
            break
    
    if last_user_message_idx == -1:
        return # No user message found to regenerate from

    # Remove the last assistant response (if any) and subsequent tool messages
    messages_to_keep = messages[:last_user_message_idx + 1]
    
    s["messages"][chat_id] = messages_to_keep
    save_chat_history_to_hf(s["user_id"], chat_id, messages_to_keep)
    
    # Get the content of the last user message
    user_input = messages_to_keep[-1]["content"]

    # Format history for agent, up to the last user message
    history_for_agent = format_llama_chat_history(messages_to_keep)

    # Clear suggested prompts
    s["suggested_prompts"] = []
    ui.find_by_id("suggested-prompts-container").refresh()

    # Add a placeholder for the assistant's response
    messages_to_keep.append({"role": "assistant", "content": ""})

    full_resp = ""
    async for chunk in get_agent_response_stream(user_input, chat_history=history_for_agent):
        full_resp += chunk
        messages_to_keep[-1]["content"] = full_resp
        s["messages"][chat_id] = messages_to_keep # Update observable dict
        ui.run_javascript(f'document.getElementById("chat-messages-container").scrollTop = document.getElementById("chat-messages-container").scrollHeight;')
        await ui.update(ui.find_by_id("chat-messages-container"))
    
    save_chat_history_to_hf(s["user_id"], chat_id, messages_to_keep)

    # Generate new suggested prompts based on updated history
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in messages_to_keep))
    ui.find_by_id("suggested-prompts-container").refresh()


async def handle_file_upload_app_impl(e: UploadEventArguments):
    s = app.storage.user
    file_name = e.name
    file_content = e.content.read() # Read content as bytes
    content_type = e.content_type

    # Save file to UI_ACCESSIBLE_WORKSPACE for potential code interpreter access
    workspace_path = app.storage.general.get("UI_ACCESSIBLE_WORKSPACE")
    os.makedirs(workspace_path, exist_ok=True)
    file_path_in_ws = os.path.join(workspace_path, file_name)
    try:
        with open(file_path_in_ws, "wb") as f:
            f.write(file_content)
        ui.notify(f"File '{file_name}' saved to workspace.", type='positive')
    except Exception as err:
        ui.notify(f"Error saving '{file_name}' to workspace: {err}", type='negative')
        return

    ptype, pname = None, None
    # Process based on content type or extension
    if content_type and ("text" in content_type or "json" in content_type or "xml" in content_type or file_name.lower().endswith(('.txt', '.md', '.json', '.xml'))):
        s["uploaded_documents"][file_name] = file_content
        _save_to_hf(s["user_id"], "uploaded_documents", s["uploaded_documents"])
        ptype, pname = "document", file_name
    elif content_type and ("csv" in content_type or "excel" in content_type or "spreadsheetml" in content_type or file_name.lower().endswith(('.csv', '.xls', '.xlsx', '.sav'))):
        s["uploaded_dataframes"][file_name] = file_content
        _save_to_hf(s["user_id"], "uploaded_dataframes", s["uploaded_dataframes"])
        ptype, pname = "dataframe", file_name
    else:
        ui.notify(f"Unsupported file type '{content_type}' for '{file_name}'. Only text-based documents and common data files are supported.", type='warning')
        # Remove from workspace if not supported
        if os.path.exists(file_path_in_ws):
            os.remove(file_path_in_ws)
        return

    if pname:
        msg = f"Received {ptype}: `{pname}`. You can ask to `read_uploaded_{ptype}('{pname}')` or `analyze_uploaded_{ptype}('{pname}')`."
        # Ensure the message list for the current chat exists
        if s["current_chat_id"] not in s["messages"]:
            s["messages"][s["current_chat_id"]] = [] # Initialize as list
        s["messages"][s["current_chat_id"]].append({"role": "assistant", "content": msg})
        # Removed: save_chat_history_to_hf(s["user_id"], s["current_chat_id"], s["messages"][s["current_chat_id"]])
        # NiceGUI will auto-persist changes to s["messages"]
    
    ui.find_by_id("chat-messages-container").refresh()
    ui.find_by_id("uploaded-files-container").refresh()


async def remove_file_app_impl(file_type: str, file_name: str):
    s = app.storage.user
    removed = False
    if file_type == "document" and file_name in s.get("uploaded_documents", {}):
        del s["uploaded_documents"][file_name]
        _save_to_hf(s["user_id"], "uploaded_documents", s["uploaded_documents"])
        removed = True
    elif file_type == "dataframe" and file_name in s.get("uploaded_dataframes", {}):
        del s["uploaded_dataframes"][file_name]
        _save_to_hf(s["user_id"], "uploaded_dataframes", s["uploaded_dataframes"])
        removed = True
    
    if removed:
        ui.notify(f"'{file_name}' removed from session.", type='info')
        # Also remove from the UI_ACCESSIBLE_WORKSPACE
        workspace_path = app.storage.general.get("UI_ACCESSIBLE_WORKSPACE")
        f_path_ws = os.path.join(workspace_path, file_name)
        if os.path.exists(f_path_ws):
            try:
                os.remove(f_path_ws)
                ui.notify(f"Disk file '{file_name}' deleted.", type='positive')
            except Exception as e:
                ui.notify(f"Error deleting disk file: {e}", type='negative')
    
    ui.find_by_id("uploaded-files-container").refresh()
    return removed


@functools.lru_cache(maxsize=10)
def generate_suggested_prompts_cached(chat_history_tuple: tuple) -> List[str]:
    """
    Generates suggested prompts based on chat history, with caching.
    The chat_history_tuple should be a hashable representation of the chat history.
    """
    # Convert the hashable tuple back to a list of dicts for the agent function
    chat_history = []
    for frozen_msg in chat_history_tuple:
        msg_dict = {}
        for k, v in frozen_msg:
            msg_dict[k] = v
        chat_history.append(msg_dict)
    
    # Import DEFAULT_PROMPTS here to avoid circular dependency if agent imports app
    from agent import DEFAULT_PROMPTS
    return DEFAULT_PROMPTS if not chat_history else generate_suggested_prompts(chat_history)

@simple_once_cache
def get_initial_greeting_text_cached() -> str:
    return generate_llm_greeting()

# --- NiceGUI UI Setup ---

async def initialize_user_session_storage(force_reload: bool = False):
    """
    Initializes or loads user session data into app.storage.user.
    This runs once per client connection.
    """
    s = app.storage.user # app.storage.user is already persistent and loaded by NiceGUI

    # Determine if this is effectively a new user session from the perspective of our application's data model
    # This happens if force_reload is true, or if our application-specific 'user_id' isn't in app.storage.user yet.
    is_new_app_session = force_reload or not s.get("user_id")

    if force_reload:
        s.clear() # Wipe slate clean if forcing reload

    if is_new_app_session:
        # Initialize for a new user session or a session that was force_reloaded

        # User ID management:
        # Try to get user_id from cookie (if it was set by an older version or previous non-reloaded session).
        # If not in cookie, or if we are force_reloading (in which case cookie shouldn't dictate a link to old data),
        # generate a fresh one.
        # This app-specific user_id is then stored in app.storage.user.
        user_id_from_cookie = _get_user_id_from_cookie()
        if user_id_from_cookie and not force_reload:
            s["user_id"] = user_id_from_cookie
        else:
            s["user_id"] = str(uuid.uuid4())
        # Update the cookie as well. While app.storage.user is primary, cookie might be useful for other contexts or debugging.
        _set_user_id_cookie(s["user_id"])

        # Initialize data structures
        s.setdefault("chat_metadata", ObservableDict())
        s.setdefault("messages", ObservableDict())
        s.setdefault("current_chat_id", None)
        s.setdefault("uploaded_documents", ObservableDict())
        s.setdefault("uploaded_dataframes", ObservableDict())

        # LTM Preference for new/reloaded session:
        # Set from cookie (if available, for transition) or use default from _get_ltm_pref_from_cookie().
        # This value is immediately persisted in app.storage.user.
        s["long_term_memory_enabled"] = _get_ltm_pref_from_cookie()

        print(f"User session initialized/reloaded. App User ID: {s['user_id']}. LTM enabled: {s['long_term_memory_enabled']}")
        await initialize_active_chat_session(new_chat=True)
    else:
        # Existing user session: app.storage.user was populated by NiceGUI.
        # Our app_user_id (s["user_id"]) is already present.

        # Ensure data structures are the correct observable types (safeguard).
        if not isinstance(s.get("chat_metadata"), ObservableDict):
            s["chat_metadata"] = ObservableDict(s.get("chat_metadata", {}))
        if not isinstance(s.get("messages"), ObservableDict):
            s["messages"] = ObservableDict(s.get("messages", {}))
            # Ensure inner message lists are standard lists.
            for chat_id_key in list(s["messages"].keys()):
                if chat_id_key in s["messages"] and not isinstance(s["messages"][chat_id_key], list):
                     s["messages"][chat_id_key] = list(s["messages"].get(chat_id_key, []))
        if not isinstance(s.get("uploaded_documents"), ObservableDict):
            s["uploaded_documents"] = ObservableDict(s.get("uploaded_documents", {}))
        if not isinstance(s.get("uploaded_dataframes"), ObservableDict):
            s["uploaded_dataframes"] = ObservableDict(s.get("uploaded_dataframes", {}))

        # LTM Preference for existing user:
        # If "long_term_memory_enabled" is not yet in app.storage.user (e.g. older session data),
        # initialize it from cookie (for transition) or default.
        # Otherwise, the value already in app.storage.user is used.
        if "long_term_memory_enabled" not in s:
            s["long_term_memory_enabled"] = _get_ltm_pref_from_cookie()
        
        print(f"Existing user session loaded. App User ID: {s['user_id']}. LTM enabled: {s['long_term_memory_enabled']}")

        # Set current chat or create a new one if none exist or current is invalid
        if not s.get("chat_metadata") or not s.get("current_chat_id") or s.get("current_chat_id") not in s.get("chat_metadata"):
            await initialize_active_chat_session(new_chat=True)
        else:
            if not s.get("current_chat_id") or s["current_chat_id"] not in s["chat_metadata"]:
                s["current_chat_id"] = list(s["chat_metadata"].keys())[0] # Set to first chat if not set
            await initialize_active_chat_session() # Initialize with existing chat

    # Set default LLM/search parameters if not already set
    s.setdefault("llm_temperature", 0.7)
    s.setdefault("llm_verbosity", 3)
    s.setdefault("search_results_count", 5)
    s.setdefault("suggested_prompts", []) # Will be populated by initialize_active_chat_session


async def initialize_active_chat_session(new_chat: bool = False):
    s = app.storage.user
    
    # If new_chat is True, or no current chat, or current chat has no messages
    if new_chat or not s.get("current_chat_id") or not s["messages"].get(s["current_chat_id"]):
        await new_chat_from_ui() # This function handles setting current_chat_id and initial messages
        # new_chat_from_ui already generates initial prompts
    else:
        # For existing chat, ensure messages are ObservableDict and generate prompts
        current_chat_id = s["current_chat_id"]
        # Ensure the message list for the current chat is an ObservableDict for reactivity
        if not isinstance(s["messages"][current_chat_id], ObservableDict):
            s["messages"][current_chat_id] = ObservableDict(s["messages"][current_chat_id])
        
        # Generate suggested prompts based on the current chat's history
        s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in s["messages"][current_chat_id]))


@ui.page('/')
async def main_page(client: Client):
    # Initialize LLM settings once globally
    llm_ok, llm_error = setup_global_llm_settings_cached()
    if not llm_ok:
        with ui.column().classes('absolute-center items-center'):
            ui.icon('error', size='5rem').classes('text-negative')
            ui.label('LLM Initialization Error').classes('text-h4 text-negative')
            ui.label(llm_error).classes('text-body1 text-negative text-center')
        return

    # Initialize agent once per app instance (cached)
    # The max_search_results can be dynamic, but the agent instance itself is cached
    agent_instance, agent_error = setup_agent_cached(max_search_results=app.storage.user.get("search_results_count", 5))
    if not agent_instance:
        with ui.column().classes('absolute-center items-center'):
            ui.icon('error', size='5rem').classes('text-negative')
            ui.label('AI Agent Initialization Error').classes('text-h4 text-negative')
            ui.label(agent_error).classes('text-body1 text-negative text-center')
        return
    
    # Store agent instance in user storage for access across requests
    app.storage.user[AGENT_INSTANCE_KEY] = agent_instance
    # Store UI_ACCESSIBLE_WORKSPACE path in general storage for tools to access
    app.storage.general["UI_ACCESSIBLE_WORKSPACE"] = os.path.join(PROJECT_ROOT, "code_interpreter_ws")

    # Initialize user session storage (chat history, user ID, etc.)
    await initialize_user_session_storage()

    # Register app-level callbacks with the UI module
    # Import ui_module here to avoid circular imports at the top level
    import ui as ui_module
    ui_module.app_callbacks.handle_user_input = handle_user_input_from_ui
    ui_module.app_callbacks.reset_chat = new_chat_from_ui # new_chat_from_ui effectively resets to a new chat
    ui_module.app_callbacks.new_chat = new_chat_from_ui
    ui_module.app_callbacks.delete_chat = delete_chat_from_ui
    ui_module.app_callbacks.rename_chat = rename_chat_from_ui
    ui_module.app_callbacks.switch_chat = switch_chat_from_ui
    ui_module.app_callbacks.get_discussion_markdown = get_discussion_markdown_from_ui
    ui_module.app_callbacks.get_discussion_docx = get_discussion_docx_from_ui
    ui_module.app_callbacks.forget_me = forget_me_from_ui
    ui_module.app_callbacks.set_long_term_memory = set_long_term_memory_from_ui
    ui_module.app_callbacks.regenerate_last_response = regenerate_last_response_from_ui
    ui_module.app_callbacks.handle_file_upload = handle_file_upload_app_impl
    ui_module.app_callbacks.remove_file = remove_file_app_impl

    # Build the main UI content
    @ui.refreshable
    async def build_main_ui_content():
        current_s_user = app.storage.user # Get fresh copy of user storage for this render pass
        
        # Pass the current state from app.storage.user to the UI function
        ui_module.create_nicegui_interface(
            ext_handle_user_input_callback=ui_module.app_callbacks.handle_user_input,
            ext_reset_callback=ui_module.app_callbacks.reset_chat,
            ext_new_chat_callback=ui_module.app_callbacks.new_chat,
            ext_delete_chat_callback=ui_module.app_callbacks.delete_chat,
            ext_rename_chat_callback=ui_module.app_callbacks.rename_chat,
            ext_switch_chat_callback=ui_module.app_callbacks.switch_chat,
            ext_get_discussion_markdown_callback=ui_module.app_callbacks.get_discussion_markdown,
            ext_get_discussion_docx_callback=ui_module.app_callbacks.get_discussion_docx,
            ext_forget_me_callback=ui_module.app_callbacks.forget_me,
            ext_set_long_term_memory_callback=ui_module.app_callbacks.set_long_term_memory,
            ext_regenerate_last_response_callback=ui_module.app_callbacks.regenerate_last_response,
            ext_handle_file_upload_app_callback=ui_module.app_callbacks.handle_file_upload,
            ext_remove_file_app_callback=ui_module.app_callbacks.remove_file,
            # Pass state values that ui.py might need for initial setup or direct binding
            current_chat_id_val=current_s_user.get("current_chat_id"),
            chat_metadata_val=current_s_user.get("chat_metadata", {}),
            llm_temp_val=current_s_user.get("llm_temperature", 0.7),
            llm_verb_val=current_s_user.get("llm_verbosity", 3),
            search_count_val=current_s_user.get("search_results_count", 5),
            ltm_enabled_val=current_s_user.get("long_term_memory_enabled", False),
            suggested_prompts_list_val=current_s_user.get("suggested_prompts", []),
            # Messages and uploaded files are accessed directly by ui.py from app.storage.user
            # but we need to ensure they are ObservableDicts for reactivity.
            # This is handled in initialize_user_session_storage.
        )
        
        # After UI structure is created by ui_module, populate dynamic parts
        # These refreshable functions are now defined in ui.py and called via app_callbacks
        # We just need to trigger their initial refresh here.
        ui.find_by_id("chat-messages-container").refresh()
        ui.find_by_id("uploaded-files-container").refresh()
        ui.find_by_id("suggested-prompts-container").refresh()

    await build_main_ui_content() # Initial build of the UI


if __name__ in {"__main__", "__mp_main__"}:
    # Ensure the workspace directory exists
    workspace_dir = os.path.join(PROJECT_ROOT, "code_interpreter_ws")
    os.makedirs(workspace_dir, exist_ok=True)
    # Add static files for the workspace so the code interpreter can access them via URL
    app.add_static_files('/workspace', workspace_dir)

    # Ensure the NiceGUI storage path exists
    storage_path_dir = os.path.join(PROJECT_ROOT, "huggingface/user_memories")
    os.makedirs(storage_path_dir, exist_ok=True)
    print(f"NiceGUI user data will be stored in: {storage_path_dir}")

    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Neither GOOGLE_API_KEY nor OPENAI_API_KEY is set. LLM functionality may be limited.")
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️ TAVILY_API_KEY not set. Search functionality may be limited.")
    if not os.getenv("HF_TOKEN"): # This is still relevant for RAG, not user memories now
        print("⚠️ HF_TOKEN not set. Hugging Face RAG functionality might be affected.")

    ui.run(
        title="ESI - NiceGUI",
        host="0.0.0.0",
        port=8080,
        reload=True, # Set to False for production
        storage_secret="a_very_secret_key_for_storage_12345", # IMPORTANT: Change this to a strong, unique secret for production!
        storage_path=storage_path_dir
    )
