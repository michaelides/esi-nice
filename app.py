import functools
import os
import uuid
import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from nicegui import app, ui, Client
from nicegui.events import UploadEventArguments
from nicegui.storage import ObservableDict

from agent import create_orchestrator_agent, generate_llm_greeting, generate_suggested_prompts, initialize_settings
from config import HF_TOKEN, HF_USER_MEMORIES_DATASET_ID, PROJECT_ROOT

# LlamaIndex imports
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

# File processing imports
from docx import Document
from io import BytesIO
import pandas as pd
import pyreadstat # For SPSS processing

# Hugging Face related
try:
    from huggingface_hub import HfFileSystem
    from datasets import load_dataset, Dataset
    fs = HfFileSystem()
except ImportError:
    print("Hugging Face Hub or datasets not installed. User data will not be persisted to HF.")
    fs = None

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

AGENT_INSTANCE_KEY = "esi_orchestrator_agent_instance"
STORAGE_SECRET = os.environ.get("STORAGE_SECRET",)

# Configure NiceGUI storage
app.storage.configure(
    user_dir=os.path.join(PROJECT_ROOT, os.path.dirname(HF_USER_MEMORIES_DATASET_ID)),
    general_file=os.path.join(PROJECT_ROOT, '.nicegui', 'storage-general.json'),
    secret=STORAGE_SECRET,
)

# Configure NiceGUI static files
app.add_static_files('/ragdb', os.path.join(PROJECT_ROOT, 'ragdb'))
app.add_static_files('/workspace_ui_accessible', os.path.join(PROJECT_ROOT, 'workspace_ui_accessible'))

def simple_once_cache(func):
    _cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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
    print("Initializing LLM settings...")
    try:
        initialize_settings()
        print("LLM settings initialized.")
        return True, None
    except Exception as e:
        error_message = f"Fatal Error: Could not initialize LLM settings. {e}"
        print(error_message)
        return False, error_message

@functools.lru_cache(maxsize=1)
def setup_agent_cached(client: Client, max_search_results: int) -> tuple[Any | None, str | None]:
    print(f"Initializing AI agent with max_search_results={max_search_results}...")
    try:
        from tools import get_duckduckgo_tool, get_tavily_tool, get_wikipedia_tool, get_semantic_scholar_tool_for_agent, get_web_scraper_tool_for_agent, get_rag_tool_for_agent, get_coder_tools

        def read_uploaded_document_tool_fn_runtime(filename: str) -> str:
            uploaded_docs = app.storage.user.get("uploaded_documents", {})
            content_bytes = uploaded_docs.get(filename)
            if content_bytes:
                return content_bytes.decode('utf-8', errors='replace')
            else:
                return f"Error: Document '{filename}' not found in uploaded documents. Available: {list(uploaded_docs.keys())}"

        def analyze_dataframe_tool_fn_runtime(filename: str, head_rows: int = 5) -> str:
            uploaded_dfs = app.storage.user.get("uploaded_dataframes", {})
            df_content_bytes = uploaded_dfs.get(filename)
            if not df_content_bytes:
                return f"Error: DataFrame '{filename}' not found. Available: {list(uploaded_dfs.keys())}"
            
            df = None
            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(BytesIO(df_content_bytes))
                elif filename.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(BytesIO(df_content_bytes))
                elif filename.lower().endswith('.sav'):
                    temp_path = os.path.join(app.storage.general.get("UI_ACCESSIBLE_WORKSPACE"), filename)
                    with open(temp_path, 'wb') as f:
                        f.write(df_content_bytes)
                    df, meta = pyreadstat.read_sav(temp_path)
                    os.remove(temp_path)
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
            get_coder_tools()
        ])
        client.storage.private[AGENT_INSTANCE_KEY] = agent_instance
        print("AI agent initialized successfully.")
        return agent_instance, None
    except Exception as e:
        error_message = f"Failed to initialize AI agent: {e}"
        print(error_message)
        return None, error_message

# --- User Session Management ---

def _get_user_id_from_cookie() -> Optional[str]:
    return app.storage.user.get("user_id")

def _set_user_id_cookie(user_id: str):
    app.storage.user["user_id"] = user_id

def _delete_user_id_cookie():
    if "user_id" in app.storage.user:
        del app.storage.user["user_id"]

def _get_ltm_pref_from_cookie() -> bool:
    return app.storage.user.get("long_term_memory_enabled", True)

def _set_ltm_pref_cookie(enabled: bool):
    app.storage.user["long_term_memory_enabled"] = enabled

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
    return llama_messages

async def get_agent_response_stream(query: str, chat_history: List[ChatMessage]) -> AsyncGenerator[str, None]:
    agent = ui.current.client.storage.private.get(AGENT_INSTANCE_KEY)
    if not agent:
        yield "Error: AI agent not initialized. Please refresh the page."
        return

    try:
        response_stream = await agent.astream_chat(query, chat_history)
        async for chunk in response_stream.response_gen:
            yield chunk
    except Exception as e:
        print(f"Error getting agent response: {e}")
        yield f"An error occurred while generating a response: {e}"

async def handle_user_input_from_ui(user_input: str):
    s = app.storage.user
    chat_id = s["current_chat_id"]
    messages = s["messages"][chat_id]

    messages.append({"role": "user", "content": user_input})
    
    s["suggested_prompts"] = []
    if 'suggested_prompts_container_ref' in s:
        await s['suggested_prompts_container_ref'].refresh()

    history_for_agent = format_llama_chat_history(messages)

    full_response_content = ""
    messages.append({"role": "assistant", "content": ""})
    
    async for chunk in get_agent_response_stream(user_input, history_for_agent):
        full_response_content += chunk
        messages[-1]["content"] = full_response_content
        s["messages"][chat_id] = messages
        if 'chat_area_ref' in s:
            await ui.run_javascript(f'document.getElementById("{s["chat_area_ref"].id}").scrollTop = document.getElementById("{s["chat_area_ref"].id}").scrollHeight;')
            await s['chat_area_ref'].refresh()

    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in messages))
    if 'suggested_prompts_container_ref' in s:
        await s['suggested_prompts_container_ref'].refresh()


async def new_chat_from_ui():
    s = app.storage.user
    new_chat_id = str(uuid.uuid4())
    prefix = "Idea" if s["long_term_memory_enabled"] else "Session"
    
    existing_nums = [int(re.search(r'\d+', name).group()) for name in s["chat_metadata"].values() if re.search(r'\d+', name)]
    next_num = max(existing_nums) + 1 if existing_nums else 1
    new_name = f"{prefix} {next_num}"

    s["chat_metadata"][new_chat_id] = new_name
    s["messages"][new_chat_id] = [{"role": "assistant", "content": await get_initial_greeting_text_cached()}]
    s["current_chat_id"] = new_chat_id
    
    save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
    
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple())
    
    if 'chat_list_container_ref' in s:
        await s['chat_list_container_ref'].refresh()
    if 'chat_area_ref' in s:
        await s['chat_area_ref'].refresh()
    if 'suggested_prompts_container_ref' in s:
        await s['suggested_prompts_container_ref'].refresh()

    if 'chat_area_ref' in s:
        await ui.run_javascript(f'document.getElementById("{s["chat_area_ref"].id}").scrollTop = document.getElementById("{s["chat_area_ref"].id}").scrollHeight;')


async def delete_chat_from_ui(chat_id: str):
    s = app.storage.user
    if chat_id in s["chat_metadata"]:
        del s["chat_metadata"][chat_id]
        if chat_id in s["messages"]:
            del s["messages"][chat_id]
        
        save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
        
        if s["current_chat_id"] == chat_id:
            if s["chat_metadata"]:
                s["current_chat_id"] = list(s["chat_metadata"].keys())[0]
                current_messages = s["messages"].get(s["current_chat_id"], [])
                s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in current_messages))
            else:
                await new_chat_from_ui()
        
        if 'chat_list_container_ref' in s:
            await s['chat_list_container_ref'].refresh()
        if 'chat_area_ref' in s:
            await s['chat_area_ref'].refresh()
        if 'suggested_prompts_container_ref' in s:
            await s['suggested_prompts_container_ref'].refresh()
        if 'chat_area_ref' in s:
            await ui.run_javascript(f'document.getElementById("{s["chat_area_ref"].id}").scrollTop = document.getElementById("{s["chat_area_ref"].id}").scrollHeight;')


async def rename_chat_from_ui(chat_id: str, new_name: str):
    s = app.storage.user
    if chat_id in s["chat_metadata"]:
        s["chat_metadata"][chat_id] = new_name
        save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
        if 'chat_list_container_ref' in s:
            await s['chat_list_container_ref'].refresh()


async def switch_chat_from_ui(chat_id: str):
    s = app.storage.user
    s["current_chat_id"] = chat_id
    current_messages = s["messages"].get(chat_id, [])
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in current_messages))
    
    if 'chat_area_ref' in s:
        await s['chat_area_ref'].refresh()
    if 'suggested_prompts_container_ref' in s:
        await s['suggested_prompts_container_ref'].refresh()
    if 'chat_area_ref' in s:
        await ui.run_javascript(f'document.getElementById("{s["chat_area_ref"].id}").scrollTop = document.getElementById("{s["chat_area_ref"].id}").scrollHeight;')


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
    _delete_user_id_cookie()
    app.storage.user.clear()
    print(f"Cleared app.storage.user for user ID: {app.storage.user.id}")

    ui.notify("All your data for this application has been cleared. Refreshing...", type='warning')
    await ui.run_javascript("setTimeout(() => window.location.reload(), 1500);")


async def set_long_term_memory_from_ui(enabled: bool):
    s = app.storage.user
    s["long_term_memory_enabled"] = enabled

    ui.notify(f"Long-term memory {'en' if enabled else 'dis'}abled.", type='info')
    
    await initialize_user_session_storage(force_reload=True)
    
    if 'chat_list_container_ref' in s:
        await s['chat_list_container_ref'].refresh()
    if 'chat_area_ref' in s:
        await s['chat_area_ref'].refresh()
    if 'suggested_prompts_container_ref' in s:
        await s['suggested_prompts_container_ref'].refresh()


async def regenerate_last_response_from_ui():
    s = app.storage.user
    chat_id = s["current_chat_id"]
    messages = s["messages"][chat_id]

    if not messages:
        return

    last_user_message_idx = -1
    for i in reversed(range(len(messages))):
        if messages[i]["role"] == "user":
            last_user_message_idx = i
            break
    
    if last_user_message_idx == -1:
        return

    messages_to_keep = messages[:last_user_message_idx + 1]
    
    s["messages"][chat_id] = messages_to_keep
    
    user_input = messages_to_keep[-1]["content"]

    history_for_agent = format_llama_chat_history(messages_to_keep)

    s["suggested_prompts"] = []
    if 'suggested_prompts_container_ref' in s:
        await s['suggested_prompts_container_ref'].refresh()

    messages_to_keep.append({"role": "assistant", "content": ""})

    full_resp = ""
    async for chunk in get_agent_response_stream(user_input, chat_history=history_for_agent):
        full_resp += chunk
        messages_to_keep[-1]["content"] = full_resp
        s["messages"][chat_id] = messages_to_keep
        if 'chat_area_ref' in s:
            await ui.run_javascript(f'document.getElementById("{s["chat_area_ref"].id}").scrollTop = document.getElementById("{s["chat_area_ref"].id}").scrollHeight;')
            await s['chat_area_ref'].refresh()
    
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in messages_to_keep))
    if 'suggested_prompts_container_ref' in s:
        await s['suggested_prompts_container_ref'].refresh()


async def handle_file_upload_app_impl(file_name: str, file_content_bytes: bytes, content_type: Optional[str]):
    s = app.storage.user
    
    workspace_path = app.storage.general.get("UI_ACCESSIBLE_WORKSPACE")
    os.makedirs(workspace_path, exist_ok=True)
    file_path_in_ws = os.path.join(workspace_path, file_name)
    try:
        with open(file_path_in_ws, "wb") as f:
            f.write(file_content_bytes)
        ui.notify(f"File '{file_name}' saved to workspace.", type='positive')
    except Exception as err:
        ui.notify(f"Error saving '{file_name}' to workspace: {err}", type='negative')
        return

    ptype, pname = None, None
    if content_type and ("text" in content_type or "json" in content_type or "xml" in content_type or file_name.lower().endswith(('.txt', '.md', '.json', '.xml'))):
        s.setdefault("uploaded_documents", ObservableDict())[file_name] = file_content_bytes
        ptype, pname = "document", file_name
    elif content_type and ("csv" in content_type or "excel" in content_type or "spreadsheetml" in content_type or file_name.lower().endswith(('.csv', '.xls', '.xlsx', '.sav'))):
        s.setdefault("uploaded_dataframes", ObservableDict())[file_name] = file_content_bytes
        ptype, pname = "dataframe", file_name
    else:
        ui.notify(f"Unsupported file type '{content_type}' for '{file_name}'. Only text-based documents and common data files are supported.", type='warning')
        if os.path.exists(file_path_in_ws):
            os.remove(file_path_in_ws)
        return

    if pname:
        msg = f"Received {ptype}: `{pname}`. You can ask to `read_uploaded_document('{pname}')` or `analyze_uploaded_dataframe('{pname}')`."
        current_chat_id = s.get("current_chat_id")
        if current_chat_id not in s["messages"]:
            s["messages"][current_chat_id] = []
        s["messages"][current_chat_id].append({"role": "assistant", "content": msg})
    
    if 'chat_area_ref' in s:
        await s['chat_area_ref'].refresh()
    if 'ui_uploaded_files_container_ref' in s:
        await s['ui_uploaded_files_container_ref'].refresh()


async def remove_file_app_impl(file_type: str, file_name: str):
    s = app.storage.user
    removed = False
    if file_type == "document" and file_name in s.get("uploaded_documents", {}):
        del s["uploaded_documents"][file_name]
        removed = True
    elif file_type == "dataframe" and file_name in s.get("uploaded_dataframes", {}):
        del s["uploaded_dataframes"][file_name]
        removed = True
    
    if removed:
        ui.notify(f"'{file_name}' removed from session.", type='info')
        workspace_path = app.storage.general.get("UI_ACCESSIBLE_WORKSPACE")
        f_path_ws = os.path.join(workspace_path, file_name)
        if os.path.exists(f_path_ws):
            try:
                os.remove(f_path_ws)
                ui.notify(f"Disk file '{file_name}' deleted.", type='positive')
            except Exception as e:
                ui.notify(f"Error deleting disk file: {e}", type='negative')
    
    if 'ui_uploaded_files_container_ref' in s:
        await s['ui_uploaded_files_container_ref'].refresh()
    return removed


@functools.lru_cache(maxsize=10)
def generate_suggested_prompts_cached(chat_history_tuple: tuple) -> List[str]:
    """
    Generates suggested prompts based on chat history, with caching.
    The chat_history_tuple should be a hashable representation of the chat history.
    """
    chat_history = [dict(item) for item in chat_history_tuple]
    
    from agent import DEFAULT_PROMPTS
    return DEFAULT_PROMPTS if not chat_history else generate_suggested_prompts(chat_history)

@simple_once_cache
def get_initial_greeting_text_cached() -> str:
    return generate_llm_greeting()

def save_chat_metadata_to_hf(user_id: str, chat_metadata: Dict[str, str]):
    """Saves chat metadata to the Hugging Face dataset."""
    if not HF_TOKEN:
        print("HF_TOKEN not set. Skipping saving chat metadata to Hugging Face.")
        return

    try:
        dataset_dict = load_dataset(HF_USER_MEMORIES_DATASET_ID, token=HF_TOKEN, split='train')
        
        df = dataset_dict.to_pandas()

        if user_id in df['user_id'].values:
            df.loc[df['user_id'] == user_id, 'chat_metadata'] = json.dumps(chat_metadata)
        else:
            new_row = pd.DataFrame([{"user_id": user_id, "chat_metadata": json.dumps(chat_metadata)}])
            df = pd.concat([df, new_row], ignore_index=True)

        updated_ds = Dataset.from_pandas(df)
        updated_ds.push_to_hub(HF_USER_MEMORIES_DATASET_ID, split='train', token=HF_TOKEN)
        print(f"Chat metadata for user {user_id} saved to Hugging Face Hub.")

    except Exception as e:
        print(f"Error saving chat metadata to Hugging Face: {e}")
        if "Repository Not Found" in str(e) or "404 Client Error" in str(e):
            print(f"Dataset {HF_USER_MEMORIES_DATASET_ID} not found. Attempting to create it.")
            try:
                new_df = pd.DataFrame([{"user_id": user_id, "chat_metadata": json.dumps(chat_metadata)}])
                new_ds = Dataset.from_pandas(new_df)
                new_ds.push_to_hub(HF_USER_MEMORIES_DATASET_ID, split='train', token=HF_TOKEN, create_pr=True)
                print(f"Dataset {HF_USER_MEMORIES_DATASET_ID} created and chat metadata saved.")
            except Exception as create_e:
                print(f"Failed to create and save dataset to Hugging Face: {create_e}")
        else:
            print(f"An unexpected error occurred while saving to Hugging Face: {e}")


async def initialize_user_session_storage(force_reload: bool = False):
    """
    Initializes or loads user session data into app.storage.user.
    This runs once per client connection.
    """
    s = app.storage.user

    is_new_app_session = force_reload or not s.get("user_id")

    if force_reload:
        s.clear()

    if is_new_app_session:
        user_id_from_cookie = _get_user_id_from_cookie()
        if user_id_from_cookie and not force_reload:
            s["user_id"] = user_id_from_cookie
        else:
            s["user_id"] = str(uuid.uuid4())
        _set_user_id_cookie(s["user_id"])

        s.setdefault("chat_metadata", ObservableDict())
        s.setdefault("messages", ObservableDict())
        s.setdefault("current_chat_id", None)
        s.setdefault("uploaded_documents", ObservableDict())
        s.setdefault("uploaded_dataframes", ObservableDict())

        s["long_term_memory_enabled"] = _get_ltm_pref_from_cookie()

        print(f"User session initialized/reloaded. App User ID: {s['user_id']}. LTM enabled: {s['long_term_memory_enabled']}")
        await initialize_active_chat_session(new_chat=True)
    else:
        if not isinstance(s.get("chat_metadata"), ObservableDict):
            s["chat_metadata"] = ObservableDict(s.get("chat_metadata", {}))
        if not isinstance(s.get("messages"), ObservableDict):
            s["messages"] = ObservableDict(s.get("messages", {}))
            for chat_id_key in list(s["messages"].keys()):
                if chat_id_key in s["messages"] and not isinstance(s["messages"][chat_id_key], list):
                     s["messages"][chat_id_key] = list(s["messages"].get(chat_id_key, []))
        if not isinstance(s.get("uploaded_documents"), ObservableDict):
            s["uploaded_documents"] = ObservableDict(s.get("uploaded_documents", {}))
        if not isinstance(s.get("uploaded_dataframes"), ObservableDict):
            s["uploaded_dataframes"] = ObservableDict(s.get("uploaded_dataframes", {}))

        if "long_term_memory_enabled" not in s:
            s["long_term_memory_enabled"] = _get_ltm_pref_from_cookie()
        
        print(f"Existing user session loaded. App User ID: {s['user_id']}. LTM enabled: {s['long_term_memory_enabled']}")

        if not s.get("chat_metadata") or not s.get("current_chat_id") or s.get("current_chat_id") not in s.get("chat_metadata"):
            await initialize_active_chat_session(new_chat=True)
        else:
            if not s.get("current_chat_id") or s["current_chat_id"] not in s["chat_metadata"]:
                s["current_chat_id"] = list(s["chat_metadata"].keys())[0]
            await initialize_active_chat_session()

    s.setdefault("llm_temperature", 0.7)
    s.setdefault("llm_verbosity", 3)
    s.setdefault("search_results_count", 5)
    s.setdefault("suggested_prompts", [])


async def initialize_active_chat_session(new_chat: bool = False):
    s = app.storage.user
    
    current_chat_id = s.get("current_chat_id")
    chat_metadata = s.get("chat_metadata", {})
    messages = s.get("messages", {})

    if new_chat or not current_chat_id or current_chat_id not in chat_metadata:
        new_chat_id = str(uuid.uuid4())
        initial_greeting_message = {"role": "assistant", "content": await get_initial_greeting_text_cached()}
        messages[new_chat_id] = [initial_greeting_message]
        
        prefix = "Idea" if s.get("long_term_memory_enabled", True) else "Session"
        existing_nums = [int(re.search(r'\d+', name).group()) for name in s["chat_metadata"].values() if re.search(r'\d+', name)]
        next_num = max(existing_nums) + 1 if existing_nums else 1
        new_name = f"{prefix} {next_num}"

        s["chat_metadata"][new_chat_id] = new_name
        s["current_chat_id"] = new_chat_id
        
        save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
        print(f"New chat created: {s['chat_metadata'][new_chat_id]} ({new_chat_id})")
    else:
        if not isinstance(messages.get(current_chat_id), list):
            print(f"Warning: Messages for chat {current_chat_id} are not a list. Re-initializing this chat.")
            messages[current_chat_id] = []
        
        if any(not isinstance(msg, dict) for msg in messages.get(current_chat_id, [])):
            print(f"Warning: Some messages in chat {current_chat_id} are not dictionaries. Re-initializing this chat.")
            messages[current_chat_id] = []

        if not messages[current_chat_id]:
            messages[current_chat_id].append({"role": "assistant", "content": await get_initial_greeting_text_cached()})
        
        print(f"Active chat session initialized: {chat_metadata.get(current_chat_id, 'Unknown Chat')} ({current_chat_id})")

    current_chat_messages = messages.get(s["current_chat_id"], [])
    if current_chat_messages:
        s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(frozenset(msg.items()) for msg in current_chat_messages))
    else:
        s["suggested_prompts"] = []


@ui.page('/')
async def main_page(client: Client):
    llm_ok, llm_error = setup_global_llm_settings_cached()
    if not llm_ok:
        with ui.column().classes('absolute-center items-center'):
            ui.icon('error', size='5rem').classes('text-negative')
            ui.label('LLM Initialization Error').classes('text-h4 text-negative')
            ui.label(llm_error).classes('text-body1 text-negative text-center')
        return

    agent_instance, agent_error = setup_agent_cached(client, max_search_results=app.storage.user.get("search_results_count", 5))
    if not agent_instance:
        with ui.column().classes('absolute-center items-center'):
            ui.icon('error', size='5rem').classes('text-negative')
            ui.label('AI Agent Initialization Error').classes('text-h4 text-negative')
            ui.label(agent_error).classes('text-body1 text-negative text-center')
        return
    
    app.storage.general["UI_ACCESSIBLE_WORKSPACE"] = os.path.join(PROJECT_ROOT, "workspace_ui_accessible")

    await initialize_user_session_storage()

    import ui as ui_module
    ui_module.app_callbacks.handle_user_input = handle_user_input_from_ui
    ui_module.app_callbacks.reset_chat = new_chat_from_ui
    ui_module.app_callbacks.new_chat = new_chat_from_ui
    ui_module.app_callbacks.delete_chat = delete_chat_from_ui
    ui_module.app_callbacks.rename_chat = rename_chat_from_ui
    ui_module.app_callbacks.switch_chat = switch_chat_from_ui
    ui_module.app_callbacks.get_discussion_markdown = get_discussion_markdown_from_ui
    ui_module.app_callbacks.get_discussion_docx = get_discussion_docx_from_ui
    ui_module.app_callbacks.forget_me = forget_me_from_ui
    ui_module.app_callbacks.set_long_term_memory = set_long_term_memory_from_ui
    ui_module.app_callbacks.regenerate_last_response = regenerate_last_response_from_ui
    ui_module.app_callbacks.handle_file_upload_app = handle_file_upload_app_impl
    ui_module.app_callbacks.remove_file_app = remove_file_app_impl

    s_user = app.storage.user
    
    chat_area, suggested_prompts_container_ref, uploaded_files_list_refresher_ui, chat_list_container_ref = ui_module.create_nicegui_interface(
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
        ext_regenerate_callback=ui_module.app_callbacks.regenerate_last_response,
        ext_handle_file_upload_app_callback=ui_module.app_callbacks.handle_file_upload_app,
        ext_remove_file_app_callback=ui_module.app_callbacks.remove_file_app,
        current_chat_id_val=s_user.get("current_chat_id"),
        chat_metadata_val=s_user.get("chat_metadata", {}),
        llm_temp_val=s_user.get("llm_temperature", 0.7),
        llm_verb_val=s_user.get("llm_verbosity", 3),
        search_count_val=s_user.get("search_results_count", 5),
        ltm_enabled_val=s_user.get("long_term_memory_enabled", False),
        suggested_prompts_list_val=s_user.get("suggested_prompts", []),
    )
    
    s_user['chat_area_ref'] = chat_area
    s_user['suggested_prompts_container_ref'] = suggested_prompts_container_ref
    s_user['ui_uploaded_files_container_ref'] = uploaded_files_list_refresher_ui
    s_user['chat_list_container_ref'] = chat_list_container_ref

    await ui_module.display_chat_messages(chat_area)
    await uploaded_files_list_refresher_ui.refresh()
    await suggested_prompts_container_ref.refresh()
    await chat_list_container_ref.refresh()


if __name__ in {"__main__", "__mp_main__"}:
    workspace_dir = os.path.join(PROJECT_ROOT, "workspace_ui_accessible")
    os.makedirs(workspace_dir, exist_ok=True)
    app.add_static_files('/workspace', workspace_dir)

    ui.run(
        title="ESI - NiceGUI",
        host="0.0.0.0",
        port=8080,
        reload=True,
        storage_secret=STORAGE_SECRET,
    )
