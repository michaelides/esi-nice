from nicegui import ui, app, Client
import os
import json
import re
import uuid
import datetime
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Coroutine # Added Coroutine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
import time
import functools
import pandas as pd # For DataFrame processing in app.py

# Project specific imports
import ui as ui_module # Renamed to avoid conflict with nicegui.ui
from agent import create_orchestrator_agent, generate_suggested_prompts, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting
from tools import UI_ACCESSIBLE_WORKSPACE
from config import HF_USER_MEMORIES_DATASET_ID

from PyPDF2 import PdfReader # For PDF processing in app.py
import pyreadstat # For SPSS processing

# Hugging Face related
from huggingface_hub import HfFileSystem
fs = HfFileSystem()

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MAX_CHAT_HISTORY_MESSAGES = 15
AGENT_INSTANCE_KEY = "esi_orchestrator_agent_instance"

def simple_once_cache(func):
    _cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in _cache: _cache[key] = func(*args, **kwargs)
        return _cache[key]
    return wrapper

@simple_once_cache
def setup_global_llm_settings_cached() -> tuple[bool, str | None]:
    print("Initializing LLM settings...")
    try:
        initialize_agent_settings(); print("LLM settings initialized.")
        return True, None
    except Exception as e:
        return False, f"Fatal Error: Could not initialize LLM settings. {e}"

@functools.lru_cache(maxsize=1)
def setup_agent_cached(max_search_results: int) -> tuple[Any | None, str | None]:
    print(f"Initializing AI agent with max_search_results={max_search_results}...")
    try:
        def read_uploaded_document_tool_fn_runtime(filename: str) -> str:
            uploaded_docs = app.storage.user.get("uploaded_documents", {})
            if filename not in uploaded_docs: return f"Error: Doc '{filename}' not found. Available: {list(uploaded_docs.keys())}"
            return uploaded_docs[filename]

        def analyze_dataframe_tool_fn_runtime(filename: str, head_rows: int = 5) -> str:
            uploaded_dfs = app.storage.user.get("uploaded_dataframes", {})
            if filename not in uploaded_dfs: return f"Error: DataFrame '{filename}' not found. Available: {list(uploaded_dfs.keys())}"
            df = uploaded_dfs[filename] # This should be a pandas DataFrame
            if not isinstance(df, pd.DataFrame): return f"Error: '{filename}' is not a DataFrame."
            info_str = f"DF: {filename}\nShape: {df.shape}\nCols: {', '.join(df.columns)}\nTypes:\n{df.dtypes.to_string()}\n"
            h_rows = max(0, min(head_rows, len(df)))
            if h_rows > 0: info_str += f"Head {h_rows}:\n{df.head(h_rows).to_string()}\n"
            info_str += f"Stats:\n{df.describe().to_string()}\n"
            return info_str

        doc_tool = FunctionTool.from_defaults(fn=read_uploaded_document_tool_fn_runtime, name="read_uploaded_document", description="Reads text of an uploaded document.")
        df_tool = FunctionTool.from_defaults(fn=analyze_dataframe_tool_fn_runtime, name="analyze_uploaded_dataframe", description="Analyzes an uploaded CSV/Excel/SPSS dataset.")

        agent_instance = create_orchestrator_agent(dynamic_tools=[doc_tool, df_tool], max_search_results=max_search_results)
        print("AI agent initialized successfully.")
        return agent_instance, None
    except Exception as e: return None, f"Failed to initialize AI agent: {e}"

def _get_user_id_from_cookie() -> Optional[str]: return app.storage.client.get("user_id")
def _set_user_id_cookie(user_id: str): app.storage.client["user_id"] = user_id
def _delete_user_id_cookie():
    if "user_id" in app.storage.client: del app.storage.client["user_id"]

def _get_ltm_pref_from_cookie() -> bool:
    pref = app.storage.client.get("long_term_memory_pref")
    return True if pref is None else (str(pref).lower() == 'true' or str(pref) == '1')

def _set_ltm_pref_cookie(enabled: bool): app.storage.client["long_term_memory_pref"] = str(enabled)

def _load_user_data_from_hf(user_id: str) -> Dict[str, Any]:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token: return {"metadata": {}, "messages": {}}
    meta, msgs = {}, {}
    try:
        meta_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/user_memories/{user_id}_metadata.json"
        msgs_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/user_memories/{user_id}_messages.json"
        try: meta = json.loads(fs.read_text(meta_path, token=hf_token))
        except: print(f"No metadata for {user_id}")
        try: msgs = json.loads(fs.read_text(msgs_path, token=hf_token))
        except: print(f"No messages for {user_id}")
    except Exception as e: print(f"HF load error for {user_id}: {e}")
    return {"metadata": meta, "messages": msgs}

def _save_to_hf(user_id: str, data_type: str, data: Dict):
    if not app.storage.user.get("long_term_memory_enabled", False): return
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token: return
    try:
        fname = f"user_memories/{user_id}_{data_type}.json"
        fpath = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/{fname}"
        with fs.open(fpath, "w", token=hf_token) as f: json.dump(data, f, indent=2)
        print(f"{data_type} for {user_id} saved to HF.")
    except Exception as e: print(f"Error saving {data_type} to HF for {user_id}: {e}")

def save_chat_history_to_hf(user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    all_msgs = app.storage.user.get("all_chat_messages", {})
    all_msgs[chat_id] = messages
    _save_to_hf(user_id, "messages", all_msgs)

def save_chat_metadata_to_hf(user_id: str, chat_metadata: Dict[str, str]):
    _save_to_hf(user_id, "metadata", chat_metadata)

def format_llama_chat_history(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    return [ChatMessage(role=MessageRole(msg["role"]), content=msg["content"]) for msg in messages[-MAX_CHAT_HISTORY_MESSAGES:]]

async def get_agent_response_stream(query: str, chat_history: List[ChatMessage]) -> AsyncGenerator[str, None]:
    agent = app.storage.general.get(AGENT_INSTANCE_KEY)
    if not agent: yield "Err: Agent missing."; return
    try:
        temp = app.storage.user.get("llm_temperature", 0.7)
        verb = app.storage.user.get("llm_verbosity", 3)
        if Settings.llm and hasattr(Settings.llm, 'temperature'): Settings.llm.temperature = temp
        
        mod_query = f"Verbosity Level: {verb}. {query}"
        response = await app.loop.run_in_executor(None, agent.chat, mod_query, chat_history)
        resp_text = response.response if hasattr(response, 'response') else str(response)
        for word in resp_text.split(" "):
            yield word + " "; await app.loop.run_in_executor(None, time.sleep, 0.02)
    except Exception as e: yield f"Agent error: {e}"


async def handle_user_input_from_ui(user_input: str):
    s = app.storage.user
    if not user_input or not (chat_area_ref := s.get('chat_area_ref')): return

    s["messages"].append({"role": "user", "content": user_input})
    await ui_module.display_chat_messages(chat_area_ref)

    history = format_llama_chat_history(s["messages"][:-1])
    s["messages"].append({"role": "assistant", "content": ""})
    msg_idx = len(s["messages"]) - 1
    full_response = ""
    async for chunk in get_agent_response_stream(user_input, chat_history=history):
        full_response += chunk
        s["messages"][msg_idx]["content"] = full_response
        await ui_module.display_chat_messages(chat_area_ref) # Refresh with chunk

    if s.get("long_term_memory_enabled") and s.get("user_id") and s.get("current_chat_id"):
        save_chat_history_to_hf(s["user_id"], s["current_chat_id"], s["messages"])
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(s["messages"]))
    if (sugg_ref := s.get('suggested_prompts_container_ref_builder')): sugg_ref.refresh()


async def new_chat_from_ui():
    await initialize_active_chat_session(new_chat=True)
    if (mcr := app.storage.user.get('main_content_ref')): mcr.refresh()

async def delete_chat_from_ui(chat_id: str):
    s = app.storage.user
    if not s.get("long_term_memory_enabled"): return
    if (uid := s.get("user_id")):
        if chat_id in s.get("all_chat_messages", {}): del s["all_chat_messages"][chat_id]
        if chat_id in s.get("chat_metadata", {}): del s["chat_metadata"][chat_id]
        save_chat_metadata_to_hf(uid, s["chat_metadata"]) # This saves the modified dict
        # Consider full message file rewrite for true deletion if necessary
        ui_module.ui.notify(f"Chat '{chat_id}' deleted.", type='info')
        if s.get("current_chat_id") == chat_id:
            s["current_chat_id"] = None
            await initialize_active_chat_session()
        if (mcr := s.get('main_content_ref')): mcr.refresh()

async def rename_chat_from_ui(chat_id: str, new_name: str):
    s = app.storage.user
    if not s.get("long_term_memory_enabled") or not (uid := s.get("user_id")) or not new_name: return
    if chat_id in s.get("chat_metadata", {}):
        s["chat_metadata"][chat_id] = new_name
        save_chat_metadata_to_hf(uid, s["chat_metadata"])
        ui_module.ui.notify(f"Chat renamed to '{new_name}'.", type='positive')
        if (mcr := s.get('main_content_ref')): mcr.refresh()

async def switch_chat_from_ui(chat_id: str):
    s = app.storage.user
    if not s.get("long_term_memory_enabled"): return
    if chat_id in s.get("chat_metadata", {}):
        s["current_chat_id"] = chat_id
        s["messages"] = s.get("all_chat_messages", {}).get(chat_id, [])
        s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(s["messages"]))
        if (mcr := s.get('main_content_ref')): mcr.refresh()

def get_discussion_markdown_from_ui(chat_id: str) -> str:
    msgs = app.storage.user.get("all_chat_messages", {}).get(chat_id, [])
    return "\n".join([f"**{m['role'].capitalize()}:**\n{m['content']}\n\n---" for m in msgs])

def get_discussion_docx_from_ui(chat_id: str) -> bytes:
    msgs = app.storage.user.get("all_chat_messages", {}).get(chat_id, [])
    meta = app.storage.user.get("chat_metadata", {})
    doc = Document()
    doc.add_heading(f"Chat: {meta.get(chat_id, 'Untitled')}", level=1)
    for m in msgs:
        doc.add_heading(f"{m['role'].capitalize()}:", level=3); doc.add_paragraph(m['content']); doc.add_paragraph("---")
    bio = BytesIO(); doc.save(bio); return bio.getvalue()

async def forget_me_from_ui():
    s = app.storage.user; uid_del = s.get("user_id"); hf_token = os.getenv("HF_TOKEN")
    if uid_del and hf_token:
        try:
            fs.rm(f"datasets/{HF_USER_MEMORIES_DATASET_ID}/user_memories/{uid_del}_metadata.json", token=hf_token, missing_ok=True)
            fs.rm(f"datasets/{HF_USER_MEMORIES_DATASET_ID}/user_memories/{uid_del}_messages.json", token=hf_token, missing_ok=True)
        except Exception as e: print(f"HF forget me error: {e}")
    _delete_user_id_cookie(); s.clear()
    ui_module.ui.notify("All data deleted. Refreshing...", type='warning')
    await ui_module.ui.run_javascript("setTimeout(() => window.location.reload(), 1500);")


async def set_long_term_memory_from_ui(enabled: bool):
    s = app.storage.user
    s["long_term_memory_enabled"] = enabled
    _set_ltm_pref_cookie(enabled)
    ui_module.ui.notify(f"LTM {'en' if enabled else 'dis'}abled.", type='info')
    await initialize_user_session_storage(force_reload=True)
    if (mcr := s.get('main_content_ref')): mcr.refresh()

async def regenerate_last_response_from_ui():
    s = app.storage.user
    if not (chat_ref := s.get('chat_area_ref')): return
    if not s["messages"] or s["messages"][-1]['role'] != 'assistant': return
    s["messages"].pop()
    if not s["messages"] or s["messages"][-1]['role'] != 'user':
        s["messages"].append({"role": "assistant", "content": "Could not regenerate."})
        await ui_module.display_chat_messages(chat_ref); return

    last_user_q = s["messages"][-1]['content']
    hist_regen = format_llama_chat_history(s["messages"][:-1])
    s["messages"].append({"role": "assistant", "content": ""})
    msg_idx = len(s["messages"]) - 1
    full_resp = ""
    async for chunk in get_agent_response_stream(last_user_q, chat_history=hist_regen):
        full_resp += chunk; s["messages"][msg_idx]["content"] = full_resp
        await ui_module.display_chat_messages(chat_ref)
    if s.get("long_term_memory_enabled") and s.get("user_id") and s.get("current_chat_id"):
        save_chat_history_to_hf(s["user_id"], s["current_chat_id"], s["messages"])
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(s["messages"]))
    if (sugg_ref := s.get('suggested_prompts_container_ref_builder')): sugg_ref.refresh()

async def handle_file_upload_app_impl(file_name: str, file_content: bytes, content_type: Optional[str]):
    s_user = app.storage.user
    file_ext = os.path.splitext(file_name)[1].lower()
    file_path_in_ws = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
    try:
        with open(file_path_in_ws, "wb") as f: f.write(file_content)
        ui_module.ui.notify(f"File '{file_name}' saved.", type='positive')
    except Exception as e: ui_module.ui.notify(f"Error saving '{file_name}': {e}", type='negative'); return

    ptype, pname = None, None
    if file_ext in [".pdf", ".docx", ".md", ".txt"]:
        txt = ""
        try:
            if file_ext == ".pdf": reader = PdfReader(BytesIO(file_content)); txt = "".join([p.extract_text() or "" for p in reader.pages])
            elif file_ext == ".docx": doc = Document(BytesIO(file_content)); txt = "\n".join([p.text for p in doc.paragraphs])
            else: txt = file_content.decode("utf-8", errors="replace")
            s_user.setdefault("uploaded_documents", {})[file_name] = txt
            ptype, pname = "document", file_name
        except Exception as e: ui_module.ui.notify(f"Error processing doc '{file_name}': {e}", type='negative')
    elif file_ext in [".csv", ".xlsx", ".sav"]:
        df = None
        try:
            if file_ext == ".csv": df = pd.read_csv(BytesIO(file_content))
            elif file_ext == ".xlsx": df = pd.read_excel(BytesIO(file_content))
            elif file_ext == ".sav": df = pyreadstat.read_sav(file_path_in_ws)[0] # pyreadstat returns (df, meta)
            if df is not None:
                s_user.setdefault("uploaded_dataframes", {})[file_name] = df
                ptype, pname = "dataframe", file_name
        except Exception as e: ui_module.ui.notify(f"Error processing data '{file_name}': {e}", type='negative')
    else: ui_module.ui.notify(f"Unsupported: {file_ext}", type='warning')

    if pname:
        msg = f"Received {ptype}: `{pname}`. You can ask to `read_uploaded_{ptype}('{pname}')`."
        s_user["messages"].append({"role": "assistant", "content": msg})
    
    if (ca_ref := s_user.get('chat_area_ref')): await ui_module.display_chat_messages(ca_ref)
    if (uf_ref := s_user.get('uploaded_files_list_refresher_ui')): uf_ref.refresh()

async def remove_file_app_impl(file_type: str, file_name: str):
    s = app.storage.user
    removed = False
    if file_type == "document" and file_name in s.get("uploaded_documents", {}):
        del s["uploaded_documents"][file_name]; removed = True
    elif file_type == "dataframe" and file_name in s.get("uploaded_dataframes", {}):
        del s["uploaded_dataframes"][file_name]; removed = True
    if removed: ui_module.ui.notify(f"'{file_name}' removed from session.", type='info')
    
    f_path_ws = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
    if os.path.exists(f_path_ws):
        try: os.remove(f_path_ws); ui_module.ui.notify(f"Disk file '{file_name}' deleted.", type='positive')
        except Exception as e: ui_module.ui.notify(f"Error deleting disk file: {e}", type='negative')
    if (uf_ref := s.get('uploaded_files_list_refresher_ui')): uf_ref.refresh()


@functools.lru_cache(maxsize=10)
def generate_suggested_prompts_cached(chat_history_tuple: tuple) -> List[str]:
    return DEFAULT_PROMPTS if not chat_history_tuple else generate_suggested_prompts(list(chat_history_tuple))

@simple_once_cache
def get_initial_greeting_text_cached() -> str: return generate_llm_greeting()


async def initialize_user_session_storage(force_reload: bool = False):
    s = app.storage.user
    if not force_reload and s.get("session_initialized"): return
    s["long_term_memory_enabled"] = _get_ltm_pref_from_cookie()
    uid = _get_user_id_from_cookie()
    if s["long_term_memory_enabled"]:
        if not uid: uid = str(uuid.uuid4()); _set_user_id_cookie(uid)
        s["user_id"] = uid
        data = _load_user_data_from_hf(uid)
        s["chat_metadata"] = data.get("metadata", {})
        s["all_chat_messages"] = data.get("messages", {})
    else:
        s["user_id"] = str(uuid.uuid4()); _delete_user_id_cookie()
        s["chat_metadata"] = {}; s["all_chat_messages"] = {}
    
    s.setdefault("current_chat_id", None)
    s.setdefault("messages", [])
    s.setdefault("uploaded_documents", {})
    s.setdefault("uploaded_dataframes", {})
    s.setdefault("llm_temperature", 0.7)
    s.setdefault("llm_verbosity", 3)
    s.setdefault("search_results_count", 5)
    s.setdefault("suggested_prompts", DEFAULT_PROMPTS)
    await initialize_active_chat_session()
    s["session_initialized"] = True

async def initialize_active_chat_session(new_chat: bool = False):
    s = app.storage.user
    if new_chat:
        new_id = str(uuid.uuid4())
        prefix = "Idea" if s["long_term_memory_enabled"] else "Session"
        nums = [int(m.group(1)) for n in s.get("chat_metadata", {}).values() if (m := re.match(rf"{prefix} (\d+)", n))]
        new_name = f"{prefix} {max(nums) + 1 if nums else 1}"
        s["current_chat_id"] = new_id
        s.get("chat_metadata", {})[new_id] = new_name
        s["messages"] = [{"role": "assistant", "content": get_initial_greeting_text_cached()}]
        s.get("all_chat_messages", {})[new_id] = s["messages"]
        if s["long_term_memory_enabled"] and s.get("user_id"):
            save_chat_metadata_to_hf(s["user_id"], s["chat_metadata"])
            save_chat_history_to_hf(s["user_id"], new_id, s["messages"])
    elif s.get("current_chat_id") and s["current_chat_id"] in s.get("all_chat_messages", {}):
        s["messages"] = s["all_chat_messages"][s["current_chat_id"]]
    elif s.get("chat_metadata"):
        first_id = next(iter(s["chat_metadata"]), None)
        if first_id:
            s["current_chat_id"] = first_id
            s["messages"] = s.get("all_chat_messages", {}).get(first_id, [])
        else: await initialize_active_chat_session(new_chat=True); return
    else: await initialize_active_chat_session(new_chat=True); return
    s["suggested_prompts"] = generate_suggested_prompts_cached(tuple(s["messages"]))


@ui.page('/')
async def main_page(client: Client):
    llm_ok, llm_error = setup_global_llm_settings_cached()
    if not llm_ok: ui_module.ui.notify(f"LLM Error: {llm_error}", type='negative'); return

    await initialize_user_session_storage()
    s_user = app.storage.user

    if AGENT_INSTANCE_KEY not in app.storage.general:
        agent_instance, agent_error = setup_agent_cached(max_search_results=s_user.get("search_results_count", 5))
        if not agent_instance: ui_module.ui.notify(f"Agent Error: {agent_error}", type='negative'); return
        app.storage.general[AGENT_INSTANCE_KEY] = agent_instance
    
    @ui.refreshable
    async def build_main_ui_content():
        # Get fresh copy of user storage for this render pass
        current_s_user = app.storage.user
        
        # Call the UI builder from ui.py
        # It will use app_callbacks to interact with app.py's logic
        # It will read display state (messages, metadata) mostly from app.storage.user
        ui_module.create_nicegui_interface(
            ext_handle_user_input_callback=handle_user_input_from_ui,
            ext_reset_callback=new_chat_from_ui, # Reset is same as new chat for now
            ext_new_chat_callback=new_chat_from_ui,
            ext_delete_chat_callback=delete_chat_from_ui,
            ext_rename_chat_callback=rename_chat_from_ui,
            ext_switch_chat_callback=switch_chat_from_ui,
            ext_get_discussion_markdown_callback=get_discussion_markdown_from_ui,
            ext_get_discussion_docx_callback=get_discussion_docx_from_ui,
            ext_forget_me_callback=forget_me_from_ui,
            ext_set_long_term_memory_callback=set_long_term_memory_from_ui,
            ext_regenerate_callback=regenerate_last_response_from_ui,
            ext_handle_file_upload_app_callback=handle_file_upload_app_impl,
            ext_remove_file_app_callback=remove_file_app_impl,
            # Pass state values that ui.py might need for initial setup or direct binding
            current_chat_id_val=current_s_user.get("current_chat_id"),
            chat_metadata_val=current_s_user.get("chat_metadata", {}),
            llm_temp_val=current_s_user.get("llm_temperature", 0.7),
            llm_verb_val=current_s_user.get("llm_verbosity", 3),
            search_count_val=current_s_user.get("search_results_count", 5),
            ltm_enabled_val=current_s_user.get("long_term_memory_enabled", False),
            suggested_prompts_list_val=current_s_user.get("suggested_prompts", [])
            # messages_val, uploaded_docs_val etc. are not passed if ui.py reads them directly from app.storage.user
        )
        
        # After UI structure is created by ui_module, populate dynamic parts
        chat_area_ref = current_s_user.get('chat_area_ref')
        if chat_area_ref: await ui_module.display_chat_messages(chat_area_ref)

        # Build and store the refreshable UI function for the uploaded files list
        uploaded_files_container = current_s_user.get('ui_uploaded_files_container_ref')
        if uploaded_files_container:
            @ui.refreshable
            def _build_and_refresh_uploaded_files_list_app_scoped():
                # This function is defined within main_page's scope, so it has access to ui_module
                # It will be called to refresh the list of uploaded files.
                s_files = app.storage.user # get latest state for this refresh
                docs = s_files.get("uploaded_documents", {})
                dfs = s_files.get("uploaded_dataframes", {})

                uploaded_files_container.clear()
                with uploaded_files_container:
                    if docs:
                        ui_module.ui.label("Documents:").classes("text-caption")
                        for name_ in docs.keys():
                            with ui_module.ui.row().classes('w-full items-center no-wrap'):
                                ui_module.ui.icon('description').classes('q-mr-xs')
                                ui_module.ui.label(name_).classes('flex-grow ellipsis')
                                ui_module.ui.button(icon='delete', on_click=lambda n=name_: remove_file_app_impl("document", n)).props('flat dense round color=negative size=sm')
                    if dfs:
                        ui_module.ui.label("Datasets:").classes("text-caption q-mt-sm")
                        for name_ in dfs.keys():
                             with ui_module.ui.row().classes('w-full items-center no-wrap'):
                                ui_module.ui.icon('table_chart').classes('q-mr-xs')
                                ui_module.ui.label(name_).classes('flex-grow ellipsis')
                                ui_module.ui.button(icon='delete', on_click=lambda n=name_: remove_file_app_impl("dataframe", n)).props('flat dense round color=negative size=sm')
                    if not docs and not dfs:
                        ui_module.ui.label("No files uploaded.").classes('q-pa-sm text-italic')
            
            current_s_user['uploaded_files_list_refresher_ui'] = _build_and_refresh_uploaded_files_list_app_scoped
            _build_and_refresh_uploaded_files_list_app_scoped() # Initial population

        # Suggested prompts refreshable builder
        suggested_prompts_container = current_s_user.get('suggested_prompts_container_ref')
        if suggested_prompts_container:
            @ui.refreshable
            def _build_and_refresh_suggested_prompts_app_scoped():
                s_prompts = app.storage.user
                prompts = s_prompts.get("suggested_prompts", [])
                suggested_prompts_container.clear()
                with suggested_prompts_container:
                    if prompts:
                        for p_text in prompts:
                            ui_module.ui.button(p_text, on_click=lambda p=p_text: handle_user_input_from_ui(p)).props('outline rounded dense color=primary').classes('q-ma-xs')

            current_s_user['suggested_prompts_container_ref_builder'] = _build_and_refresh_suggested_prompts_app_scoped
            _build_and_refresh_suggested_prompts_app_scoped()


    s_user['main_content_ref'] = build_main_ui_content # Store ref to the refreshable UI builder
    await build_main_ui_content() # Initial build


if __name__ in {"__main__", "__mp_main__"}:
    app.add_static_files('/workspace', UI_ACCESSIBLE_WORKSPACE)
    if not os.getenv("GOOGLE_API_KEY"): print("⚠️ GOOGLE_API_KEY not set.")
    ui.run(
        title="ESI - NiceGUI",
        host="0.0.0.0", port=8080, reload=True,
        storage_secret="a_very_secret_key_for_storage_12345" # Change for production
    )
