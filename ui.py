from nicegui import ui, app, Client # Added Client here
from typing import List, Dict, Any, Optional, Callable, Coroutine # Added Coroutine
import os
import re
import json
import html
import pandas as pd # Keep for type hints if any, though app.py handles actual pd logic
from PyPDF2 import PdfReader # Keep for type hints if any
from docx import Document # Keep for type hints if any
import io # Keep for type hints if any
import mimetypes # Keep for standalone test
from dotenv import load_dotenv # Keep for standalone test

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UI_ACCESSIBLE_WORKSPACE = os.path.join(PROJECT_ROOT, "workspace_ui_accessible")
os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)
STORAGE_SECRET = os.environ.get("STORAGE_SECRET",)
CODE_DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER = "---RAG_SOURCE---"

class ChatState:
    def __init__(self):
        # This chat_state is primarily for UI-bound elements like the input field value.
        # Most other state (messages, uploaded files, etc.) will be driven by app.storage.user from app.py.
        self.chat_input_value: Optional[str] = "" # Bound to the input field
        # Other ui-specific temporary states can be added if needed, but avoid duplicating app.storage.user

chat_state = ChatState() # Local instance for UI bindings

class AppCallbacks: # Defines the structure of callbacks app.py will provide
    def __init__(self):
        self.handle_user_input: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None
        self.reset_chat: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
        self.new_chat: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
        self.delete_chat: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None
        self.rename_chat: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None
        self.switch_chat: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None
        self.get_discussion_markdown: Optional[Callable[[str], str]] = None
        self.get_discussion_docx: Optional[Callable[[str], bytes]] = None
        self.forget_me: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
        self.set_long_term_memory: Optional[Callable[[bool], Coroutine[Any, Any, None]]] = None
        self.regenerate_last_response: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
        self.handle_file_upload_app: Optional[Callable[[str, bytes, Optional[str]], Coroutine[Any, Any, None]]] = None
        self.remove_file_app: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None

app_callbacks = AppCallbacks() # Global instance to hold registered callbacks

async def display_chat_messages(chat_container: ui.column):
    s = app.storage.user # Source of truth for messages
    messages_to_display = s.get("messages", [])

    chat_container.clear()
    with chat_container:
        for msg_idx, message in enumerate(messages_to_display):
            is_user = message["role"] == "user"
            content = message["content"]
            text_for_message = content
            rag_sources_data = []
            code_download_filename = None
            code_is_image = False

            if message["role"] == "assistant":
                rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER)}({{\"type\":.*?}})", re.DOTALL)
                processed_text_after_rag = text_for_message
                for match in reversed(list(rag_source_pattern.finditer(text_for_message))):
                    json_str = match.group(1)
                    try: rag_sources_data.append(json.loads(json_str))
                    except json.JSONDecodeError as e: print(f"UI Warning: Could not decode RAG JSON: {e}")
                    processed_text_after_rag = processed_text_after_rag[:match.start()] + processed_text_after_rag[match.end():]
                text_for_message = processed_text_after_rag.strip()

                code_marker_match = re.search(rf"^{re.escape(CODE_DOWNLOAD_MARKER)}(.*)$", text_for_message, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    text_for_message = text_for_message[:code_marker_match.start()].strip() + text_for_message[code_marker_match.end():].strip()
                    code_download_filename = extracted_filename
                    if os.path.exists(os.path.join(UI_ACCESSIBLE_WORKSPACE, extracted_filename)):
                        if os.path.splitext(code_download_filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                            code_is_image = True
                    else:
                        text_for_message += f"\n\n*(UI Warning: File '{extracted_filename}' for download not in workspace.)*"

            with ui.chat_message(name=message["role"].title(), sent=is_user, avatar=f"https://robohash.org/{message['role']}?set=set4&size=50x50"):
                if text_for_message: ui.markdown(text_for_message)

                rag_sources_data.sort(key=lambda x: x.get('citation_number', float('inf')) if x.get('type') == 'pdf' else x.get('url', x.get('title', '')))
                displayed_rag_ids = set()
                for rag_data in rag_sources_data:
                    stype, ident, ditem = rag_data.get("type"), None, False
                    if stype == "pdf":
                        name, path = rag_data.get("name","s.pdf"), rag_data.get("path")
                        cite = f"[{rag_data.get('citation_number')}] " if rag_data.get('citation_number') else ""
                        ident = path
                        if ident and ident not in displayed_rag_ids:
                            if path and path.startswith("http"):
                                ui.html(f"Src: {cite}<a href='{path}' target='_blank'>{html.escape(name)}</a>"); ditem=True
                            elif path and path.startswith("file://"):
                                f_ws = os.path.join(UI_ACCESSIBLE_WORKSPACE, os.path.basename(path[len("file://"):]))
                                if os.path.exists(f_ws):
                                    dl_url = f"/workspace/{os.path.basename(f_ws)}" # Assumes app.add_static_files in app.py
                                    ui.html(f"Src: {cite}<a href='{dl_url}' target='_blank' download='{html.escape(name)}'>DL PDF: {html.escape(name)}</a>"); ditem=True
                                else: ui.label(f"PDF '{name}' missing.").classes('text-warning')
                    elif stype == "web":
                        url, title = rag_data.get("url"), rag_data.get("title", "Web Src")
                        ident = url
                        if ident and ident not in displayed_rag_ids and url:
                            ui.html(f"Src: <a href='{url}' target='_blank'>{html.escape(title)}</a>"); ditem=True
                    if ditem and ident: displayed_rag_ids.add(ident); ui.separator()

                if code_download_filename:
                    f_path_ws = os.path.join(UI_ACCESSIBLE_WORKSPACE, code_download_filename)
                    if os.path.exists(f_path_ws):
                        static_url = f"/workspace/{code_download_filename}"
                        if code_is_image: ui.image(static_url).props('width=300px')
                        else: ui.html(f"<a href='{static_url}' target='_blank' download='{html.escape(code_download_filename)}'>DL: {html.escape(code_download_filename)}</a>")

            last_asst = (message["role"] == "assistant" and msg_idx == len(messages_to_display) - 1)
            can_regen = last_asst and (len(messages_to_display) == 1 or (len(messages_to_display) > 1 and messages_to_display[msg_idx-1]["role"] == "user"))
            with ui.row().classes('ml-8'):
                ui.button(icon='content_copy', on_click=lambda c=text_for_message, i=msg_idx: copy_to_clipboard(c,i)).props('flat dense round')
                if can_regen and app_callbacks.regenerate_last_response:
                    ui.button(icon='refresh', on_click=app_callbacks.regenerate_last_response).props('flat dense round')

async def copy_to_clipboard(text_to_copy: str, msg_idx: int):
    escaped_text = json.dumps(text_to_copy)
    await ui.run_javascript(f"navigator.clipboard.writeText({escaped_text})", timeout=5.0)
    ui.notify(f"Message {msg_idx + 1} content copied!", icon="📋")

async def handle_send_message():
    user_input = chat_state.chat_input_value
    if user_input and app_callbacks.handle_user_input:
        chat_state.chat_input_value = ""
        await app_callbacks.handle_user_input(user_input)

async def handle_suggested_prompt(prompt: str):
    if app_callbacks.handle_user_input:
        await app_callbacks.handle_user_input(prompt)

async def handle_ui_file_upload(e: Any): # NiceGUI's UploadEventArguments
    if app_callbacks.handle_file_upload_app:
        file_content = e.content.read() if hasattr(e.content, 'read') else e.content # Handle BytesIO or bytes
        await app_callbacks.handle_file_upload_app(e.name, file_content, e.type)
    else: ui.notify("File upload processing not configured.", type='negative')

# Note: handle_ui_remove_file is not needed here if app.py's list builder directly calls app_callbacks.remove_file_app

def create_nicegui_interface(
    ext_handle_user_input_callback: Callable[[str], Coroutine[Any, Any, None]],
    ext_reset_callback: Callable[[], Coroutine[Any, Any, None]],
    ext_new_chat_callback: Callable[[], Coroutine[Any, Any, None]],
    ext_delete_chat_callback: Callable[[str], Coroutine[Any, Any, None]],
    ext_rename_chat_callback: Callable[[str, str], Coroutine[Any, Any, None]],
    ext_switch_chat_callback: Callable[[str], Coroutine[Any, Any, None]],
    ext_get_discussion_markdown_callback: Callable[[str], str],
    ext_get_discussion_docx_callback: Callable[[str], bytes],
    ext_forget_me_callback: Callable[[], Coroutine[Any, Any, None]],
    ext_set_long_term_memory_callback: Callable[[bool], Coroutine[Any, Any, None]],
    ext_regenerate_callback: Callable[[], Coroutine[Any, Any, None]],
    ext_handle_file_upload_app_callback: Callable[[str, bytes, Optional[str]], Coroutine[Any, Any, None]],
    ext_remove_file_app_callback: Callable[[str, str], Coroutine[Any, Any, None]],
    # State values from app.py for initial setup / read-only display parts
    current_chat_id_val: Optional[str], # Read from app.storage.user by UI logic
    chat_metadata_val: Dict[str, str],   # Read from app.storage.user by UI logic
    # messages_val is not needed here as display_chat_messages reads from app.storage.user
    # uploaded_docs_val & uploaded_dfs_val not needed here if app.py builds the list
    llm_temp_val: float,
    llm_verb_val: int,
    search_count_val: int,
    ltm_enabled_val: bool,
    suggested_prompts_list_val: Optional[List[str]]
):
    global app_callbacks
    app_callbacks.handle_user_input = ext_handle_user_input_callback
    app_callbacks.reset_chat = ext_reset_callback
    app_callbacks.new_chat = ext_new_chat_callback
    app_callbacks.delete_chat = ext_delete_chat_callback
    app_callbacks.rename_chat = ext_rename_chat_callback
    app_callbacks.switch_chat = ext_switch_chat_callback
    app_callbacks.get_discussion_markdown = ext_get_discussion_markdown_callback
    app_callbacks.get_discussion_docx = ext_get_discussion_docx_callback
    app_callbacks.forget_me = ext_forget_me_callback
    app_callbacks.set_long_term_memory = ext_set_long_term_memory_callback
    app_callbacks.regenerate_last_response = ext_regenerate_callback
    app_callbacks.handle_file_upload_app = ext_handle_file_upload_app_callback
    app_callbacks.remove_file_app = ext_remove_file_app_callback

    # Bind local chat_state for UI input to app.storage.user for central management
    # This assumes app.py will initialize these in app.storage.user for binding
    chat_state.llm_temperature = llm_temp_val # Initial value
    chat_state.llm_verbosity = llm_verb_val
    chat_state.search_results_count = search_count_val
    # ltm_enabled_val is used for conditional UI, actual state in app.storage.user

    with ui.header(elevated=True).classes('bg-primary text-white'):
        ui.label("🎓 ESI: ESI Scholarly Instructor").classes('text-h5')

    with ui.left_drawer(value=True, bordered=True).props('width=300'):
        ui.label("Chat Controls & Settings").classes("text-h6 q-pa-md")
        with ui.expansion("Chat History", icon="forum").props('group="sidebar" expanded'):
            s_user = app.storage.user # Access user-specific storage
            if not s_user.get("long_term_memory_enabled", False):
                ui.label("LTM disabled. Chats are temporary.").classes("q-pa-sm text-warning")
                ui.button("➕ New Chat (Temp)", on_click=app_callbacks.new_chat).props('flat color=primary icon=add').classes('w-full')
            else:
                ui.button("➕ New Chat", on_click=app_callbacks.new_chat).props('flat color=primary icon=add').classes('w-full')

            current_meta = s_user.get("chat_metadata", {})
            curr_chat_id = s_user.get("current_chat_id")
            if current_meta:
                for chat_id, chat_name in sorted(current_meta.items(), key=lambda item: item[1].lower()):
                    with ui.row().classes('w-full items-center no-wrap'):
                        is_c = (chat_id == curr_chat_id)
                        btn_props = f'flat color="{"primary" if is_c else "grey-7"}" {"" if is_c else "text-color=dark"}'
                        ui.button(chat_name, on_click=lambda cid=chat_id: app_callbacks.switch_chat(cid) if not is_c else None).props(btn_props).classes('flex-grow text-left')
                        with ui.button(icon='more_vert').props('flat round dense'):
                            with ui.menu() as menu: # Keep menu reference for handlers
                                ui.menu_item('Rename', on_click=lambda cid=chat_id, cname=chat_name, m=menu: handle_rename_chat_dialog(cid, cname, m))
                                ui.menu_item('Download MD', on_click=lambda cid=chat_id, cname=chat_name, m=menu: handle_download_md(cid, cname, m))
                                ui.menu_item('Download DOCX', on_click=lambda cid=chat_id, cname=chat_name, m=menu: handle_download_docx(cid, cname, m))
                                ui.menu_item('Delete', on_click=lambda cid=chat_id, m=menu: handle_delete_chat_confirm(cid, m))
            else: ui.label("No saved chats.").classes('q-pa-sm text-italic')

        with ui.expansion("Upload Files", icon="upload_file").props('group="sidebar"'):
            ui.upload(label="Upload document or dataset", auto_upload=True, on_upload=handle_ui_file_upload, max_file_size=50*1024*1024).props('accept=".pdf,.docx,.md,.txt,.csv,.xlsx,.sav"')
            # The container for the list of uploaded files
            # app.py will create a refreshable function that populates this.
            uploaded_files_container = ui.column().classes('w-full q-pt-sm')
            app.storage.user['ui_uploaded_files_container_ref'] = uploaded_files_container # app.py will use this ref

        with ui.expansion("LLM Settings", icon="tune").props('group="sidebar"'):
            ui.slider(min=0.0, max=2.0, step=0.1).bind_value(app.storage.user, 'llm_temperature').props('label="Creativity (Temp.)"')
            ui.slider(min=1, max=5, step=1).bind_value(app.storage.user, 'llm_verbosity').props('label="Verbosity"')
            ui.slider(min=3, max=15, step=1).bind_value(app.storage.user, 'search_results_count').props('label="Search Results"')
            ui.switch("Enable Long-term Memory").bind_value(app.storage.user, 'long_term_memory_enabled').on_change(lambda e: app_callbacks.set_long_term_memory(e.value) if app_callbacks.set_long_term_memory else None)

            # Conditional "Forget Me" button display
            with ui.column().bind_visibility_from(app.storage.user, 'long_term_memory_enabled'):
                 with ui.button("Forget Me (Delete All Data)").props('color=negative flat').classes('w-full q-mt-md'):
                    with ui.menu() as menu_f:
                        ui.label("This will delete ALL saved data. Are you sure?").classes('q-pa-md bg-warning text-white')
                        with ui.row():
                            ui.button("Yes, Delete All", on_click=lambda: (app_callbacks.forget_me(), menu_f.close()) if app_callbacks.forget_me else menu_f.close(), color='negative')
                            ui.button("Cancel", on_click=menu_f.close)
            with ui.column().bind_visibility_from(app.storage.user, 'long_term_memory_enabled', backward=lambda x: not x):
                 ui.label("Long-term memory disabled.").classes('q-pa-sm text-italic')


        with ui.expansion("About ESI", icon="info").props('group="sidebar"'):
            ui.markdown("ESI uses AI to help you navigate the dissertation process...").classes('q-pa-md')

    with ui.column().classes('w-full h-full no-wrap items-stretch'):
        chat_area = ui.column().classes('w-full flex-grow overflow-y-auto q-pa-md').style('height: calc(100vh - 150px);') # Adjusted height
        app.storage.user['chat_area_ref'] = chat_area

        suggested_prompts_container = ui.row().classes('w-full q-pa-sm justify-center')
        # Suggested prompts will be populated by app.py using a refreshable component or direct creation
        app.storage.user['suggested_prompts_container_ref'] = suggested_prompts_container
        if suggested_prompts_list_val: # Initial prompts from app.py state
             with suggested_prompts_container:
                for p_text in suggested_prompts_list_val:
                    ui.button(p_text, on_click=lambda p=p_text: handle_suggested_prompt(p)).props('outline rounded dense color=primary').classes('q-ma-xs')

        with ui.row().classes('w-full items-center q-pa-md bg-grey-2'):
            ui.input(placeholder="Ask me about dissertations...").props('rounded outlined dense input-class="ml-3"').classes('flex-grow').bind_value(chat_state, 'chat_input_value').on('keydown.enter', handle_send_message)
            ui.button(icon='send', on_click=handle_send_message).props('round dense color=primary')

    async def handle_rename_chat_dialog(chat_id: str, current_name: str, parent_menu: ui.menu):
        parent_menu.close()
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Rename chat: '{current_name}'").classes('text-h6')
            new_name_input = ui.input("New name", value=current_name)
            with ui.row():
                ui.button("Rename", on_click=lambda: (app_callbacks.rename_chat(chat_id, new_name_input.value) if app_callbacks.rename_chat and new_name_input.value else None, dialog.close()))
                ui.button("Cancel", on_click=dialog.close)
        await dialog

    async def handle_delete_chat_confirm(chat_id: str, parent_menu: ui.menu):
        parent_menu.close()
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete this chat?").classes('text-h6')
            with ui.row():
                ui.button("Delete", color='negative', on_click=lambda: (app_callbacks.delete_chat(chat_id) if app_callbacks.delete_chat else None, dialog.close()))
                ui.button("Cancel", on_click=dialog.close)
        await dialog

    async def handle_download_md(chat_id: str, chat_name: str, parent_menu: ui.menu):
        parent_menu.close()
        if app_callbacks.get_discussion_markdown:
            content = app_callbacks.get_discussion_markdown(chat_id)
            ui.download(content.encode(), f"{chat_name.replace(' ', '_')}.md")
        else: ui.notify("MD download not set up.", type='negative')

    async def handle_download_docx(chat_id: str, chat_name: str, parent_menu: ui.menu):
        parent_menu.close()
        if app_callbacks.get_discussion_docx:
            content = app_callbacks.get_discussion_docx(chat_id)
            ui.download(content, f"{chat_name.replace(' ', '_')}.docx")
        else: ui.notify("DOCX download not set up.", type='negative')

    # Returns key containers for app.py to manage or refresh their content
    return chat_area, suggested_prompts_container, uploaded_files_container


if __name__ == "__main__":
    # This standalone test needs careful setup of app.storage.user
    # and mocking of all callbacks defined in AppCallbacks.

    @ui.page('/')
    async def index_page(client: Client):
        # Initialize app.storage.user for testing
        # This would normally be handled by app.py's on_connect or similar logic
        app.storage.user.clear()
        app.storage.user.update({
            "messages": [{"role": "assistant", "content": "ESI NiceGUI Standalone Test"}],
            "chat_metadata": {"test_id_1": "Test Chat Alpha", "test_id_2": "Beta Conversation"},
            "current_chat_id": "test_id_1",
            "uploaded_documents": {}, # Start empty, will be populated by mock_handle_upload_app
            "uploaded_dataframes": {},
            "llm_temperature": 0.6,
            "llm_verbosity": 2,
            "search_results_count": 4,
            "long_term_memory_enabled": False, # Test LTM disabled state
            "suggested_prompts": ["Hello ESI", "What can you do?"]
        })

        # --- Mock Callbacks for Standalone Test ---
        async def mock_handle_input_app(text):
            print(f"APP_MOCK: User input: {text}")
            app.storage.user["messages"].append({"role": "user", "content": text})
            # Simulate agent response
            await app.loop.run_in_executor(None, time.sleep, 0.5) # Simulate delay
            app.storage.user["messages"].append({"role": "assistant", "content": f"NiceGUI echo: {text}"})

            chat_area_ref = app.storage.user.get('chat_area_ref')
            if chat_area_ref: await display_chat_messages(chat_area_ref) # Refresh chat

        async def mock_handle_upload_app(name, content, content_type):
            print(f"APP_MOCK: File upload: {name}, type: {content_type}, size: {len(content)}")
            app.storage.user.setdefault("uploaded_documents", {})[name] = f"Mock content for {name}"
            app.storage.user["messages"].append({"role": "assistant", "content": f"File '{name}' received by mock app."})

            chat_area_ref = app.storage.user.get('chat_area_ref')
            if chat_area_ref: await display_chat_messages(chat_area_ref)

            files_refresher = app.storage.user.get('uploaded_files_list_refresher_ui')
            if files_refresher: files_refresher.refresh()

        async def mock_remove_file_app(ftype, fname):
            print(f"APP_MOCK: Remove file: {ftype} - {fname}")
            if ftype == "document" and fname in app.storage.user.get("uploaded_documents", {}):
                del app.storage.user["uploaded_documents"][fname]
            files_refresher = app.storage.user.get('uploaded_files_list_refresher_ui')
            if files_refresher: files_refresher.refresh()

        async def mock_set_ltm(val: bool):
            print(f"APP_MOCK: Set LTM to {val}")
            app.storage.user['long_term_memory_enabled'] = val
            # This would typically trigger a larger UI refresh or state reload in a full app
            # For this test, we might need to manually refresh relevant parts or the whole page
            # For now, let the create_nicegui_interface re-render based on this new state.
            # This requires the main content area to be refreshable.
            main_content_ref = app.storage.user.get('main_content_ref_for_test')
            if main_content_ref: main_content_ref.refresh()


        # Main UI building call
        # The create_nicegui_interface will set up app_callbacks with these mocks
        # It will also return containers that app.py (or this test setup) can manage.
        # We need a refreshable main content area for LTM toggle to work visually in test.
        @ui.refreshable
        async def build_test_content():
            # Retrieve the latest state from app.storage.user for each build/refresh
            s_user = app.storage.user
            create_nicegui_interface(
                ext_handle_user_input_callback=mock_handle_input_app,
                ext_handle_file_upload_app_callback=mock_handle_upload_app,
                ext_remove_file_app_callback=mock_remove_file_app,
                ext_set_long_term_memory_callback=mock_set_ltm,
                # Provide other mocks or simple lambdas
                ext_reset_callback=lambda: print("Mock Reset"),
                ext_new_chat_callback=lambda: print("Mock New Chat"),
                ext_delete_chat_callback=lambda cid: print(f"Mock Delete: {cid}"),
                ext_rename_chat_callback=lambda cid, name: print(f"Mock Rename: {cid} to {name}"),
                ext_switch_chat_callback=lambda cid: print(f"Mock Switch: {cid}"),
                ext_get_discussion_markdown_callback=lambda cid: f"## Markdown for {cid}\n- Point 1",
                ext_get_discussion_docx_callback=lambda cid: b"DOCX_CONTENT_FOR_" + cid.encode(),
                ext_forget_me_callback=lambda: print("Mock Forget Me"),
                ext_regenerate_callback=lambda: print("Mock Regenerate"),
                current_chat_id_val=s_user.get("current_chat_id"),
                chat_metadata_val=s_user.get("chat_metadata", {}),
                llm_temp_val=s_user.get("llm_temperature"),
                llm_verb_val=s_user.get("llm_verbosity"),
                search_count_val=s_user.get("search_results_count"),
                ltm_enabled_val=s_user.get("long_term_memory_enabled"),
                suggested_prompts_list_val=s_user.get("suggested_prompts")
            )
            # After UI is created, populate dynamic parts
            chat_area_r = app.storage.user.get('chat_area_ref')
            if chat_area_r: await display_chat_messages(chat_area_r)

            # Setup and call the uploaded files list builder
            uploaded_files_container = app.storage.user.get('ui_uploaded_files_container_ref')
            if uploaded_files_container:
                @ui.refreshable
                def _build_test_uploaded_files():
                    s_user_files = app.storage.user
                    docs = s_user_files.get("uploaded_documents", {})
                    uploaded_files_container.clear()
                    with uploaded_files_container:
                        if docs:
                            ui.label("Documents:").classes("text-caption")
                            for name_ in docs:
                                with ui.row().classes('w-full items-center'):
                                    ui.icon('description'); ui.label(name_).classes('flex-grow')
                                    ui.button(icon='delete', on_click=lambda n=name_: mock_remove_file_app("document", n)).props('flat dense round color=negative')
                        if not docs: ui.label("No files uploaded.").classes('q-pa-sm text-italic')
                app.storage.user['uploaded_files_list_refresher_ui'] = _build_test_uploaded_files
                _build_test_uploaded_files()

        app.storage.user['main_content_ref_for_test'] = build_test_content
        await build_test_content()


    app.add_static_files('/workspace', UI_ACCESSIBLE_WORKSPACE)
    mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')
    ui.run(title="ESI NiceGUI Test", storage_secret=STORAGE_SECRET, host="0.0.0.0", port=8080)
