import os
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from nicegui import ui, app, Client
from nicegui.events import UploadEventArguments
from nicegui.elements.markdown import Markdown
import asyncio

# Assuming PROJECT_ROOT is defined in config.py or similar, or define it here
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UI_ACCESSIBLE_WORKSPACE = os.path.join(PROJECT_ROOT, "workspace_ui_accessible")
os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)

# Fix: Provide a default string for STORAGE_SECRET to prevent it from being None
STORAGE_SECRET = os.environ.get("STORAGE_SECRET", "a_fallback_secret_for_dev")

# Configure NiceGUI storage for UI-specific test runs
app.storage.TEMP_BASE_DIR = ".nicegui"
app.storage.user.set_storage_secret(STORAGE_SECRET)

CODE_DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER = "---RAG_SOURCE---"

class ChatState:
    def __init__(self):
        self.current_chat_id: Optional[str] = None
        self.chat_history: Dict[str, List[Dict[str, Any]]] = {}
        self.chat_names: Dict[str, str] = {}
        self.uploaded_files: Dict[str, Dict[str, Any]] = {} # {file_type: {filename: content_bytes}}
        self.long_term_memory_enabled: bool = False
        self.last_agent_response: Optional[str] = None # Store the last full agent response
        self.last_user_query: Optional[str] = None # Store the last user query
        self.last_chat_history_before_regen: Optional[List[Dict[str, Any]]] = None # Store history before last query
        self.chat_input_value: str = "" # To store the current value of the chat input

    def set_chat_input_value(self, value: str):
        self.chat_input_value = value

# This chat_state instance will be passed from app.py
chat_state = ChatState()

class AppCallbacks:
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
        self.handle_file_upload: Optional[Callable[[str, bytes, Optional[str]], Coroutine[Any, Any, None]]] = None
        self.remove_file: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None
        self.get_initial_greeting_text_cached: Optional[Callable[[], str]] = None
        self.generate_suggested_prompts_cached: Optional[Callable[[Tuple], List[str]]] = None

app_callbacks = AppCallbacks()

async def display_chat_messages(chat_container: ui.column):
    """
    Displays chat messages in the UI, including streaming responses.
    """
    chat_container.clear()
    if not chat_state.current_chat_id:
        chat_container.add(ui.label("Start a new chat or select an existing one.").classes("text-grey-6"))
        return

    messages = chat_state.chat_history.get(chat_state.current_chat_id, [])
    
    for i, msg in enumerate(messages):
        is_user = msg["role"] == "user"
        with chat_container.row().classes('w-full no-wrap' + ('justify-end' if is_user else '')):
            with ui.card().classes('max-w-[70%] ' + ('bg-blue-100' if is_user else 'bg-gray-100')):
                ui.label(msg["content"]).classes('whitespace-pre-wrap')
                if not is_user:
                    # Add copy button for assistant messages
                    ui.icon('content_copy') \
                        .classes('absolute top-2 right-2 cursor-pointer text-gray-500 hover:text-gray-800') \
                        .on('click', lambda msg=msg, i=i: copy_to_clipboard(msg["content"], i))

    # If the last message is from the assistant and it's still streaming, handle it
    # This logic might need to be more sophisticated if streaming is handled directly in handle_user_input_from_ui
    # For now, assuming full messages are added to history before display_chat_messages is called.
    # If streaming, you'd typically have a placeholder message and update its content.

async def copy_to_clipboard(text_to_copy: str, msg_idx: int):
    """Copies text to clipboard and shows a notification."""
    await ui.run_javascript(f'navigator.clipboard.writeText(`{text_to_copy.replace("`", "\\`")}`)')
    ui.notify(f'Message {msg_idx+1} copied to clipboard!', type='info', timeout=1000)

async def handle_send_message():
    """
    Handles sending a message from the chat input.
    """
    user_input = chat_state.chat_input_value.strip()
    if not user_input:
        return

    chat_state.set_chat_input_value("") # Clear input immediately
    chat_input_ui.refresh() # Refresh the input field

    # Add user message to UI immediately
    if not chat_state.current_chat_id:
        # This should ideally be handled by app.py's initialize_active_chat_session
        # but as a fallback, ensure a chat exists.
        await app_callbacks.new_chat()
        chat_list_container_ui.refresh() # Refresh chat list to show new chat

    chat_state.chat_history[chat_state.current_chat_id].append({"role": "user", "content": user_input})
    await display_chat_messages(chat_area) # Update UI with user message

    # Add a placeholder for the assistant's response
    assistant_message_placeholder = {"role": "assistant", "content": "..."}
    chat_state.chat_history[chat_state.current_chat_id].append(assistant_message_placeholder)
    await display_chat_messages(chat_area) # Update UI with placeholder

    # Scroll to bottom
    await ui.run_javascript('var element = document.querySelector(".q-page-container > div > div"); element.scrollTop = element.scrollHeight;', timeout=0.5)

    full_response_content = ""
    try:
        # Call the app-level handler to get the streaming response
        async for chunk in app_callbacks.handle_user_input(user_input):
            full_response_content += chunk
            # Update the last assistant message in chat_state.chat_history
            if chat_state.chat_history[chat_state.current_chat_id][-1] is assistant_message_placeholder:
                assistant_message_placeholder["content"] = full_response_content
            else:
                # This case should ideally not happen if placeholder logic is correct
                # But as a fallback, append if somehow missing
                chat_state.chat_history[chat_state.current_chat_id].append({"role": "assistant", "content": full_response_content})
            await display_chat_messages(chat_area) # Refresh UI with new chunk
            await ui.run_javascript('var element = document.querySelector(".q-page-container > div > div"); element.scrollTop = element.scrollHeight;', timeout=0.5)
    except Exception as e:
        error_message = f"Error during response streaming: {e}"
        print(error_message)
        if chat_state.chat_history[chat_state.current_chat_id][-1] is assistant_message_placeholder:
            assistant_message_placeholder["content"] = f"Error: {error_message}"
        else:
            chat_state.chat_history[chat_state.current_chat_id].append({"role": "assistant", "content": f"Error: {error_message}"})
        await display_chat_messages(chat_area) # Show error in UI

    # After streaming, ensure the final message is stored and UI is updated
    # The app_callbacks.handle_user_input is expected to update app.storage.user
    # and chat_state.chat_history with the final message.
    # So, a final refresh of display_chat_messages should suffice.
    await display_chat_messages(chat_area)
    chat_list_container_ui.refresh() # Refresh chat list to show updated chat (e.g., last message preview)
    suggested_prompts_container_ui.refresh() # Refresh suggested prompts

async def handle_suggested_prompt(prompt: str):
    """
    Handles a click on a suggested prompt.
    """
    chat_state.set_chat_input_value(prompt)
    chat_input_ui.refresh()
    await handle_send_message()

async def handle_ui_file_upload(e: UploadEventArguments):
    """
    Handles file uploads from the UI.
    """
    file_name = e.name
    file_content_bytes = e.content.read()
    content_type = e.content_type

    if app_callbacks.handle_file_upload:
        await app_callbacks.handle_file_upload(file_name, file_content_bytes, content_type)
        uploaded_files_list_ui.refresh()
    else:
        ui.notify("File upload handler not set.", type='negative')

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
    ext_regenerate_last_response_callback: Callable[[], Coroutine[Any, Any, None]],
    ext_handle_file_upload_callback: Callable[[str, bytes, Optional[str]], Coroutine[Any, Any, None]],
    ext_remove_file_callback: Callable[[str, str], Coroutine[Any, Any, None]],
    chat_state: ChatState, # Receive the chat_state instance
    get_initial_greeting_text_cached: Callable[[], str],
    generate_suggested_prompts_cached: Callable[[Tuple], List[str]]
):
    """
    Creates the main NiceGUI interface for the chat application.
    """
    global app_callbacks, chat_input_ui, chat_area, chat_list_container_ui, suggested_prompts_container_ui, uploaded_files_list_ui
    
    # Assign external callbacks to the global app_callbacks instance
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
    app_callbacks.regenerate_last_response = ext_regenerate_last_response_callback
    app_callbacks.handle_file_upload = ext_handle_file_upload_callback
    app_callbacks.remove_file = ext_remove_file_callback
    app_callbacks.get_initial_greeting_text_cached = get_initial_greeting_text_cached
    app_callbacks.generate_suggested_prompts_cached = generate_suggested_prompts_cached

    # Assign the passed chat_state to the global one in ui.py
    globals()['chat_state'] = chat_state

    ui.add_head_html('<style>html { scroll-behavior: smooth; }</style>')
    ui.colors(primary='#585858', secondary='#F0F0F0', accent='#6C6C6C', positive='#21BA45', negative='#C10015', info='#31CCEC', warning='#FB8C00')

    with ui.header().classes('items-center justify-between text-white bg-primary'):
        ui.label('Research Assistant').classes('text-h5')
        with ui.row().classes('items-center'):
            ui.button(icon='add', on_click=lambda: app_callbacks.new_chat()).tooltip('New Chat')
            ui.button(icon='refresh', on_click=lambda: app_callbacks.regenerate_last_response()).tooltip('Regenerate Last Response')
            ui.checkbox('Long-Term Memory', value=chat_state.long_term_memory_enabled, on_change=lambda e: app_callbacks.set_long_term_memory(e.value)).props('dark')
            ui.button(icon='delete_forever', on_click=lambda: app_callbacks.forget_me()).tooltip('Forget All My Data')

    with ui.left_drawer(value=True, bordered=True).props('width=300'):
        ui.label("Chat Controls & Settings").classes("text-h6 q-pa-md")
        
        @ui.refreshable
        def chat_list_container_ui():
            ui.separator()
            ui.label("Your Chats").classes("text-subtitle1 q-mt-md q-mb-sm")
            if not chat_state.chat_names:
                ui.label("No chats yet. Start a new one!").classes("text-grey-6")
            else:
                with ui.list().classes('w-full'):
                    for chat_id, chat_name in chat_state.chat_names.items():
                        with ui.item().classes('w-full') as item:
                            item.props(f'clickable {"bg-blue-1" if chat_id == chat_state.current_chat_id else ""}')
                            item.on('click', lambda chat_id=chat_id: app_callbacks.switch_chat(chat_id))
                            with ui.item_section():
                                ui.item_label(chat_name)
                                # Show a preview of the last message
                                last_message = chat_state.chat_history.get(chat_id, [])[-1]["content"] if chat_state.chat_history.get(chat_id) else "Empty chat"
                                ui.item_label(last_message).classes('text-caption text-grey-6 truncate')
                            with ui.item_section().props('side'):
                                with ui.icon('more_vert').props('size=sm').classes('cursor-pointer'):
                                    with ui.menu() as menu:
                                        ui.menu_item('Rename', on_click=lambda chat_id=chat_id, current_name=chat_name, menu=menu: handle_rename_chat_dialog(chat_id, current_name, menu))
                                        ui.menu_item('Delete', on_click=lambda chat_id=chat_id, menu=menu: handle_delete_chat_confirm(chat_id, menu))
                                        ui.separator()
                                        ui.menu_item('Download as Markdown', on_click=lambda chat_id=chat_id, chat_name=chat_name, menu=menu: handle_download_md(chat_id, chat_name, menu))
                                        ui.menu_item('Download as DOCX', on_click=lambda chat_id=chat_id, chat_name=chat_name, menu=menu: handle_download_docx(chat_id, chat_name, menu))
        chat_list_container_ui()

        ui.separator()
        ui.label("Uploaded Files").classes("text-subtitle1 q-mt-md q-mb-sm")
        ui.upload(on_upload=handle_ui_file_upload, auto_upload=True, multiple=True).props('accept=".pdf,.txt,.csv,.xlsx,.xls"') \
            .classes('max-w-full')
        
        @ui.refreshable
        def uploaded_files_list_ui():
            if not chat_state.uploaded_files.get("document") and not chat_state.uploaded_files.get("dataframe"):
                ui.label("No files uploaded yet.").classes("text-grey-6")
            else:
                with ui.list().classes('w-full'):
                    for file_type, files in chat_state.uploaded_files.items():
                        if files:
                            ui.label(f"{file_type.capitalize()} Files:").classes("text-caption text-grey-7 q-mt-sm")
                            for filename in files.keys():
                                with ui.item().classes('w-full'):
                                    with ui.item_section():
                                        ui.item_label(filename)
                                    with ui.item_section().props('side'):
                                        ui.icon('delete', on_click=lambda ft=file_type, fn=filename: app_callbacks.remove_file(ft, fn)).classes('cursor-pointer text-negative')
        uploaded_files_list_ui()


    with ui.column().classes('w-full h-full no-wrap items-stretch'):
        chat_area = ui.column().classes('w-full flex-grow overflow-y-auto q-pa-md').style('height: calc(100vh - 200px);')
        
        # Initial greeting
        if not chat_state.chat_history.get(chat_state.current_chat_id):
            with chat_area:
                with ui.row().classes('w-full'):
                    with ui.card().classes('max-w-[70%] bg-gray-100'):
                        ui.label(app_callbacks.get_initial_greeting_text_cached()).classes('whitespace-pre-wrap')
        
        # Display existing messages
        ui.timer(0.1, lambda: display_chat_messages(chat_area), once=True) # Display messages after UI is built

        @ui.refreshable
        def suggested_prompts_container_ui():
            current_chat_history = chat_state.chat_history.get(chat_state.current_chat_id, [])
            # Convert list of dicts to tuple of tuples for caching
            hashable_history = tuple(tuple(msg.items()) for msg in current_chat_history)
            prompts = app_callbacks.generate_suggested_prompts_cached(hashable_history)
            if prompts:
                with ui.row().classes('w-full justify-center q-mb-md'):
                    for prompt in prompts:
                        ui.button(prompt, on_click=lambda p=prompt: handle_suggested_prompt(p)) \
                            .props('outline rounded dense color=grey-7') \
                            .classes('q-ma-xs')
        suggested_prompts_container_ui()

        with ui.row().classes('w-full bg-white q-pa-md items-center'):
            with ui.input(placeholder='Type your message here...').props('rounded outlined dense').classes('flex-grow') as chat_input_ui:
                chat_input_ui.bind_value(chat_state, 'chat_input_value')
                chat_input_ui.on('keydown.enter', handle_send_message)
            ui.button(icon='send', on_click=handle_send_message).props('round flat').classes('ml-2')

    async def handle_rename_chat_dialog(chat_id: str, current_name: str, parent_menu: ui.menu):
        parent_menu.close() # Close the context menu
        with ui.dialog() as dialog, ui.card():
            ui.label("Rename Chat").classes("text-h6")
            new_name_input = ui.input("New Name", value=current_name).props('outlined dense')
            with ui.row():
                ui.button("Cancel", on_click=dialog.close).props('flat')
                ui.button("Rename", on_click=lambda: (
                    app_callbacks.rename_chat(chat_id, new_name_input.value),
                    dialog.close(),
                    chat_list_container_ui.refresh() # Refresh chat list after rename
                ))
        dialog.open()

    async def handle_delete_chat_confirm(chat_id: str, parent_menu: ui.menu):
        parent_menu.close() # Close the context menu
        with ui.dialog() as dialog, ui.card():
            ui.label("Confirm Deletion").classes("text-h6")
            ui.label(f"Are you sure you want to delete chat '{chat_state.chat_names.get(chat_id, chat_id)}'? This cannot be undone.").classes("q-mt-md")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close).props('flat')
                ui.button("Delete", on_click=lambda: (
                    app_callbacks.delete_chat(chat_id),
                    dialog.close(),
                    chat_list_container_ui.refresh(), # Refresh chat list after delete
                    display_chat_messages(chat_area), # Refresh chat area if current chat deleted
                    suggested_prompts_container_ui.refresh() # Refresh suggested prompts
                )).props('color=negative')
        dialog.open()

    async def handle_download_md(chat_id: str, chat_name: str, parent_menu: ui.menu):
        parent_menu.close()
        if app_callbacks.get_discussion_markdown:
            markdown_content = app_callbacks.get_discussion_markdown(chat_id)
            # Use NiceGUI's download function
            ui.download(
                data=markdown_content.encode('utf-8'),
                filename=f'{chat_name.replace(" ", "_")}_discussion.md',
                mime_type='text/markdown'
            )
            ui.notify(f"Downloading '{chat_name}' as Markdown.", type='info')
        else:
            ui.notify("Markdown download handler not set.", type='negative')

    async def handle_download_docx(chat_id: str, chat_name: str, parent_menu: ui.menu):
        parent_menu.close()
        if app_callbacks.get_discussion_docx:
            docx_bytes = app_callbacks.get_discussion_docx(chat_id)
            if docx_bytes.startswith(b"Error:"): # Check for error message from the callback
                ui.notify(docx_bytes.decode('utf-8'), type='negative')
                return
            ui.download(
                data=docx_bytes,
                filename=f'{chat_name.replace(" ", "_")}_discussion.docx',
                mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
            ui.notify(f"Downloading '{chat_name}' as DOCX.", type='info')
        else:
            ui.notify("DOCX download handler not set.", type='negative')


if __name__ in {"__main__", "__mp_main__"}:
    # Define storage paths for ui.run in standalone test mode
    TEST_USER_STORAGE_DIR = os.path.join(PROJECT_ROOT, '.nicegui_test_storage', 'user')
    TEST_GENERAL_STORAGE_DIR = os.path.join(PROJECT_ROOT, '.nicegui_test_storage', 'general')

    # Ensure test storage directories exist
    os.makedirs(TEST_USER_STORAGE_DIR, exist_ok=True)
    os.makedirs(TEST_GENERAL_STORAGE_DIR, exist_ok=True)

    # Configure NiceGUI to use test storage paths
    app.storage.user.set_path(TEST_USER_STORAGE_DIR)
    app.storage.general.set_path(TEST_GENERAL_STORAGE_DIR)
    app.storage.user.set_storage_secret(STORAGE_SECRET) # Use the defined STORAGE_SECRET

    @ui.page('/')
    async def index_page(client: Client):
        # Clear storage for a clean test run
        app.storage.user.clear()
        # Re-initialize chat_state for the test run
        globals()['chat_state'] = ChatState()
        await app.storage.user.clear() # Clear user storage for a clean test run

        # Mock implementations for testing the UI in isolation
        async def mock_handle_input_app(text):
            print(f"Mock handle_user_input: {text}")
            # Simulate streaming response
            response_chunks = [f"You said: '{text}'. ", "This is a mock response. ", "How can I assist further?"]
            for chunk in response_chunks:
                yield chunk
                await asyncio.sleep(0.1) # Simulate delay

        async def mock_handle_upload_app(name, content, content_type):
            print(f"Mock handle_file_upload: {name}, type: {content_type}, size: {len(content)} bytes")
            file_type = "dataframe" if content_type and ('csv' in content_type or 'excel' in content_type) else "document"
            if file_type not in chat_state.uploaded_files:
                chat_state.uploaded_files[file_type] = {}
            chat_state.uploaded_files[file_type][name] = {"content": content, "content_type": content_type}
            ui.notify(f"Mock uploaded {name}", type='positive')

        async def mock_remove_file_app(ftype, fname):
            print(f"Mock remove_file: {ftype}/{fname}")
            if ftype in chat_state.uploaded_files and fname in chat_state.uploaded_files[ftype]:
                del chat_state.uploaded_files[ftype][fname]
                ui.notify(f"Mock removed {fname}", type='info')
            else:
                ui.notify(f"Mock file {fname} not found", type='warning')

        async def mock_set_ltm(val: bool):
            print(f"Mock set_long_term_memory: {val}")
            chat_state.long_term_memory_enabled = val
            ui.notify(f"Mock LTM set to {val}", type='info')

        async def mock_new_chat():
            print("Mock new chat")
            new_chat_id = str(uuid.uuid4())
            chat_state.chat_history[new_chat_id] = []
            chat_state.chat_names[new_chat_id] = f"Mock Chat {len(chat_state.chat_history)}"
            chat_state.current_chat_id = new_chat_id
            ui.notify("Mock new chat started!", type='positive')

        async def mock_delete_chat(chat_id: str):
            print(f"Mock delete chat: {chat_id}")
            if chat_id in chat_state.chat_history:
                del chat_state.chat_history[chat_id]
                del chat_state.chat_names[chat_id]
                if chat_state.current_chat_id == chat_id:
                    chat_state.current_chat_id = None
                ui.notify(f"Mock chat {chat_id} deleted.", type='info')
            
        async def mock_rename_chat(chat_id: str, new_name: str):
            print(f"Mock rename chat: {chat_id} to {new_name}")
            if chat_id in chat_state.chat_names:
                chat_state.chat_names[chat_id] = new_name
                ui.notify(f"Mock chat renamed to {new_name}.", type='positive')

        async def mock_switch_chat(chat_id: str):
            print(f"Mock switch chat: {chat_id}")
            if chat_id in chat_state.chat_history:
                chat_state.current_chat_id = chat_id
                ui.notify(f"Mock switched to chat {chat_id}.", type='info')

        async def mock_regenerate():
            print("Mock regenerate last response")
            if chat_state.last_user_query:
                ui.notify("Mock regenerating last response...", type='info')
                # Simulate regeneration by re-sending the last query
                # In a real scenario, you'd use chat_state.last_chat_history_before_regen
                # and call the agent with it.
                await mock_handle_input_app(chat_state.last_user_query)
            else:
                ui.notify("No last query to regenerate.", type='warning')

        def mock_get_discussion_markdown(chat_id: str) -> str:
            print(f"Mock get markdown for {chat_id}")
            return f"# Mock Markdown for {chat_state.chat_names.get(chat_id, chat_id)}\n\nMock content."

        def mock_get_discussion_docx(chat_id: str) -> bytes:
            print(f"Mock get docx for {chat_id}")
            return b"Mock DOCX content."

        def mock_get_initial_greeting_text_cached() -> str:
            return "Hello from mock UI! How can I help you today?"

        def mock_generate_suggested_prompts_cached(chat_history_tuple: tuple) -> List[str]:
            if len(chat_history_tuple) < 2:
                return ["Mock Prompt A", "Mock Prompt B"]
            else:
                return ["Mock Follow-up C", "Mock Follow-up D"]

        # Initialize chat_state for the mock UI
        await mock_new_chat() # Start with a new mock chat

        create_nicegui_interface(
            ext_handle_user_input_callback=mock_handle_input_app,
            ext_reset_callback=mock_new_chat, # Use mock_new_chat for reset in test
            ext_new_chat_callback=mock_new_chat,
            ext_delete_chat_callback=mock_delete_chat,
            ext_rename_chat_callback=mock_rename_chat,
            ext_switch_chat_callback=mock_switch_chat,
            ext_get_discussion_markdown_callback=mock_get_discussion_markdown,
            ext_get_discussion_docx_callback=mock_get_discussion_docx,
            ext_forget_me_callback=lambda: (app.storage.user.clear(), globals()['chat_state'].__init__(), mock_new_chat(), ui.notify("Mock data forgotten.", type='positive')),
            ext_set_long_term_memory_callback=mock_set_ltm,
            ext_regenerate_last_response_callback=mock_regenerate,
            ext_handle_file_upload_callback=mock_handle_upload_app,
            ext_remove_file_callback=mock_remove_file_app,
            chat_state=chat_state, # Pass the mock chat_state
            get_initial_greeting_text_cached=mock_get_initial_greeting_text_cached,
            generate_suggested_prompts_cached=mock_generate_suggested_prompts_cached
        )

    ui.run(storage_secret=STORAGE_SECRET)
