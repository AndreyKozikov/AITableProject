# Enhanced AITableProject Application
# Implements custom UI/UX design with professional styling and enhanced user experience

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import re
import hashlib
from datetime import datetime

# Import existing utilities
from src.utils.config import INBOX_DIR
from src.utils.process_files import process_files

# Import template loader
from src.app.templates.template_loader import render_template, load_template

# Configure page settings
st.set_page_config(
    page_title="AITableProject - Professional Document Processing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_custom_css():
    """Load custom CSS from external file for compact, professional styling"""
    try:
        # Load CSS from external file
        css_file_path = Path(__file__).parent / "static" / "styles.css"
        with open(css_file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        
        # Apply CSS using st.html for better compatibility
        st.html(f"<style>{css_content}</style>")
        
    except FileNotFoundError:
        # Fallback to basic styling if CSS file not found
        st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc !important;
            color: #1e293b !important;
        }
        .main > div {
            background: #f8fafc !important;
            padding: 1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)


def create_compact_header():
    """Create compact header with modern styling"""
    header_html = load_template("header")
    st.markdown(header_html, unsafe_allow_html=True)


def create_compact_upload_component() -> list:
    """Render a single modern drag & drop zone using Streamlit's file_uploader.

    Returns a list of uploaded files.
    """
    upload_card_html = load_template("upload_card")
    st.markdown(upload_card_html, unsafe_allow_html=True)

    # Use collapsed label visibility for accessibility while keeping UI minimal
    uploaded_files = st.file_uploader(
        label="Загрузите файлы документов для обработки",
        type=["txt", "csv", "xlsx", "xls", "pdf", "doc", "docx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="main_uploader",
        label_visibility="collapsed",
    )

    return uploaded_files or []


def create_progress_tracker(current_step, total_steps, status_message):
    """Create enhanced progress tracking component"""
    progress_percentage = (current_step / total_steps) * 100
    
    progress_html = render_template(
        "progress_tracker",
        current_step=current_step,
        total_steps=total_steps,
        progress_percentage=progress_percentage,
        status_message=status_message
    )
    
    components.html(progress_html, height=150)


def _get_file_icon(file_path: Path) -> str:
    """
    Get appropriate emoji icon for file type.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Emoji icon string
    """
    ext = file_path.suffix.lower()
    icon_map = {
        '.pdf': "📄",
        '.xlsx': "📊",
        '.xls': "📊",
        '.doc': "📘",
        '.docx': "📘",
        '.jpg': "🖼️",
        '.jpeg': "🖼️",
        '.png': "🖼️",
        '.txt': "📝",
        '.csv': "📝",
    }
    return icon_map.get(ext, "📎")


def create_compact_file_list(files):
    """Create compact file list with modern styling for iframe display"""
    if not files:
        return
    
    # Build file items HTML
    file_items_html = []
    
    for file_path in files:
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        file_size_kb = file_size / 1024
        icon = _get_file_icon(Path(file_path))
        
        file_item_html = render_template(
            "file_item",
            icon=icon,
            file_name=file_name,
            file_size=f"{file_size_kb:.1f}"
        )
        file_items_html.append(file_item_html)
    
    # Render complete file list
    file_list_html = render_template(
        "file_list",
        file_count=len(files),
        file_items="".join(file_items_html)
    )
    
    components.html(file_list_html, height=400, scrolling=True)


def create_results_display(result_path, processing_time):
    """Create enhanced results display component"""
    if not result_path or not Path(result_path).exists():
        return
    
    file_size = Path(result_path).stat().st_size / 1024  # KB
    
    results_html = render_template(
        "results_display",
        file_size=f"{file_size:.1f}",
        processing_time=processing_time
    )
    
    st.markdown(results_html, unsafe_allow_html=True)


# File handling utility functions
def _transliterate_ru_to_latin(text: str) -> str:
    """
    Transliterate Russian text to Latin characters.
    
    Args:
        text: Input text in Russian
    
    Returns:
        Transliterated text in Latin characters
    """
    mapping = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E', 
        'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 
        'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 
        'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch', 
        'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 
        'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    return ''.join(mapping.get(ch, ch) for ch in text)


def _sanitize_stem(stem: str) -> str:
    """
    Sanitize filename stem for safe filesystem usage.
    
    Args:
        stem: Filename stem to sanitize
    
    Returns:
        Sanitized filename stem
    """
    translit = _transliterate_ru_to_latin(stem)
    translit = translit.strip()
    translit = re.sub(r"[^A-Za-z0-9._-]+", "_", translit)
    translit = re.sub(r"_+", "_", translit)
    translit = translit.strip("._-")
    translit = translit.lower()
    return translit or "file"


def _make_unique_name(original_name: str, start_index: int) -> tuple[str, int]:
    """
    Generate a unique filename that doesn't conflict with existing files.
    
    Args:
        original_name: Original filename
        start_index: Starting index for numbering
    
    Returns:
        Tuple of (unique_filename, used_index)
    """
    p = Path(original_name)
    base = _sanitize_stem(p.stem)
    ext = p.suffix.lower()
    idx = max(1, start_index)
    while True:
        candidate = f"{base}_{idx:03d}{ext}"
        if not (INBOX_DIR / candidate).exists():
            return candidate, idx
        idx += 1


def _file_signature(uploaded_file) -> str:
    """
    Generate a unique signature for an uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Unique file signature string
    """
    buf = uploaded_file.getbuffer()
    md5 = hashlib.md5(buf).hexdigest()
    return f"{uploaded_file.name}:{uploaded_file.size}:{md5}"


def _initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "saved_files" not in st.session_state:
        st.session_state.saved_files = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "upload_map" not in st.session_state:
        st.session_state.upload_map = {}
    if "processing_start_time" not in st.session_state:
        st.session_state.processing_start_time = None


def _handle_file_uploads(uploaded_files):
    """
    Process and save uploaded files to the inbox directory.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
    """
    if not uploaded_files:
        return
    
    next_idx = 1
    for file in uploaded_files:
        sig = _file_signature(file)
        if sig in st.session_state.upload_map:
            file_path = Path(st.session_state.upload_map[sig])
        else:
            safe_name, used_idx = _make_unique_name(file.name, next_idx)
            file_path = INBOX_DIR / safe_name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.session_state.upload_map[sig] = str(file_path)
            next_idx = used_idx + 1
        if file_path not in st.session_state.saved_files:
            st.session_state.saved_files.append(file_path)


def _render_processing_settings():
    """Render the processing settings card and handle processing logic."""
    settings_card_html = load_template("settings_card")
    st.markdown(settings_card_html, unsafe_allow_html=True)

    if st.session_state.saved_files and not st.session_state.processed:
        # Model selection
        st.markdown("**🤖 Модель ИИ**")
        model_choice = st.selectbox(
            "Выберите модель",
            ["Локальная модель Qwen2", "Облачная модель ChatGPT"],
            help="Локальная модель обеспечивает конфиденциальность, облачная - высокую точность",
            label_visibility="collapsed",
        )

        remote_model = model_choice == "Облачная модель ChatGPT"

        st.markdown("**📊 Режим обработки**")
        # Processing mode
        mode = st.radio(
            "Режим",
            ["Умное распределение", "Упрощенное распределение"],
            help="Умное распределение обеспечивает более точную категоризацию данных",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Process button
        if st.button("🚀 Начать обработку", type="primary", use_container_width=True):
            st.session_state.processing_start_time = datetime.now()

            with st.spinner("Обработка файлов..."):
                result = process_files(
                    st.session_state.saved_files,
                    extended=(mode == "Умное распределение"),
                    remote_model=remote_model,
                )

            # Calculate processing time
            end_time = datetime.now()
            processing_duration = end_time - st.session_state.processing_start_time

            st.success(f"✅ Обработка завершена за {str(processing_duration).split('.')[0]}")

            # Provide download
            if result and Path(result).exists():
                with open(result, "rb") as f:
                    st.download_button(
                        label="📥 Скачать результат Excel",
                        data=f,
                        file_name=Path(result).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True,
                    )

            # Reset state
            st.session_state.saved_files = []
            st.session_state.processed = True
            st.session_state.upload_map = {}

    elif not st.session_state.saved_files:
        st.info("👆 Загрузите файлы для начала обработки")


def main():
    """Main application entry point."""
    # Load custom styling
    load_custom_css()
    
    # Create compact header
    create_compact_header()
    
    # Initialize session state
    _initialize_session_state()
    
    # Create main content area with balanced grid
    col1, col2 = st.columns([1.2, 1], gap="medium")
    
    with col1:
        # Single compact drag & drop zone
        uploaded_files = create_compact_upload_component()
        
        # Process uploaded files
        _handle_file_uploads(uploaded_files)

        # Display uploaded files in compact format
        if st.session_state.saved_files:
            create_compact_file_list(st.session_state.saved_files)

    with col2:
        # Processing settings card
        _render_processing_settings()

    # Reset processed state when new files are uploaded
    if st.session_state.processed and uploaded_files:
        st.session_state.processed = False


if __name__ == '__main__':
    main()