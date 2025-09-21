# file: app.py
import streamlit as st
from pathlib import Path
import re
import hashlib

from contourpy.util.data import simple

from src.utils.config import INBOX_DIR
from utils.process_files import process_files
from datetime import datetime

st.title("📂 Загрузка файлов")

# инициализация состояния
if "saved_files" not in st.session_state:
    st.session_state.saved_files = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "upload_map" not in st.session_state:
    # Maps file signature -> saved Path
    st.session_state.upload_map = {}

st.subheader("Загрузите файлы")
uploaded_files = st.file_uploader(
    "Перетащите файлы сюда или выберите вручную",
    type=["txt", "csv", "xlsx", "xls", "pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

def _transliterate_ru_to_latin(text: str) -> str:
    mapping = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E', 'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y',
        'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F',
        'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y',
        'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f',
        'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    return ''.join(mapping.get(ch, ch) for ch in text)


def _sanitize_stem(stem: str) -> str:
    translit = _transliterate_ru_to_latin(stem)
    translit = translit.strip()
    translit = re.sub(r"[^A-Za-z0-9._-]+", "_", translit)
    translit = re.sub(r"_+", "_", translit)
    translit = translit.strip("._-")
    translit = translit.lower()
    return translit or "file"


def _make_unique_name(original_name: str, start_index: int) -> tuple[str, int]:
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
    # Stable signature across reruns for the same content
    buf = uploaded_file.getbuffer()
    md5 = hashlib.md5(buf).hexdigest()
    return f"{uploaded_file.name}:{uploaded_file.size}:{md5}"

# сохраняем файлы при загрузке
if uploaded_files:
    # Ensure deterministic handling across reruns: only save new files
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

    st.success(f"Загружено файлов: {len(st.session_state.saved_files)}")
    for f in st.session_state.saved_files:
        st.write(f"✅ {f}")

# показываем кнопку только если есть файлы и ещё не обработано
if st.session_state.saved_files and not st.session_state.processed:
    # выбор модели
    model_choice = st.selectbox(
        "Выберите модель",
        ["Локальная модель", "Модель ChatGPT 5"]
    )
    remote_model = True
    if model_choice == "Локальная модель":
        remote_model = False

    mode = st.radio(
        "Выберите режим распределения",
        ["умное распределение", "упрощенное распределение"]
    )
    if st.button("🚀 Начать обработку"):
        with st.spinner("Обработка файлов..."):
            start_time = datetime.now()
            st.write(f"⏳ Время начала обработки: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            result = process_files(st.session_state.saved_files,
                                   extended=(mode == "умное распределение"),
                                   remote_model=remote_model)
        st.success("✅ Обработка завершена")
        end_time = datetime.now()
        st.write(f"🏁 Время окончания обработки: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"⏱ Длительность: {end_time - start_time}")

        file_path = result
        if file_path.exists():
            with open(file_path, "rb") as f:
                st.download_button(
                    label="📥 Скачать результат",
                    data=f,
                    file_name=file_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("Файл результата не найден")

        # сбрасываем состояние
        st.session_state.saved_files = []
        st.session_state.processed = True
        st.session_state.upload_map = {}

# если всё обработано, предложить загрузить новые
if st.session_state.processed:
    st.info("Загрузите новые файлы для следующей обработки")
    # сбросим processed, как только пользователь загрузит новые файлы
    if uploaded_files:
        st.session_state.processed = False