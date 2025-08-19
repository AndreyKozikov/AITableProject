# file: app.py
import streamlit as st
from pathlib import Path

from utils import process_files

st.title("📂 Загрузка файлов")

# Папка для сохранения входящих данных
INBOX_DIR = Path("inbox")
INBOX_DIR.mkdir(exist_ok=True)

# инициализация состояния
if "saved_files" not in st.session_state:
    st.session_state.saved_files = []
if "processed" not in st.session_state:
    st.session_state.processed = False

st.subheader("Загрузите файлы")
uploaded_files = st.file_uploader(
    "Перетащите файлы сюда или выберите вручную",
    type=["txt", "csv", "xlsx", "xls", "pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# сохраняем файлы при загрузке
if uploaded_files:
    for file in uploaded_files:
        file_path = INBOX_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        if file_path not in st.session_state.saved_files:
            st.session_state.saved_files.append(file_path)

    st.success(f"Загружено файлов: {len(st.session_state.saved_files)}")
    for f in st.session_state.saved_files:
        st.write(f"✅ {f}")

# показываем кнопку только если есть файлы и ещё не обработано
if st.session_state.saved_files and not st.session_state.processed:
    if st.button("🚀 Начать обработку"):
        with st.spinner("Обработка файлов..."):
            result = process_files(st.session_state.saved_files)
        st.success("✅ Обработка завершена")
        st.write("📊 Результат:")
        st.write(result)

        # сбрасываем состояние
        st.session_state.saved_files = []
        st.session_state.processed = True

# если всё обработано, предложить загрузить новые
if st.session_state.processed:
    st.info("Загрузите новые файлы для следующей обработки")
    # сбросим processed, как только пользователь загрузит новые файлы
    if uploaded_files:
        st.session_state.processed = False