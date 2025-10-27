import streamlit as st
import json
from pathlib import Path

# Set page config
st.set_page_config(page_title="Training Data Viewer", layout="wide")

# Path to the training data file
DATA_FILE = Path(__file__).parent / "qwen3_cot_structured_train.jsonl"


@st.cache_data
def load_training_data():
    """Load training data from JSONL file."""
    data = []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        st.error(f"Файл не найден: {DATA_FILE}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Ошибка парсинга JSON: {e}")
        return []
    
    return data


def format_json_value(value):
    """Format a JSON value for display."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    elif isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, indent=2)
    else:
        return str(value)


def main():
    # Load data
    training_data = load_training_data()
    
    if not training_data:
        st.warning("Нет данных для отображения")
        return
    
    total_records = len(training_data)
    
    # Initialize session state if needed
    if 'record_num' not in st.session_state:
        st.session_state['record_num'] = 1
    
    # Sidebar with navigation controls
    with st.sidebar:
        st.header("Навигация")
        st.write(f"Всего записей: {total_records}")
        
        # Navigation buttons - must be processed BEFORE number_input
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("◀", use_container_width=True):
                if st.session_state['record_num'] > 1:
                    st.session_state['record_num'] -= 1
                    st.rerun()
        
        with col2:
            if st.button("▶", use_container_width=True):
                if st.session_state['record_num'] < total_records:
                    st.session_state['record_num'] += 1
                    st.rerun()
        
        # Record number selector - use session state value as default
        record_num = st.number_input(
            "Номер записи",
            min_value=1,
            max_value=total_records,
            value=st.session_state['record_num'],
            step=1,
            help="Выберите номер записи для просмотра",
            key="record_num_input"
        )
        
        # Update session state when number input changes
        if st.session_state['record_num_input'] != st.session_state['record_num']:
            st.session_state['record_num'] = st.session_state['record_num_input']
            st.rerun()
    
    # Ensure record_num is valid
    record_num = st.session_state['record_num']
    if record_num < 1:
        record_num = 1
        st.session_state['record_num'] = 1
    elif record_num > total_records:
        record_num = total_records
        st.session_state['record_num'] = total_records
    
    # Get current record
    current_record = training_data[record_num - 1]
    
    # Main content area
    st.title("📊 Просмотр тренировочных данных")
    
    # Record info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Текущая запись", f"{record_num} / {total_records}")
    with col2:
        mode = current_record.get('mode', 'N/A')
        st.metric("Режим", mode)
    with col3:
        progress = int((record_num / total_records) * 100)
        st.metric("Прогресс", f"{progress}%")
    
    st.divider()
    
    # Record details in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Основная информация", "👤 Система", "👨‍💻 Пользователь", "🤖 Ассистент"])
    
    with tab1:
        st.subheader("Основная информация")
        
        # Mode field
        mode_col1, mode_col2 = st.columns([1, 3])
        with mode_col1:
            st.write("**Режим:**")
        with mode_col2:
            st.code(current_record.get('mode', ''))
    
    with tab2:
        st.subheader("Системный промпт")
        system_text = current_record.get('system', '')
        st.text_area(
            "Система",
            system_text,
            height=400,
            label_visibility="collapsed"
        )
    
    with tab3:
        st.subheader("Пользовательский запрос")
        user_text = current_record.get('user', '')
        st.text_area(
            "Пользователь",
            user_text,
            height=400,
            label_visibility="collapsed"
        )
        
        # Show assistant data table on user tab as well
        assistant_data = current_record.get('assistant', {})
        if 'rows' in assistant_data and assistant_data['rows']:
            st.divider()
            st.subheader("Таблица данных")
            st.caption("📋 Ответ ассистента")
            
            # Get all column names from all rows
            all_columns = set()
            for row in assistant_data['rows']:
                all_columns.update(row.keys())
            
            columns_list = sorted(list(all_columns))
            
            if columns_list:
                # Create DataFrame for display
                import pandas as pd
                df = pd.DataFrame(assistant_data['rows'])
                st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.subheader("Ответ ассистента")
        assistant_data = current_record.get('assistant', {})
        
        # Display as formatted JSON
        st.json(assistant_data)
        
        # Also show as editable table if rows exist
        if 'rows' in assistant_data and assistant_data['rows']:
            st.divider()
            st.subheader("Таблица данных")
            
            # Get all column names from all rows
            all_columns = set()
            for row in assistant_data['rows']:
                all_columns.update(row.keys())
            
            columns_list = sorted(list(all_columns))
            
            if columns_list:
                # Create DataFrame for display
                import pandas as pd
                df = pd.DataFrame(assistant_data['rows'])
                st.dataframe(df, use_container_width=True)
    
    # Raw JSON viewer
    with st.expander("🔍 Сырой JSON"):
        st.json(current_record)


if __name__ == "__main__":
    main()
