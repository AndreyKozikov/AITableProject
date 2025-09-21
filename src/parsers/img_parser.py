from paddleocr import PPStructureV3, PaddleOCR
import pandas as pd
from PIL import Image
from src.utils.config import PARSING_DIR
from src.utils.preprocess_image import preprocess_image
import numpy as np
import re, ast
from src.mapper.ask_qwen2 import ask_qwen2
from src.utils.df_utils import write_to_json, clean_dataframe
from src.utils.registry import register_parser



def load_images(image_path):
    return np.array(Image.open(image_path).convert("RGB"))


@register_parser(".jpg", ".jpeg", "")
def image_ocr(image_path):
    image = load_images(image_path)
    ocr_model = PaddleOCR(use_angle_cls=True, lang="ru")
    files = []
    table_engine = PPStructureV3(ocr_version="PP-OCRv5", lang="ru", device="cpu")
    results = table_engine.predict(image)
    for i, r in enumerate(results):
        print(f"{i}: {r}")
        for item in r['parsing_res_list']:
            if item.label == 'table':
                html = item.content
            else:
                continue
        dfs = pd.read_html(html)
        if dfs:
            df=dfs[0]
            df = clean_dataframe(df, use_languagetool=True)
            file_path = PARSING_DIR / f"table_{i}.xlsx"
            df.to_excel(file_path, index=False)
            files.append(file_path)
    return files


def extract_table(answer: str):
    match = re.search(r"\[.*\]", answer, re.S)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except:
            pass
    return None


#@register_parser(".jpg", ".jpeg", ".png")
def image_ocr_dd(image_path):
    image_path = preprocess_image(image_path)
    # -------------------- Параметры окружения --------------------
    DD_USE_TORCH = True  # True=PyTorch, False=TensorFlow
    CUDA_VISIBLE_DEVICES = ""  # ""=CPU, "0"=GPU0 и т.д.
    TESSDATA_PREFIX = r"C:\\Program Files\\Tesseract-OCR\\tessdata"  # путь к tessdata
    OCR_LANGS = "eng+rus"  # языки Tesseract (пример: "rus+eng")
    # -------------------------------------------------------------

    import os
    os.environ["DD_USE_TORCH"] = "1" if DD_USE_TORCH else "0"
    os.environ.pop("DD_USE_TF", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

    # --- Импорт и сборка пайплайна ---
    from deepdoctection.analyzer.dd import get_dd_analyzer
    from deepdoctection.pipe.doctectionpipe import DoctectionPipe
    from deepdoctection.extern.tessocr import TesseractOcrDetector
    from deepdoctection.utils.fs import get_configs_dir_path
    from deepdoctection.pipe.text import TextExtractionService
    from deepdoctection.analyzer.config import cfg as _cfg

    # Сначала собираем стандартный пайплайн с Tesseract и сегментацией таблиц
    custom_args = [
        "USE_TABLE_SEGMENTATION=True",
        "USE_OCR=True",
        "OCR.USE_TESSERACT=True",
        "OCR.USE_DOCTR=False",
        "USE_PDF_MINER=False",
    ]

    pipe: DoctectionPipe = get_dd_analyzer(
        reset_config_file=False,
        load_default_config_file=True,
        config_overwrite=custom_args,
    )

    # Затем создаём собственный TesseractOcrDetector и задаём язык как в примере
    _tess_yaml = get_configs_dir_path() / _cfg.OCR.CONFIG.TESSERACT
    ocr = TesseractOcrDetector(_tess_yaml.as_posix())
    ocr.config.freeze(False)
    ocr.config.LANGUAGES = OCR_LANGS  # например "rus+eng"
    ocr.config.LINES = True
    ocr.config.psm = 1
    ocr.config.freeze(True)

    # Подменяем компонент OCR в пайплайне
    _text_comp = TextExtractionService(ocr)
    for _i, _comp in enumerate(pipe.pipe_component_list):
        if _comp.__class__.__name__ == "TextExtractionService":
            pipe.pipe_component_list[_i] = _text_comp
            break

    # --- Чтение изображения и запуск ---
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # output="page" вернёт Page-объекты со всеми таблицами
    try:
        pages_df = pipe.analyze(path=image_path, bytes=img_bytes, file_type=".jpg", output="page")
        pages_df.reset_state()
    except Exception as e:
        import traceback
        print("Pipeline failed:", type(e).__name__, e)
        traceback.print_exc()
        raise
    # --- Извлечение таблиц ---
    from IPython.display import display
    import pandas as pd
    for page in pages_df:
        print(f"\n=== PAGE {page.page_number} ===")
        if not page.tables:
            print("Таблиц не найдено")
        for i, table in enumerate(page.tables, 1):
            print(f"--- Table {i} ---")
            df_tbl = pd.DataFrame(table.csv)  # Table не имеет .df(); используем csv->DataFrame
            display(df_tbl)
            # При необходимости сохраните:
            # df_tbl.to_excel(f"table_{page.page_number}_{i}.xlsx", index=False)


def extract_table(answer: str):
    match = re.search(r"\[.*\]", answer, re.S)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except:
            pass
    return None


#@register_parser(".jpg", ".jpeg")
def parse_images_ai(image_path):
    image = load_images(image_path)
    image = preprocess_image(image_path, image)
    prompt = """
    Ты — распознаватель таблиц. 
    Нужно извлечь все строки и все столбцы таблицы.
    Формат ответа:
    - Строго список списков (Python-подобный массив).
    - Каждый вложенный список = одна строка таблицы.
    - Первый вложенный список = заголовки (если они есть).
    - Остальные вложенные списки = все строки таблицы по порядку.
    - В каждой строке должно быть одинаковое количество элементов (как в таблице).
    - Не добавляй пояснений, текста или комментариев вне списка.
    - Тип данных в одном столбце должен быть одинаковый для всех строк.
    
    Пример правильного ответа:
    [["Колонка1", "Колонка2"],
     ["Значение1", "Значение2"]]
    """
    answer = ask_qwen2(image_path=image, prompt=prompt)
    print(f"answer: {answer}")
    data = extract_table(answer)
    print(f"data: {data}")
    file_name = f"{image_path.stem}.json"
    print(file_name)
    out_path = PARSING_DIR / f"{file_name}"
    write_to_json(out_path, data)
    return file_name
