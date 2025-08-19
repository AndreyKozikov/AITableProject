# полный конфиг для PP-StructureV3 со всеми известными модулями
import os, yaml
from pathlib import Path
from paddleocr import PPStructureV3

root = r"D:\Andrew\AITableProject_cfg\official_models"
cfg_path = r"D:\Andrew\AITableProject_cfg\configs\ppstructure.yaml"

Path(root).mkdir(parents=True, exist_ok=True)
Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)

# если конфиг отсутствует или не содержит "pipeline_name" → создаём начальный
need_seed = True
if Path(cfg_path).exists():
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            tmp = yaml.safe_load(f) or {}
        need_seed = "pipeline_name" not in tmp
    except Exception:
        need_seed = True
if need_seed:
    PPStructureV3().export_paddlex_config_to_yaml(cfg_path)

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def set_nested(cfg_dict, dotted_key, value):
    cur = cfg_dict
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if not (isinstance(cur, dict) and p in cur):
            return
        cur = cur[p]
    if isinstance(cur, dict) and parts[-1] in cur:
        cur[parts[-1]] = value

def set_flat(cfg_dict, key, value):
    if key in cfg_dict:
        cfg_dict[key] = value


# ──────────────────────────────
# ВСЕ возможные модули (nested)
# ──────────────────────────────
pairs_nested = [
    # Предобработка документа
    ("SubModules.DocOrientation.model_dir", os.path.join(root, "PP-LCNet_x1_0_doc_ori")),         # ориентация страницы
    ("SubModules.PageDewarp.model_dir",     os.path.join(root, "UVDoc")),                         # выправление геометрии

    # OCR
    ("SubModules.TextlineOrientation.model_dir", os.path.join(root, "PP-LCNet_x1_0_textline_ori")), # ориентация строк
    ("SubPipelines.GeneralOCR.SubModules.TextDetection.model_dir",   os.path.join(root, "PP-OCRv5_server_det")), # детекция текста
    ("SubPipelines.GeneralOCR.SubModules.TextRecognition.model_dir", os.path.join(root, "eslav_PP-OCRv5_mobile_rec")), # распознавание текста

    # Разметка макета
    ("SubModules.LayoutDetection.model_dir",      os.path.join(root, "PP-DocBlockLayout")),       # базовый детектор блоков
    ("SubModules.LayoutDetectionPlus.model_dir",  os.path.join(root, "PP-DocLayout_plus-L")),     # расширенный детектор макета

    # Таблицы
    ("SubModules.TableClassification.model_dir",          os.path.join(root, "PP-LCNet_x1_0_table_cls")),
    ("SubModules.TableStructureWired.model_dir",          os.path.join(root, "SLANeXt_wired")),
    ("SubModules.TableStructureWireless.model_dir",       os.path.join(root, "SLANet_plus")),
    ("SubModules.WiredTableCellsDetection.model_dir",     os.path.join(root, "RT-DETR-L_wired_table_cell_det")),
    ("SubModules.WirelessTableCellsDetection.model_dir",  os.path.join(root, "RT-DETR-L_wireless_table_cell_det")),
    ("SubModules.TableOrientation.model_dir",             os.path.join(root, "PP-LCNet_x1_0_table_ori")), # бывает опционально

    # Печати
    ("SubModules.SealTextDetection.model_dir",    os.path.join(root, "PP-OCRv4_server_seal_det")),
    ("SubModules.SealTextRecognition.model_dir",  os.path.join(root, "PP-OCRv5_server_rec")),

    # Формулы
    ("SubModules.FormulaRecognition.model_dir",   os.path.join(root, "PP-FormulaNet_plus-L")),

    # Диаграммы
    ("SubModules.ChartRecognition.model_dir",     os.path.join(root, "PP-Chart2Table")),
]

# ──────────────────────────────
# ВСЕ возможные модули (flat)
# ──────────────────────────────
pairs_flat = [
    # Предобработка
    ("doc_orientation_classify_model_dir", os.path.join(root, "PP-LCNet_x1_0_doc_ori")),
    ("doc_unwarping_model_dir",            os.path.join(root, "UVDoc")),

    # OCR
    ("textline_orientation_model_dir",     os.path.join(root, "PP-LCNet_x1_0_textline_ori")),
    ("text_detection_model_dir",           os.path.join(root, "PP-OCRv5_server_det")),
    ("text_recognition_model_dir",         os.path.join(root, "eslav_PP-OCRv5_mobile_rec")),

    # Макет
    ("region_detection_model_dir",         os.path.join(root, "PP-DocBlockLayout")),
    ("layout_detection_model_dir",         os.path.join(root, "PP-DocLayout_plus-L")),

    # Таблицы
    ("table_classification_model_dir",             os.path.join(root, "PP-LCNet_x1_0_table_cls")),
    ("wired_table_structure_recognition_model_dir",os.path.join(root, "SLANeXt_wired")),
    ("wireless_table_structure_recognition_model_dir", os.path.join(root, "SLANet_plus")),
    ("wired_table_cells_detection_model_dir",      os.path.join(root, "RT-DETR-L_wired_table_cell_det")),
    ("wireless_table_cells_detection_model_dir",   os.path.join(root, "RT-DETR-L_wireless_table_cell_det")),
    ("table_orientation_classify_model_dir",       os.path.join(root, "PP-LCNet_x1_0_table_ori")),

    # Печати
    ("seal_text_detection_model_dir",      os.path.join(root, "PP-OCRv4_server_seal_det")),
    ("seal_text_recognition_model_dir",    os.path.join(root, "PP-OCRv5_server_rec")),

    # Формулы
    ("formula_recognition_model_dir",      os.path.join(root, "PP-FormulaNet_plus-L")),

    # Диаграммы
    ("chart_recognition_model_dir",        os.path.join(root, "PP-Chart2Table")),
]


# применяем
for k, v in pairs_nested: set_nested(cfg, k, v)
for k, v in pairs_flat:   set_flat(cfg, k, v)

# флаги исполнения
for k, v in {
    "use_doc_orientation_classify": True,
    "use_doc_unwarping": True,
    "use_textline_orientation": True,
    "use_region_detection": True,
    "use_table_recognition": True,
    "use_seal_recognition": False,
    "use_formula_recognition": False,
    "use_chart_recognition": False,
    "device": "cpu",
    "enable_mkldnn": True,
    "cpu_threads": 16,
}.items():
    if k in cfg:
        cfg[k] = v

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print("Обновил конфиг:", cfg_path)
