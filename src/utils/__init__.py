# Export new parsers so they are discoverable when package imported
from importlib import import_module

# Ensure side-effects (registration) happen at package import time.
for _mod in ("parsers.tesseract_img",):
    try:
        import_module(f"src.{_mod}")
    except ModuleNotFoundError:
        pass

# # Экспорт модуля постобработки OCR
# from tmp.ocr_postprocessor import OCRPostProcessor
#
# __all__ = ['OCRPostProcessor']

