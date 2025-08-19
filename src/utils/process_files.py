from parsers.pdf_parser import parse_pdf


def process_files(files):
    results = {}
    for file in files:
        suffix = file.suffix.lower()
        file_name = file.name
        if suffix in (".pdf",):
            results[file_name] = parse_pdf(file)
        else:
            results[file_name] = "❌ Неподдерживаемый формат"
    return results