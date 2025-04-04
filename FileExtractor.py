import fitz

class FileExtractor:

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text

    def extract_text_from_txt(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read()