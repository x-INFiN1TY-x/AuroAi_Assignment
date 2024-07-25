from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ''

def segment_text(text):
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]
