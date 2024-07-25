from pdfminer.high_level import extract_text


def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def segment_text(text):
    paragraphs = re.split(r"\s{2,}", text)
    merged_paragraphs = []
    current_paragraph = ""
    for p in paragraphs:
        if len(p) < 50:  # adjust the length threshold as needed
            current_paragraph += " " + p
        else:
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = p
    if current_paragraph:
        merged_paragraphs.append(current_paragraph.strip())
    return merged_paragraphs
