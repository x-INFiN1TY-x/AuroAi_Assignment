from pdfminer.high_level import extract_text
import spacy


def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def segment_text(text, min_length=50, max_length=300):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    paragraphs = []
    current_paragraph = []

    for sent in doc.sents:
        current_paragraph.append(sent.text)
        current_paragraph_text = " ".join(current_paragraph).strip()

        if (
            len(current_paragraph_text) >= min_length
            and len(current_paragraph_text) <= max_length
        ):
            paragraphs.append(current_paragraph_text)
            current_paragraph = []
        elif len(current_paragraph_text) > max_length:
            words = current_paragraph_text.split()
            chunk = []
            chunk_length = 0
            for word in words:
                chunk.append(word)
                chunk_length += len(word) + 1
                if chunk_length >= max_length:
                    paragraphs.append(" ".join(chunk))
                    chunk = []
                    chunk_length = 0
            if chunk:
                paragraphs.append(" ".join(chunk))
            current_paragraph = []

    if current_paragraph:
        remaining_text = " ".join(current_paragraph).strip()
        if len(remaining_text) < min_length and paragraphs:
            paragraphs[-1] += " " + remaining_text
        else:
            paragraphs.append(remaining_text)

    return [p.strip() for p in paragraphs if p.strip()]
