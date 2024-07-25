import spacy

nlp = spacy.load('en_core_web_sm')

def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        if text:
            doc = nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            processed_texts.append(' '.join(tokens))
    return processed_texts
