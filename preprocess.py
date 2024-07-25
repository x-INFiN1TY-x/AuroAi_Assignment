import torch
from transformers import AutoTokenizer, AutoModel
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


def preprocess_text(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    processed_texts = []
    for text in texts:
        if text:
            tokens = word_tokenize(text.lower())
            tokens = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in stop_words
            ]
            processed_texts.append(" ".join(tokens))
    return processed_texts
