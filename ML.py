import torch
from transformers import BertTokenizer, BertModel
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# BERT-based Embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text(texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        all_embeddings.extend(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return all_embeddings

# Doc2Vec Embeddings
def train_doc2vec_model(paragraphs, topics):
    documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(paragraphs + topics)]
    model = Doc2Vec(documents, vector_size=128, window=5, min_count=2, workers=4)
    return model

def infer_doc2vec_embeddings(model, texts):
    return [model.infer_vector(text.split()) for text in texts]

# Calculate Cosine Similarity
def calculate_similarities(paragraph_embeddings, topic_embeddings, batch_size=8):
    similarities = []
    for i in range(0, len(paragraph_embeddings), batch_size):
        batch_paragraph_embeddings = paragraph_embeddings[i:i+batch_size]
        batch_similarities = cosine_similarity(batch_paragraph_embeddings, topic_embeddings)
        similarities.extend(batch_similarities)
    return similarities
