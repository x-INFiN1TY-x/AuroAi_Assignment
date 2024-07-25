import streamlit as st
from extract_text import extract_text_from_pdf, segment_text
from preprocess import preprocess_text
from embeddings import encode_text, train_doc2vec_model, infer_doc2vec_embeddings, calculate_similarities

st.title("PDF Topic Matching Using NLP")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    paragraphs = segment_text(pdf_text)
    st.write("Extracted Paragraphs:")
    for paragraph in paragraphs:
        st.write(paragraph)
    
    topic_list = st.text_area("Enter topics (one per line)")
    if st.button("Match Topics"):
        topics = topic_list.split('\n')
        
        # BERT embeddings
        bert_paragraph_embeddings = encode_text(paragraphs)
        bert_topic_embeddings = encode_text(topics)
        bert_similarities = calculate_similarities(bert_paragraph_embeddings, bert_topic_embeddings)

        # Doc2Vec embeddings
        doc2vec_model = train_doc2vec_model(paragraphs, topics)
        paragraphs_preprocessed = preprocess_text(paragraphs)
        topics_preprocessed = preprocess_text(topics)
        doc2vec_paragraph_embeddings = infer_doc2vec_embeddings(doc2vec_model, paragraphs_preprocessed)
        doc2vec_topic_embeddings = infer_doc2vec_embeddings(doc2vec_model, topics_preprocessed)
        doc2vec_similarities = calculate_similarities(doc2vec_paragraph_embeddings, doc2vec_topic_embeddings)

        # Display results
        for i, paragraph in enumerate(paragraphs):
            st.write(f"Paragraph {i+1}: {paragraph}")
            bert_best_match = topics[bert_similarities[i].argmax()]
            doc2vec_best_match = topics[doc2vec_similarities[i].argmax()]
            st.write(f"BERT Matched Topic: {bert_best_match}")
            st.write(f"Doc2Vec Matched Topic: {doc2vec_best_match}\n")
