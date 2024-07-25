import streamlit as st
from extract_text import extract_text_from_pdf, segment_text
from preprocess import preprocess_text
from ML import (
    encode_text,
    train_doc2vec_model,
    infer_doc2vec_embeddings,
    calculate_similarities,
)


def main():
    st.set_page_config(page_title="PDF Topic Matcher", layout="wide")
    st.title("PDF Topic Matching Using NLP")

    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        st.header("2. Enter Topics")
        topic_list = st.text_area("Enter topics (one per line)")
        topics = topic_list.split("\n") if topic_list else []

        if st.button("Match Topics"):
            if not uploaded_file:
                st.error("Please upload a PDF file.")
            elif not topics:
                st.error("Please enter at least one topic.")
            else:
                process_pdf(uploaded_file, topics)

    with col2:
        st.header("3. Results")
        if "results" in st.session_state:
            display_results(st.session_state.results)


def process_pdf(uploaded_file, topics):
    with st.spinner("Processing PDF and matching topics..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        paragraphs = segment_text(pdf_text)

        bert_matches, doc2vec_matches = match_topics(paragraphs, topics)

        st.session_state.results = {
            "paragraphs": paragraphs,
            "topics": topics,
            "bert_matches": bert_matches,
            "doc2vec_matches": doc2vec_matches,
        }


def display_results(results):
    paragraphs = results["paragraphs"]
    bert_matches = results["bert_matches"]
    doc2vec_matches = results["doc2vec_matches"]

    for i, paragraph in enumerate(paragraphs):
        with st.expander(f"Paragraph {i+1}"):
            st.write(paragraph)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**BERT Match:** {bert_matches[i]}")
            with col2:
                st.markdown(f"**Doc2Vec Match:** {doc2vec_matches[i]}")


if __name__ == "__main__":
    main()
