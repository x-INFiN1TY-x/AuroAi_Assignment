import streamlit as st
from extract_text import extract_text_from_pdf, segment_text
from preprocess import preprocess_text
from ML import (
    encode_text,
    train_doc2vec_model,
    infer_doc2vec_embeddings,
    calculate_similarities,
    match_topics,
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

        bert_similarities, doc2vec_similarities = match_topics(paragraphs, topics)

        st.session_state.results = {
            "paragraphs": paragraphs,
            "topics": topics,
            "bert_similarities": bert_similarities,
            "doc2vec_similarities": doc2vec_similarities,
        }


def display_results(results):
    paragraphs = results["paragraphs"]
    topics = results["topics"]
    bert_similarities = results["bert_similarities"]
    doc2vec_similarities = results["doc2vec_similarities"]

    for i, paragraph in enumerate(paragraphs):
        with st.expander(f"Paragraph {i+1}"):
            st.write(paragraph)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### BERT Similarities")
                for topic, score in zip(topics, bert_similarities[i]):
                    st.markdown(f"**{topic}:** {score:.4f}")

                bert_best_match_index = bert_similarities[i].argmax()
                st.markdown(f"**BERT Best Match:** {topics[bert_best_match_index]}")

            with col2:
                st.markdown("### Doc2Vec Similarities")
                for topic, score in zip(topics, doc2vec_similarities[i]):
                    st.markdown(f"**{topic}:** {score:.4f}")

                doc2vec_best_match_index = doc2vec_similarities[i].argmax()
                st.markdown(
                    f"**Doc2Vec Best Match:** {topics[doc2vec_best_match_index]}"
                )


if __name__ == "__main__":
    main()
