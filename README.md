
# PDF Topic Matcher Using NLP

## Table of Contents

* [Overview](#overview)
* [Installation and Requirements](#installation-and-requirements)
* [Usage and Examples](#usage-and-examples)
* [Model Architecture and Training](#model-architecture-and-training)
* [Evaluation Metrics and Results](#evaluation-metrics-and-results)
* [Code Structure](#code-structure)
* [Running the App](#running-the-app)

## Overview

This project uses Natural Language Processing (NLP) techniques to match topics with paragraphs extracted from a PDF file. It utilizes two different approaches: BERT (Bidirectional Encoder Representations from Transformers) and Doc2Vec. The project is built using Python and Streamlit for the user interface.

## Installation and Requirements

To run this project locally, follow these steps:

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```
### Step 2: Download required models
```bash
python -m spacy download en_core_web_sm
```
### Step 3: Run the app
```bash
streamlit run app.py
```

## Usage and Examples

1. Upload a PDF file using the file uploader.
2. Enter topics (one per line) in the text area.
3. Click the "Match Topics" button to process the PDF and match topics.
4. View the results, including paragraph text, BERT similarities, and Doc2Vec similarities.

## Model Architecture and Training

The project uses two NLP models:

1. **BERT**: Pre-trained RoBERTa model for text encoding.
2. **Doc2Vec**: Trained on the uploaded PDF text and topics using Gensim.

## Evaluation Metrics and Results

The project uses cosine similarity to evaluate the similarity between paragraph embeddings and topic embeddings.

## Code Structure

The code is organized into the following files:

* `app.py`: Streamlit app code
* `extract_text.py`: PDF text extraction code
* `preprocess.py`: Text preprocessing code
* `ML.py`: Machine learning model code
* `ui.py`: Streamlit UI code

## Running the App

To run the app, execute the following command:
```bash
streamlit run app.py
```
