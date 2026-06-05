# Research Paper Assistant (RAG Chatbot)

## Overview

This project implement a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded research papers.

## Features

* PDF document uploading
* Text chunking
* Vector storage using FAISS
* Semantic retrieval using Hugging Face embeddings
* Answer generation using Groq LLM
* Source citation for retrieved answers

## Technologies Used

* Python
* LangChain
* FAISS
* Streamlit
* Hugging Face Embeddings
* Groq API

## How to Run

Install :

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Example Questions

* What is LoRA?
* What is Llama 2?
* Explain RAG.
* What are the advantages of instruction tuning?

## Note

The chatbot answers only from the uploaded research papers and returns source references used to generate the answer.
## Memory Handling

This project maintains basic conversational context using Streamlit's session state. User questions and chatbot responses are stored during the active session and are included in subsequent prompts to provide short-term conversational continuity but-

how current memory is implementated:

* Uses `st.session_state` to store chat history.
* Maintains context within the current browser session.
* Previous interactions are appended to prompts before generating responses.

Limitations:

* Does not implement semantic memory retrieval.
* Does not use vector-based memory storage for conversations.
* Chat history is lost when the session is restarted.
* Long-term memory across multiple sessions is not supported.

This implementation provides simple conversational context rather than a  memory retrieval system.
