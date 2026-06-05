# SpiderML Task 1 Submission

## Overview

This repository contains my submission for **SpiderML Task 1**. The submission consists of two independent machine learning projects:

1. **Base Task** – Fashion-MNIST Image Classification using PyTorch.
2. **Applied ML Domain Task** – Research Paper Assistant using Retrieval-Augmented Generation (RAG).

The repository is organized into separate folders for each task, with individual documentation and source code.

---

## Repository Structure

```text
SpiderML_Task1
│
├── README.md
│
├── base_task
│   ├── README.md
│   ├── train.py
│   ├── submission.csv
│   ├── save_models
│   │   └── fashion_mnist_model.pth
│   └── plots
│       └── training_plots.png
│
└── applied_ml_domain
    ├── README.md
    ├── app.py
    ├── requirements.txt
    ├── papers
    └── ...
```

---

# Task A: Base Task

## Fashion-MNIST Image Classification

A custom neural network was implemented using PyTorch to classify Fashion-MNIST images into 10 clothing categories.

### Features

* Custom Dataset and DataLoader implementation
* Multi-branch neural network architecture
* Skip connection in the left branch
* SGD optimizer
* CrossEntropyLoss
* Accuracy and loss tracking
* Model weight saving
* Submission file generation
* Training visualization

### Outputs

* Trained model weights (`.pth`)
* Prediction file (`submission.csv`)
* Accuracy and loss plots

Detailed documentation is available in:

```text
base_task/README.md
```

---

# Task B: Applied ML Domain Task

## Research Paper Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application built using LangChain, FAISS, HuggingFace Embeddings, Groq LLMs, and Streamlit.

The application allows users to query research papers stored as PDFs and receive context-aware answers generated from retrieved document chunks.

### Features

* PDF ingestion
* Document chunking
* Semantic search with FAISS
* HuggingFace embeddings
* Groq LLM integration
* Conversational chat interface
* Source attribution
* Streamlit web application

### Technologies Used

* LangChain
* FAISS
* HuggingFace Embeddings
* Groq
* Streamlit

Detailed documentation is available in:

```text
applied_ml_domain/README.md
```

---

## Technologies Used

* Python
* PyTorch
* Pandas
* Matplotlib
* LangChain
* FAISS
* HuggingFace Embeddings
* Groq API
* Streamlit

---

## Notes

* API keys are excluded from the repository.
* Generated files and environment-specific files are ignored through `.gitignore`.
* Each task contains its own README file with implementation details and usage instructions.

---

## Author

Ayush Giri

SpiderML Task 1 Submission
