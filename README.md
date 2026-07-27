# Advanced Retrieval-Augmented Generation (RAG)

This project compares three Retrieval-Augmented Generation (RAG) pipelines using LlamaIndex.

## Features

- Basic Vector Retrieval
- Sentence Window Retrieval
- Hierarchical Chunking
- Auto-Merging Retriever
- BAAI Embedding Models
- BAAI Reranker
- OpenAI GPT Integration
- Retrieval Performance Comparison

## Technologies

Python

LlamaIndex

OpenAI API

Sentence Transformers

HuggingFace Embeddings

Vector Database

## Retrieval Methods

1. Basic RAG
2. Sentence Window Retrieval
3. Hierarchical Auto-Merging Retrieval

## Installation

pip install -r requirements.txt

## Usage

python src/main.py --input-file data/sample_document.txt

## Example

Query:

Who is the beautiful person in Hong Kong?

The system retrieves the most relevant chunks and generates answers using GPT.
