# Hybrid RAG with LangGraph and ChromaDB

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LangGraph**, **ChromaDB**, and real-time **Tavily** web search. It combines query transformation, hybrid retrieval, reranking, and answer generation.

# Overview

The pipeline consists of:

- **Routing**: Routes queries to web search (Tavily) or vectorstore retrieval.
- **Hybrid Retrieval**: Combines BM25 (keyword) and ChromaDB (vector) search.
- **Query Transformation**: Improves queries using step-back and sub-queries.
- **Reranking**: LLM-based scoring ensures relevant documents are used.
- **Answer Generation**: Generates concise answers grounded in retrieved context.
- **Grading**: Validates the generated answers for relevance and accuracy.

The demo notebook **`test.ipynb`** contains a small example of the LangGraph app and its components in action.

# QA Pair Generation and Testing Module

This module provides functionality for generating and evaluating factoid **Question-Answer (QA)** pairs within the context of a Retrieval-Augmented Generation (RAG) system. It helps ensure that the system retrieves relevant information and generates accurate, well-grounded answers.

The notebook **`rag_testing.ipynb`** contains examples of the components of the testing system.

# Future Work

- Caching for repeated queries.
- Advanced grading metrics for better validation.
- Chat persistence.
- Fleshed out example notebooks.
