# PDF-Query-System

This project is a Streamlit-based chatbot that allows users to upload multiple PDF documents and interact with their contents using natural language queries. The chatbot uses powerful natural language processing (NLP) models to answer questions based on the text extracted from the PDFs.

## Features
- **Multiple PDF Support**: Upload and interact with several PDFs at once.
- **Natural Language Interaction**: Ask questions in plain language, and the chatbot will respond based on the document contents.
- **Text Extraction**: Uses PyPDF2 to extract text from PDF files.
- **Text Chunking**: Efficiently splits text into manageable chunks using LangChain's `CharacterTextSplitter`.
- **Embeddings & Vector Search**: Embeds text using OpenAI or Hugging Face models and stores it in a FAISS vector database for fast retrieval.
- **Conversational AI**: Leverages LangChain's `ConversationalRetrievalChain` to provide coherent, context-aware responses.

## Tech Stack
- **Streamlit**: Web framework for creating interactive apps.
- **PyPDF2**: Extracts text from PDF files.
- **LangChain**: Powers text processing and conversational chains.
- **FAISS**: Vector search engine for efficient text retrieval.
- **OpenAI / Hugging Face**: Used for generating text embeddings and handling conversation logic.


