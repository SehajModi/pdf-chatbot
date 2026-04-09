# 📄 PDF Chatbot (RAG-based)

A streamlined Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content. This project uses **Groq** for lightning-fast inference and **ChromaDB** as a vector store to provide context-aware answers.

# Check out the Live-App
https://pdf-chatbot-jhmkescxkzyqev3aya2tvo.streamlit.app/

## 🚀 Features
* [cite_start]**PDF Processing**: Automatically extracts text and splits it into manageable chunks using LangChain[cite: 4].
* [cite_start]**Semantic Search**: Utilizes ChromaDB to index document sections and retrieve the most relevant context for any query[cite: 4].
* [cite_start]**AI Conversations**: Powered by Llama 3 via Groq for high-speed, precise responses[cite: 4].
* [cite_start]**Clean UI**: A minimalist, responsive interface built entirely with Streamlit[cite: 4].

## 🛠️ Tech Stack
* **Language**: Python 3.12 (Recommended)
* **Frontend**: [Streamlit](https://streamlit.io/)
* **LLM**: [Groq Cloud](https://console.groq.com/) (Llama 3.1)
* **Vector Database**: ChromaDB
* **Orchestration**: LangChain & PyPDF

## 📋 Prerequisites
* A Groq API Key (Available at the [Groq Console](https://console.groq.com/)).
* Python 3.12 (Avoid 3.14+ due to library compatibility issues with Pydantic/ChromaDB).

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SehajModi/pdf-chatbot.git](https://github.com/SehajModi/pdf-chatbot.git)
   cd pdf-chatbot
