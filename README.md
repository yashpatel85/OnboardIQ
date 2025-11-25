🚀 Factorial24 OnboardIQ: AI-Powered Employee Onboarding Bot

📋 Project Overview

OnboardIQ is a RAG-based (Retrieval-Augmented Generation) conversational AI designed to streamline the onboarding process at Factorial24. It allows new employees to query internal organizational data—including HR policies, technical project specifications, and team hierarchies—using natural language.

The system retrieves accurate context from internal documents (.pdf, .txt, .csv) and generates grounded responses using Google's Gemini Pro model, ensuring zero hallucinations by citing specific source documents.

🛠️ Tech Stack & Architecture

Component

Technology Used

Reason for Choice

LLM

Google Gemini Pro

High reasoning capability, low latency, and large context window.

Orchestration

LangChain

robust framework for chaining retrieval and generation steps.

Vector Database

FAISS (Facebook AI Similarity Search)

Efficient, local in-memory vector storage (privacy-friendly).

Embeddings

HuggingFace (all-MiniLM-L6-v2)

Lightweight, high-performance open-source embedding model.

Frontend

Streamlit

Rapid deployment of an interactive, chat-based UI.

🚀 Features

Context-Aware Q&A: Answers questions specifically about Factorial24 (e.g., "What is the leave policy?", "Who manages Project Alpha?").

Source Citations: Every answer includes an expandable "Reference Documents" section, proving the validity of the information.

Multi-Format Ingestion: Supports PDF, DOCX, TXT, and CSV files.

Privacy-First: Embeddings are generated locally; only the final prompt is sent to the LLM.

⚙️ Installation & Setup

1. Clone the Repository

git clone <repository-url>
cd Factorial24_OnboardIQ


2. Install Dependencies

Ensure you have Python 3.10+ installed.

pip install -r requirements.txt


3. Setup Knowledge Base

Place your organizational documents in the data/ folder and run the ingestion script:

python src/ingest.py


This will create the vectorstore folder.

4. Run the Application

streamlit run app.py


📂 Project Structure

Factorial24_OnboardIQ/
├── data/                   # Source documents (PDFs, CSVs, TXTs)
├── vectorstore/            # FAISS Index (Generated locally)
├── src/
│   └── ingest.py           # Script to process data & create embeddings
├── app.py                  # Main Chatbot Application (Streamlit)
├── requirements.txt        # Dependency list
└── README.md               # Project Documentation


🔮 Future Improvements

Hybrid Search: Implementing keyword search alongside vector search for better precision.

Admin Dashboard: A UI to upload new documents without touching the code.

Chat History: Persisting sessions using a database like SQLite.

Submitted by: [Your Name]
Date: November 27, 2025