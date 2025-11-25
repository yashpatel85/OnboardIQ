import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Defined the data paths

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    print("----- Starting Data Ingestion -----")

    # We use DirectoryLoader to grab all files in the data folder
    print("Loading documents....")

    # Load text files
    text_loader = DirectoryLoader(DATA_PATH,glob = "*.txt", loader_cls = TextLoader)
    text_documents = text_loader.load()

    # Load CSV files
    csv_loader = DirectoryLoader(DATA_PATH, glob = "*.csv", loader_cls = CSVLoader)
    csv_documents = csv_loader.load()

    # Combining them
    documents = text_documents + csv_documents
    print(f"Loaded {len(documents)} documents.")

    # Splitting text into chunks
    print("Splitting text into chunks....")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks.")

    # Create Embeddings
    print("Creating embeddings (this might take a moment)....")
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device': 'cpu'})
    
    print("Building Vector Store....")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    print(f"Success! Vector Database saved to '{DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()