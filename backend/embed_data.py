import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
nomic = NomicEmbeddings(model="nomic-embed-text-v1.5",nomic_api_key=os.getenv("NOMIC_API_KEY"))
VECTOR_DIR = "faiss_store.pkl"

def embed_documents(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, nomic)
    vectordb.save_local(VECTOR_DIR)

def clear_vectordb():
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
