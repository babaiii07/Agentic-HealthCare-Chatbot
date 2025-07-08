import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), '../healthcarechatbot_data/medical_book.pdf')
CHROMA_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')

EMBEDDING_MODEL = "models/embedding-001"


def build_vector_store():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_PATH)
    vectordb.persist()
    return vectordb


def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectordb.as_retriever()

if __name__ == "__main__":
    build_vector_store()
    print("Vector store built and persisted at:", CHROMA_PATH) 