from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

CHROMA_DB_DIR = "app/services/db"

def load_documents():
    loaders = [
        UnstructuredFileLoader("app/services/documents/inverter_guides.pdf"),
        UnstructuredFileLoader("app/services/documents/solar_panel_specs.txt"),
        UnstructuredFileLoader("app/services/documents/battery_optimization.md"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def build_vectorstore():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_DB_DIR
    )
    db.persist()
    return db
