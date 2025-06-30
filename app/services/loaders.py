from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import json
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

CHROMA_DB_DIR = "app/services/db"

def load_documents():
    doc_paths = [
        "guide_to_the_installation_of_pv_systems_2nd_edition.pdf",
        "homeowners-guide-to-solar-pv.pdf",
        "introduction-to-solar-electricity.pdf",
        "lg_beginners_guide_to_solar.pdf",
        "solar-guide.pdf",
        "solar-panel-guide-e-book.pdf",
        "solar-power-system-an-introductory-guidebook.pdf",
        "trainer_guide.pdf",
    ]
    
    docs = []

    for path in doc_paths:
        try:
            loader = UnstructuredLoader(f"app/services/documents/{path}")
            raw_docs = loader.load()

            for item in raw_docs:
                # Handle case where item is a tuple
                if isinstance(item, tuple):
                    # Look for Document objects in the tuple
                    for element in item:
                        if isinstance(element, Document):
                            doc = element
                            doc.metadata["source"] = path
                            docs.append(doc)
                # Handle direct Document case
                elif isinstance(item, Document):
                    item.metadata["source"] = path
                    docs.append(item)
                else:
                    print(f"[SKIPPED] Unsupported document type in {path}: {type(item)}")

        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")

    return docs


def clean_metadata(doc):
    """Clean metadata for ChromaDB compatibility"""
    if not hasattr(doc, 'metadata'):
        raise ValueError("Document has no metadata attribute")
    
    # Make a copy of the original metadata
    metadata = doc.metadata.copy()
    
    # Remove problematic metadata fields
    for key in ['coordinates', 'points', 'system', 'layout_width', 'layout_height']:
        metadata.pop(key, None)
    
    # Convert complex values to strings
    for key, value in list(metadata.items()):
        if isinstance(value, (dict, list, tuple)):
            metadata[key] = str(value)
        elif not isinstance(value, (str, int, float, bool)) and value is not None:
            metadata[key] = str(value)
    
    # Create a new document with cleaned metadata
    return Document(
        page_content=doc.page_content,
        metadata=metadata
    )


def build_vectorstore():
    # Load and filter documents
    raw_documents = load_documents()
    
    if not raw_documents:
        raise ValueError("No documents were loaded - check your source files")
    
    print(f"Loaded {len(raw_documents)} raw documents")

    # Clean documents
    cleaned_docs = []
    for doc in raw_documents:
        try:
            cleaned = clean_metadata(doc)
            if cleaned.page_content.strip():  # Check for non-empty content
                cleaned_docs.append(cleaned)
        except Exception as e:
            print(f"[WARNING] Failed to clean document: {e}")

    if not cleaned_docs:
        raise ValueError("No valid documents after cleaning - check document content")

    print(f"Processing {len(cleaned_docs)} cleaned documents")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    try:
        chunks = splitter.split_documents(cleaned_docs)
    except Exception as e:
        raise ValueError(f"Failed to split documents: {e}")

    if not chunks:
        raise ValueError("No chunks created after splitting")

    print(f"Created {len(chunks)} chunks")

    # Final cleaning pass
    final_chunks = []
    for chunk in chunks:
        try:
            final_chunk = clean_metadata(chunk)
            if final_chunk.page_content.strip():
                final_chunks.append(final_chunk)
        except Exception as e:
            print(f"[WARNING] Failed to prepare final chunk: {e}")

    if not final_chunks:
        raise ValueError("No valid chunks remaining after final processing")

    print(f"Final count: {len(final_chunks)} chunks ready for embedding")

    # Create vectorstore
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Test embeddings
        test_embedding = embeddings.embed_query("test")
        if not test_embedding:
            raise ValueError("Failed to create test embedding")
        
        db = Chroma.from_documents(
            documents=final_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        db.persist()
        print("Vectorstore created successfully")
        return db
    except Exception as e:
        raise ValueError(f"Failed to create vectorstore: {e}")