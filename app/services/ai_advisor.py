from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from app.services.crew_advisor import run_crew_with_context
from app.services.loaders import build_vectorstore
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

CHROMA_DB_DIR = "app/services/db"

def get_or_create_vectorstore():
    """Ensure vectorstore exists before using it."""
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        print("Vectorstore missing or empty. Rebuilding it...")
        build_vectorstore()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )

def get_rag_context(query: str) -> str:
    db = get_or_create_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm = ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = chain({"query": query})
    answer = result["result"]
    sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
    
    return f"{answer}\n\nSources:\n" + "\n".join(set(sources))

def query_advisor(question: str, user_id=None) -> str:
    rag_context = get_rag_context(question)
    final_response = run_crew_with_context(user_query=question, context=rag_context)
    return final_response


