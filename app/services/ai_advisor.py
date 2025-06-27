from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
from app.services.crew_advisor import run_crew_with_context
import os

load_dotenv()


CHROMA_DB_DIR = "app/services/db"

def get_rag_context(query: str) -> str:
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=OpenAIEmbeddings()
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0.2),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    result = chain({"query": query})
    return result["result"]


def query_advisor(question: str, user_id=None) -> str:
    rag_context = get_rag_context(question)
    final_response = run_crew_with_context(user_query=question, context=rag_context)
    return final_response