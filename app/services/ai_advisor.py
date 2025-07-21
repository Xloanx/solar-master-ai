# ai_advisor.py

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone  

from app.services.crew_advisor import run_crew_with_context, is_solar_related

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "solar-master-vectorstore"

# Initialize Pinecone client once
pc = Pinecone(api_key=pinecone_api_key)

def get_vectorstore():
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )

    index = pc.Index(pinecone_index_name)
    embeddings = OpenAIEmbeddings()
    return LangchainPinecone(index, embeddings, "text")

def get_rag_context(query: str) -> str:
    db = get_vectorstore()
    docs_with_scores = db.similarity_search_with_score(query, k=4)
    threshold = 0.75
    filtered = [(doc, score) for doc, score in docs_with_scores if score >= threshold]

    if not filtered:
        return ""

    docs = [doc for doc, score in filtered]
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    return f"{context}\n\nSources:\n" + "\n".join(set(sources))

def query_advisor(question: str, user_id: str = "default_user") -> str:
    if not is_solar_related(question):
        return "Hi! I specialize in solar energy systems. Please ask a question related to solar panels, sizing, or troubleshooting."

    rag_context = get_rag_context(question)
    context_to_use = rag_context if rag_context.strip() else (
        "You are a solar energy expert. Answer the user's question based on your experience."
    )

    return run_crew_with_context(user_query=question, context=context_to_use, user_id=user_id)
