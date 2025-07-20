from dotenv import load_dotenv
import os

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone  # This is the new v3 class
# from langchain.embeddings.openai import OpenAIEmbeddings

from app.services.crew_advisor import run_crew_with_context, is_solar_related

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "solar-master-2"

# Initialize once at module level
pc = Pinecone(api_key=pinecone_api_key)

def get_vectorstore():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "solar-master-vectorstore"

    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if not exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )

    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings()

    return LangchainPinecone(index, embeddings, "text")

# def get_rag_context(query: str) -> str:
#     db = get_vectorstore()
#     retriever = db.as_retriever(search_kwargs={"k": 4})

#     chain = RetrievalQA.from_chain_type(
#         llm=ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=openai_api_key),
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )

#     result = chain({"query": query})
#     answer = result["result"]
#     sources = sorted(set(doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])))

#     return f"{answer}\n\nSources:\n" + "\n".join(sources)


def get_rag_context(query: str) -> str:
    """Fetch relevant documents with similarity filtering."""
    db = get_vectorstore()

    # Use similarity_search_with_score to get scores
    docs_with_scores = db.similarity_search_with_score(query, k=4)

    threshold = 0.75  # can be tweaked
    filtered = [(doc, score) for doc, score in docs_with_scores if score >= threshold]

    if not filtered:
        return ""

    docs = [doc for doc, score in filtered]
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]

    return f"{context}\n\nSources:\n" + "\n".join(set(sources))




def query_advisor(question: str, user_id=None) -> str:
    if not is_solar_related(question):
        return "Hi! I specialize in solar energy systems. Can you please ask a question related to solar panels, sizing, or troubleshooting?"

    rag_context = get_rag_context(question)
    if not rag_context.strip():
        return "I'm sorry, I couldn't find relevant information. Could you rephrase or ask a more specific solar-related question?"

    final_response = run_crew_with_context(user_query=question, context=rag_context)
    return final_response
