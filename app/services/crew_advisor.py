# crew_advisor.py

from dotenv import load_dotenv
import json
import os

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from typing import Dict

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Shared LLM instance
llm = ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

# Simple memory store per user_id (in-memory; replace with Redis in prod)
memory_store: Dict[str, ConversationBufferMemory] = {}

# Define agents
def create_agents(llm_instance):
    return {
        "Sizing Expert": Agent(
            role="Sizing Expert",
            goal="Recommend accurate solar system sizing using user needs and contextual knowledge.",
            backstory="A seasoned electrical and solar system engineer with 20 years of off-grid and grid-tied design experience.",
            tools=[],
            llm=llm_instance,
            allow_delegation=False,
        ),
        "Cost Optimizer": Agent(
            role="Cost Optimizer",
            goal="Suggest ways to reduce cost of panels, batteries, and inverters using the most efficient components.",
            backstory="An experienced solar procurement strategist and expert with versatile knowledge of African markets.",
            tools=[],
            llm=llm_instance,
            allow_delegation=False,
        ),
        "Maintenance Troubleshooter": Agent(
            role="Maintenance Troubleshooter",
            goal="Diagnose solar system issues based on user complaints and suggest solutions.",
            backstory="An expert in post-installation maintenance and solar diagnostics.",
            tools=[],
            llm=llm_instance,
            allow_delegation=False,
        )
    }

# Determine agent(s) based on query
def llm_route(user_query: str, context: str) -> list[str]:
    routing_prompt = [
        SystemMessage(content="You're an AI router. Based on the user's query and context, return a JSON list of the most appropriate agent(s) for the task."),
        HumanMessage(content=f"""
User Query: {user_query}
Context: {context}

Available Agents:
1. Sizing Expert - for load analysis and solar component sizing (inverter, battery, panels)
2. Cost Optimizer - for recommending cost-effective solar system components
3. Maintenance Troubleshooter - for diagnosing issues, risks, or failures in a solar setup

Return only the agent names as a JSON list. Example: ["Sizing Expert", "Cost Optimizer"]
""")
    ]
    response = llm(routing_prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return ["Sizing Expert"]

# Determine if the query is solar-related
def is_solar_related(query: str) -> bool:
    check_prompt = [
        SystemMessage(content="You're a domain classifier. Determine if the following query is related to solar energy systems."),
        HumanMessage(content=f"Query: {query}\n\nAnswer only with true or false.")
    ]
    response = llm(check_prompt)
    return "true" in response.content.lower()

# Core function to run the agent crew with context and memory
def run_crew_with_context(user_query: str, context: str, user_id: str = "default_user") -> str:
    # Get memory for user or create new
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(return_messages=True)

    memory = memory_store[user_id]
    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(f"[context used]\n{context}")

    agents = create_agents(llm)
    selected_agents = llm_route(user_query, context)

    tasks = []

    if "Sizing Expert" in selected_agents:
        tasks.append(Task(
            agent=agents["Sizing Expert"],
            description=f"User Query: {user_query}\nContext: {context}\nAnalyze user energy needs and recommend accurate inverter, battery, and panel sizing.",
            expected_output="Detailed component sizing recommendation."
        ))

    if "Cost Optimizer" in selected_agents:
        tasks.append(Task(
            agent=agents["Cost Optimizer"],
            description=f"Based on user's needs and context: {context}\nSuggest cost-saving strategies for system efficiency.",
            expected_output="Optimized solar component list with cost recommendations."
        ))

    if "Maintenance Troubleshooter" in selected_agents:
        tasks.append(Task(
            agent=agents["Maintenance Troubleshooter"],
            description=f"User Query: {user_query}\nContext: {context}\nProvide maintenance recommendations or diagnose potential issues.",
            expected_output="Preventive maintenance tips and troubleshooting guidance."
        ))

    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        verbose=False
    )

    result = crew.kickoff()

    # Store final result in memory
    memory.chat_memory.add_ai_message(result)

    return result
