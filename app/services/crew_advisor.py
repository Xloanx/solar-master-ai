from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import json
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Shared LLM instance
llm = ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

# Define agents
sizing_agent = Agent(
    role="Sizing Expert",
    goal="Recommend accurate solar system sizing using user needs and contextual knowledge.",
    backstory="A seasoned electrical and solar system engineer with 20 years of off-grid and grid-tied design experience.",
    tools=[],
    llm=llm,
    allow_delegation=False,
)

optimization_agent = Agent(
    role="Cost Optimizer",
    goal="Suggest ways to reduce cost of panels, batteries, and inverters using the most efficient components.",
    backstory="An experienced solar procurement strategist and expert with versatile knowledge of African markets.",
    tools=[],
    llm=llm,
    allow_delegation=False,
)

troubleshooter_agent = Agent(
    role="Maintenance Troubleshooter",
    goal="Diagnose solar system issues based on user complaints and suggest solutions.",
    backstory="An expert in post-installation maintenance and solar diagnostics.",
    tools=[],
    llm=llm,
    allow_delegation=False,
)

# Map agent names to agent instances
agent_map = {
    "Sizing Expert": sizing_agent,
    "Cost Optimizer": optimization_agent,
    "Maintenance Troubleshooter": troubleshooter_agent
}

# Function: LLM routing logic
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
        # fallback or default if routing fails
        return ["Sizing Expert"]


# Main function
def run_crew_with_context(user_query: str, context: str):
    selected_agent_names = llm_route(user_query, context)

    if not selected_agent_names:
        return "Could not determine the appropriate expert(s). Please refine your query."

    tasks = []

    if "Sizing Expert" in selected_agent_names:
        tasks.append(Task(
            agent=agent_map["Sizing Expert"],
            description=f"""
User Query: {user_query}
Context: {context}
Analyze user energy needs and recommend accurate inverter, battery, and panel sizing.
""",
            expected_output="Detailed component sizing recommendation."
        ))

    if "Cost Optimizer" in selected_agent_names:
        tasks.append(Task(
            agent=agent_map["Cost Optimizer"],
            description=f"""
Based on the user's solar system sizing needs and context: {context}
Suggest strategies to reduce costs while maintaining system efficiency and reliability.
""",
            expected_output="Optimized list of solar components with estimated costs."
        ))

    if "Maintenance Troubleshooter" in selected_agent_names:
        tasks.append(Task(
            agent=agent_map["Maintenance Troubleshooter"],
            description=f"""
User Query: {user_query}
Context: {context}
Identify potential risks, common issues, and provide preventive maintenance suggestions.
""",
            expected_output="List of troubleshooting tips and maintenance recommendations."
        ))

    agents = list({task.agent for task in tasks})  # Ensure no duplicates

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=False  # Set to True for detailed logs
    )

    result = crew.kickoff()
    return result
