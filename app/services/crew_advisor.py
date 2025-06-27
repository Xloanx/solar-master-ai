from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.2)

def run_crew_with_context(user_query: str, context: str):
    
    # Agent 1: Sizing Expert
    sizing_agent = Agent(
        role="Sizing Expert",
        goal="Recommend accurate solar system sizing using user needs and contextual knowledge.",
        backstory="A seasoned electrical and solar system engineer with 20 years of off-grid and grid-tied design experience.",
        tools=[],
        llm=llm,
        allow_delegation=False,
    )

    # Agent 2: Cost Optimizer
    optimization_agent = Agent(
        role="Cost Optimizer and Strategist",
        goal="Suggest ways to reduce cost of panels, batteries, and inverters using the most efficient components.",
        backstory="An experienced solar procurement strategist and expert with versatile knowledge of African markets.",
        tools=[],
        llm=llm,
        allow_delegation=False,
    )

    # Agent 3: Troubleshooter
    troubleshooter_agent = Agent(
        role="Maintenance Troubleshooter",
        goal="Diagnose solar system issues based on user complaints and suggest solutions.",
        backstory="An expert in post-installation maintenance and solar diagnostics.",
        tools=[],
        llm=llm
    )

    # Task 1: Sizing
    sizing_task = Task(
        agent=sizing_agent,
        description=f"""
        User Question: {user_query}
        Context: {context}
        Analyze user energy needs and recommend accurate inverter, battery, and panel sizing.
        """,
        expected_output="Detailed component sizing recommendation."
    )

    # Task 2: Cost Optimization
    optimization_task = Task(
        agent=optimization_agent,
        description=f"""
        Based on the sizing recommendation and context: {context}
        Suggest strategies to reduce costs while maintaining system efficiency and reliability.
        """,
        expected_output="Optimized list of solar components with estimated costs."
    )

    # Task 3: Troubleshooting & Risk Mitigation
    troubleshooting_task = Task(
        agent=troubleshooter_agent,
        description=f"""
        Consider the user scenario and system context: {context}
        Identify potential risks, common issues, and provide preventive maintenance suggestions.
        """,
        expected_output="List of troubleshooting tips and maintenance recommendations."
    )

    # Assemble the crew
    crew = Crew(
        agents=[sizing_agent, optimization_agent, troubleshooter_agent],
        tasks=[sizing_task, optimization_task, troubleshooting_task],
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()
    return result
