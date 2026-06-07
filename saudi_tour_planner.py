"""
Saudi Tour Planner - Multi-Agent System
A 4-agent system for planning tours in Saudi Arabia based on user's desired city.

Agents:
1. Web Research Agent - Gathers information about destinations, attractions, prices
2. Planner Agent - Creates a structured itinerary plan
3. Calculation Agent - Handles budget calculations and cost estimates
4. Executor Agent - Coordinates all agents and produces final tour plan
"""

import os
from getpass import getpass
from typing import Dict, List, Optional
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langsmith import Client

# Setup
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

client = Client()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tools
search_tool = DuckDuckGoSearchRun(name="Web_Search")

@tool
def calculate_math(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        allowed = {"__builtins__": {}}
        return str(eval(expression, allowed, {}))
    except Exception as e:
        return f"Error: {str(e)}"

react_prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

print("Setup complete!")

# ============================================================================
# AGENT 1: Web Research Agent
# ============================================================================

web_researcher_instruction = """You are a Web Research Agent specialized in Saudi Arabia tourism.
Your role is to gather comprehensive information about destinations, attractions, hotels, restaurants, and activities in Saudi cities.

When researching:
- Focus on the specific city mentioned by the user
- Find information about: top attractions, historical sites, cultural experiences, shopping areas
- Research accommodation options (hotels, resorts) with price ranges
- Find restaurant recommendations and local cuisine
- Look for transportation options within the city
- Search for current prices in SAR (Saudi Riyal)
- Find information about best times to visit, weather, and seasonal events

Use the Web_Search tool to gather this information. Be thorough and specific.
Return your findings in a structured format that can be used by the Planner Agent."""

web_researcher_prompt = react_prompt.partial(system_message=web_researcher_instruction)
web_researcher_agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=web_researcher_prompt
)
web_researcher_executor = AgentExecutor(
    agent=web_researcher_agent,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True
)

# ============================================================================
# AGENT 2: Planner Agent
# ============================================================================

planner_instruction = """You are a Planner Agent for Saudi Arabia tours.
Your role is to create a detailed, day-by-day itinerary based on research findings.

When creating a plan:
- Structure the itinerary by days (morning, afternoon, evening)
- Group nearby attractions together to minimize travel time
- Include a mix of cultural, historical, and entertainment activities
- Consider travel time between locations
- Plan for meal times at recommended restaurants
- Include rest periods and flexible time slots
- Ensure the plan is realistic and achievable
- Consider the user's interests if mentioned

Create a numbered day-by-day plan with specific times and activities.
Do not include prices or calculations - that's the Calculation Agent's job.
Focus on the logical flow and sequencing of activities."""

planner_prompt = ChatPromptTemplate.from_template(
    """You are a Planner Agent for Saudi Arabia tours.

Research Findings:
{research_findings}

User Request: {user_request}

Create a detailed day-by-day itinerary plan. Structure it clearly with:
- Day number
- Time slots (morning, afternoon, evening)
- Specific activities and locations
- Logical flow considering travel time

Focus only on planning - no prices or calculations."""
)

planner_chain = planner_prompt | llm

# ============================================================================
# AGENT 3: Calculation Agent
# ============================================================================

calculation_instruction = """You are a Calculation Agent for Saudi Arabia tour budgets.
Your role is to calculate all costs and provide budget estimates.

When calculating:
- Use the calculate_math tool for all mathematical operations
- Calculate costs for: accommodation, meals, transportation, activities, shopping
- Provide estimates in SAR (Saudi Riyal)
- Break down costs by category and by day
- Include a total budget estimate
- Consider different budget levels (economy, mid-range, luxury) if appropriate
- Account for taxes and service charges where applicable

Use the research findings for price data and the calculate_math tool for computations.
Return a detailed budget breakdown with clear totals."""

calculation_prompt = react_prompt.partial(system_message=calculation_instruction)
calculation_agent = create_react_agent(
    llm=llm,
    tools=[search_tool, calculate_math],
    prompt=calculation_prompt
)
calculation_executor = AgentExecutor(
    agent=calculation_agent,
    tools=[search_tool, calculate_math],
    verbose=True,
    handle_parsing_errors=True
)

# ============================================================================
# AGENT 4: Executor Agent
# ============================================================================

executor_instruction = """You are the Executor Agent - the coordinator of the Saudi Tour Planner system.
Your role is to combine all agent outputs into a comprehensive, user-friendly tour plan.

When creating the final output:
- Integrate the research findings, itinerary plan, and budget calculations
- Present information in a clear, organized format
- Include practical tips and recommendations
- Add a summary section with key highlights
- Ensure all information is consistent and well-structured
- Format the output to be easy to read and follow
- Include contact information or booking suggestions where relevant

Produce a complete tour guide that the user can follow directly."""

executor_prompt = ChatPromptTemplate.from_template(
    """You are the Executor Agent coordinating a Saudi Arabia tour plan.

Research Findings:
{research_findings}

Itinerary Plan:
{itinerary_plan}

Budget Calculations:
{budget_calculations}

User Request: {user_request}

Combine all this information into a comprehensive, beautifully formatted tour plan.
Include:
- Executive summary
- Detailed day-by-day itinerary
- Complete budget breakdown
- Practical tips and recommendations
- Contact/booking information where relevant

Make it professional, easy to follow, and actionable."""
)

executor_chain = executor_prompt | llm

# ============================================================================
# Multi-Agent System Orchestration
# ============================================================================

def run_saudi_tour_planner(city: str, days: int, budget: Optional[str] = None, interests: Optional[str] = None):
    """
    Run the 4-agent Saudi Tour Planner system.
    
    Args:
        city: The Saudi city to visit (e.g., Riyadh, Jeddah, Dammam, AlUla)
        days: Number of days for the tour
        budget: Optional budget range (e.g., "5000-10000 SAR")
        interests: Optional interests (e.g., "history, food, shopping")
    """
    print("="*80)
    print("SAUDI TOUR PLANNER - MULTI-AGENT SYSTEM")
    print("="*80)
    print(f"\nDestination: {city}")
    print(f"Duration: {days} days")
    if budget:
        print(f"Budget: {budget}")
    if interests:
        print(f"Interests: {interests}")
    print("\n" + "="*80 + "\n")
    
    interactions = []
    user_request = f"Plan a {days}-day tour of {city}"
    if budget:
        user_request += f" with budget {budget}"
    if interests:
        user_request += f" focusing on {interests}"
    
    # Step 1: Web Research Agent
    print("🔍 AGENT 1: Web Research Agent - Gathering information...")
    print("-" * 80)
    research_query = f"Find comprehensive information about tourism in {city} Saudi Arabia including attractions, hotels, restaurants, activities, and prices for a {days}-day visit"
    
    research_result = web_researcher_executor.invoke({"input": research_query})
    research_findings = research_result["output"]
    
    interactions.append({
        "from": "Web Research Agent",
        "to": "Planner Agent & Calculation Agent",
        "message": "Research data gathered"
    })
    print(f"✓ Research completed ({len(research_findings)} characters)\n")
    
    # Step 2: Planner Agent
    print("📋 AGENT 2: Planner Agent - Creating itinerary...")
    print("-" * 80)
    
    plan_result = planner_chain.invoke({
        "research_findings": research_findings,
        "user_request": user_request
    })
    itinerary_plan = plan_result.content if hasattr(plan_result, "content") else str(plan_result)
    
    interactions.append({
        "from": "Planner Agent",
        "to": "Executor Agent",
        "message": "Itinerary plan created"
    })
    print(f"✓ Itinerary planned ({len(itinerary_plan)} characters)\n")
    
    # Step 3: Calculation Agent
    print("💰 AGENT 3: Calculation Agent - Computing budget...")
    print("-" * 80)
    
    calculation_query = f"""Based on this research for {city}:
{research_findings}

And this itinerary:
{itinerary_plan}

Calculate the total budget estimate in SAR for a {days}-day tour. Break down costs by category (accommodation, food, transport, activities) and provide daily estimates."""
    
    calculation_result = calculation_executor.invoke({"input": calculation_query})
    budget_calculations = calculation_result["output"]
    
    interactions.append({
        "from": "Calculation Agent",
        "to": "Executor Agent",
        "message": "Budget calculated"
    })
    print(f"✓ Budget calculated ({len(budget_calculations)} characters)\n")
    
    # Step 4: Executor Agent
    print("🎯 AGENT 4: Executor Agent - Compiling final tour plan...")
    print("-" * 80)
    
    final_result = executor_chain.invoke({
        "research_findings": research_findings,
        "itinerary_plan": itinerary_plan,
        "budget_calculations": budget_calculations,
        "user_request": user_request
    })
    final_plan = final_result.content if hasattr(final_result, "content") else str(final_result)
    
    interactions.append({
        "from": "Executor Agent",
        "to": "User",
        "message": "Complete tour plan delivered"
    })
    print("✓ Final tour plan compiled\n")
    
    # Output Final Result
    print("\n" + "="*80)
    print("FINAL SAUDI TOUR PLAN")
    print("="*80 + "\n")
    print(final_plan)
    
    # Communication Log
    print("\n" + "="*80)
    print("AGENT COMMUNICATION LOG")
    print("="*80)
    for i, msg in enumerate(interactions, 1):
        print(f"{i}. {msg['from']} → {msg['to']}: {msg['message']}")
    
    return final_plan

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Example usage - you can modify these parameters
    print("\n🏰 Saudi Tour Planner - Multi-Agent System")
    print("="*80)
    print("Available cities: Riyadh, Jeddah, Dammam, AlUla, Medina, Mecca, Abha, Tabuk")
    print("="*80 + "\n")
    
    # Get user input
    city = input("Enter the city you want to visit in Saudi Arabia: ").strip()
    days = input("Enter number of days for the tour: ").strip()
    budget = input("Enter your budget range in SAR (optional, press Enter to skip): ").strip() or None
    interests = input("Enter your interests (optional, e.g., history, food, shopping): ").strip() or None
    
    try:
        days = int(days)
        if days <= 0:
            print("Error: Days must be a positive number.")
        else:
            result = run_saudi_tour_planner(
                city=city,
                days=days,
                budget=budget,
                interests=interests
            )
    except ValueError:
        print("Error: Please enter a valid number for days.")
