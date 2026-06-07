"""
Saudi Tour Planner - Streamlit App
A 4-agent system for planning tours in Saudi Arabia based on user's desired city.
"""

import os
import streamlit as st
from typing import Dict, List, Optional
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langsmith import Client

# Page configuration
st.set_page_config(
    page_title="Saudi Tour Planner",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1a5f2a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1a5f2a;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to use the system")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key set!")
    else:
        st.warning("⚠️ Please enter your OpenAI API Key")

# Main header
st.markdown('<h1 class="main-header">🏰 Saudi Tour Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Multi-Agent System for Planning Your Perfect Saudi Adventure</p>', unsafe_allow_html=True)

# Information about the system
with st.expander("ℹ️ About This System"):
    st.markdown("""
    This system uses **4 specialized AI agents** to plan your Saudi tour:
    
    - **🔍 Web Research Agent**: Gathers comprehensive information about destinations, attractions, hotels, restaurants, and activities
    - **📋 Planner Agent**: Creates a detailed day-by-day itinerary with logical flow and timing
    - **💰 Calculation Agent**: Computes budget estimates and cost breakdowns in SAR
    - **🎯 Executor Agent**: Coordinates all agents and produces a comprehensive final tour plan
    """)

# Input form
st.header("🎯 Plan Your Tour")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "Select City to Visit",
        ["Riyadh", "Jeddah", "Dammam", "AlUla", "Medina", "Mecca", "Abha", "Tabuk", "Khobar", "Taif"],
        help="Choose the Saudi city you want to explore"
    )
    
    days = st.slider(
        "Number of Days",
        min_value=1,
        max_value=14,
        value=3,
        help="Select the duration of your tour"
    )

with col2:
    budget = st.text_input(
        "Budget Range (SAR) - Optional",
        placeholder="e.g., 5000-10000",
        help="Enter your budget range in Saudi Riyals"
    )
    
    interests = st.text_input(
        "Interests - Optional",
        placeholder="e.g., history, food, shopping, adventure",
        help="Specify your interests to customize the tour"
    )

# Plan button
plan_button = st.button("🚀 Generate Tour Plan", type="primary", use_container_width=True)

# Function to initialize agents
@st.cache_resource
def initialize_agents(api_key):
    """Initialize all agents with the given API key."""
    os.environ["OPENAI_API_KEY"] = api_key
    
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
    
    # Web Research Agent
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
    web_researcher_agent = create_react_agent(llm=llm, tools=[search_tool], prompt=web_researcher_prompt)
    web_researcher_executor = AgentExecutor(agent=web_researcher_agent, tools=[search_tool], verbose=False, handle_parsing_errors=True)
    
    # Planner Agent
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
    
    # Calculation Agent
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
    calculation_agent = create_react_agent(llm=llm, tools=[search_tool, calculate_math], prompt=calculation_prompt)
    calculation_executor = AgentExecutor(agent=calculation_agent, tools=[search_tool, calculate_math], verbose=False, handle_parsing_errors=True)
    
    # Executor Agent
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
    
    return {
        'web_researcher_executor': web_researcher_executor,
        'planner_chain': planner_chain,
        'calculation_executor': calculation_executor,
        'executor_chain': executor_chain
    }

# Main execution
if plan_button:
    if not api_key:
        st.error("❌ Please enter your OpenAI API Key in the sidebar to continue.")
    else:
        with st.spinner("🔄 Initializing AI agents..."):
            try:
                agents = initialize_agents(api_key)
                st.success("✅ Agents initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing agents: {str(e)}")
                st.stop()
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        user_request = f"Plan a {days}-day tour of {city}"
        if budget:
            user_request += f" with budget {budget} SAR"
        if interests:
            user_request += f" focusing on {interests}"
        
        interactions = []
        
        # Step 1: Web Research
        status_text.text("🔍 Agent 1: Web Research Agent - Gathering information...")
        progress_bar.progress(25)
        
        try:
            research_query = f"Find comprehensive information about tourism in {city} Saudi Arabia including attractions, hotels, restaurants, activities, and prices for a {days}-day visit"
            research_result = agents['web_researcher_executor'].invoke({"input": research_query})
            research_findings = research_result["output"]
            interactions.append({"from": "Web Research Agent", "to": "Planner Agent & Calculation Agent", "message": "Research data gathered"})
            st.markdown('<div class="success-box">✅ Web Research completed</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Web Research failed: {str(e)}")
            st.stop()
        
        # Step 2: Planner
        status_text.text("📋 Agent 2: Planner Agent - Creating itinerary...")
        progress_bar.progress(50)
        
        try:
            plan_result = agents['planner_chain'].invoke({
                "research_findings": research_findings,
                "user_request": user_request
            })
            itinerary_plan = plan_result.content if hasattr(plan_result, "content") else str(plan_result)
            interactions.append({"from": "Planner Agent", "to": "Executor Agent", "message": "Itinerary plan created"})
            st.markdown('<div class="success-box">✅ Itinerary planned</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Planning failed: {str(e)}")
            st.stop()
        
        # Step 3: Calculation
        status_text.text("💰 Agent 3: Calculation Agent - Computing budget...")
        progress_bar.progress(75)
        
        try:
            calculation_query = f"""Based on this research for {city}:
{research_findings}

And this itinerary:
{itinerary_plan}

Calculate the total budget estimate in SAR for a {days}-day tour. Break down costs by category (accommodation, food, transport, activities) and provide daily estimates."""
            calculation_result = agents['calculation_executor'].invoke({"input": calculation_query})
            budget_calculations = calculation_result["output"]
            interactions.append({"from": "Calculation Agent", "to": "Executor Agent", "message": "Budget calculated"})
            st.markdown('<div class="success-box">✅ Budget calculated</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Calculation failed: {str(e)}")
            st.stop()
        
        # Step 4: Executor
        status_text.text("🎯 Agent 4: Executor Agent - Compiling final tour plan...")
        progress_bar.progress(90)
        
        try:
            final_result = agents['executor_chain'].invoke({
                "research_findings": research_findings,
                "itinerary_plan": itinerary_plan,
                "budget_calculations": budget_calculations,
                "user_request": user_request
            })
            final_plan = final_result.content if hasattr(final_result, "content") else str(final_result)
            interactions.append({"from": "Executor Agent", "to": "User", "message": "Complete tour plan delivered"})
            progress_bar.progress(100)
            status_text.text("✅ Tour plan completed!")
        except Exception as e:
            st.error(f"❌ Final compilation failed: {str(e)}")
            st.stop()
        
        # Display Results
        st.markdown("---")
        st.header("🎉 Your Saudi Tour Plan")
        
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📋 Complete Plan", "📊 Agent Communication", "💾 Download"])
        
        with tab1:
            st.markdown(final_plan)
        
        with tab2:
            st.subheader("Agent Communication Log")
            for i, msg in enumerate(interactions, 1):
                st.markdown(f'<div class="agent-card"><strong>{i}. {msg["from"]} → {msg["to"]}</strong><br>{msg["message"]}</div>', unsafe_allow_html=True)
        
        with tab3:
            st.subheader("Download Your Plan")
            st.download_button(
                label="📥 Download as Text",
                data=final_plan,
                file_name=f"saudi_tour_plan_{city}_{days}_days.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Built with ❤️ using LangChain & OpenAI | Multi-Agent System for Saudi Tourism</p>
</div>
""", unsafe_allow_html=True)
