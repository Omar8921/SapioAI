from pydantic import BaseModel

from typing import Dict, List, Optional, Literal
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
# from langchain_core.language_models.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json



# consts
# 1
FIELDS = [
    "name",
    "user_id",
    "country",
    "preferred_language",
    "education",
    "experience",
    "skills_detected",
    "goals",
    "target_industry",
    "github_url",
    "linkedin_url",
    "financial_capability",
    "weekly_commitment",
    "learning_preferences"
]


LIST_FIELDS = {
    "education",
    "experience",
    "skills_detected",
    "learning_preferences",
    "goals"
}


# baseModels
# 1
class ClarifiedProfile(BaseModel):
    name: str
    user_id: str
    country: Optional[str] = None
    preferred_language: Optional[str] = "English"
    education: Optional[List[str]] = None
    experience: Optional[List[str]] = None
    skills_detected: Optional[List[str]] = None
    goals: List[str]
    target_industry: Optional[str] = None
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    financial_capability: Optional[str] = None
    weekly_commitment: int
    learning_preferences: Optional[List[str]] = None

# 2
# Define the structured output schema
class FieldClarificationResponse(BaseModel):
    clarity: Literal["clear", "unclear"]
    data: str  # either the rewritten value OR a clarification question

# 3
class ResourceItem(BaseModel):
    name: str
    url: str
    type: Literal["video", "text", "course", "book", "interactive"]
    description: Optional[str] = None

# 4
class ResourceListResponse(BaseModel):
    resources: List[ResourceItem]

# 5
class StepItem(BaseModel):
    step: str
    description: str
    learning_outcomes: List[str]
    estimated_time: int  # in weeks
    resources: Optional[ResourceListResponse]

# 6
class PlannerResponse(BaseModel):
    steps: List[StepItem]



# states
# 1
class AgentState(TypedDict):
    profile: Dict
    clarified_profile: Dict

# 2
class PlannerState(TypedDict):
    profile: ClarifiedProfile
    roadmap: Optional[List[Dict]]             # output from planner agent
    enriched_roadmap: Optional[List[Dict]]    # optional enrichment phase


# LLM calls
# 1:
def clarify_field_with_gpt(
    llm: ChatOpenAI,
    field: str,
    value: str,
    education: str,
    experience: str,
    goals: str,
    history: List[BaseMessage] = None
) -> FieldClarificationResponse:

    system_prompt = (
        "You are a profile clarification assistant. Your job is to evaluate each field in a user profile "
        "and determine whether the value is clear and complete.\n"
        "If the value is clear and usable, respond with clarity='clear' and rewrite the value if needed.\n"
        "Always reply to weekly commitment and financial capability with clarity='clear'"
        "If the value is unclear, vague, or missing, respond with clarity='unclear' and provide a question (except for GitHub URL and LinkedIn URL, if any of them is empty, just make sure to let the individual confirm he has no account then move on) "
        "if any of them is missing and the user confirms he does not have one, it is okay).\n"
        "Do not assume that the user knows about the field. Expect him to ask things about the questions you provide. For example, if you ask him 'What is your preferred programming language' "
        "and he answers that he has no prior experience in programming, then you can suggest some languages based on the goals, education, experience, and industry demands.\n"
        "Always respond using the exact format: clarity + data.\n"
        f"User Education: {education}\n"
        f"User Experience: {experience}\n"
        f"User Goals: {goals}"
        f"Here is the field: {field}. If it is weekly commitment, store it as an integer. And this value means the number of hours the individual is willing to spend weekly to learn"
    )

    user_prompt = (
        f"Field name: {field}\n"
        f"User value: \"{value}\"\n\n"
        f"Is the value clear or unclear?"
    )

    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    if history:
        messages += history[-8:]  # only add recent history to avoid context overflow

    structured_llm = llm.with_structured_output(FieldClarificationResponse)
    return structured_llm.invoke(messages)



# prompts:
# 1
def build_planner_prompt(profile: ClarifiedProfile) -> str:
    return f"""
You are a professional career planner helping learners achieve their goals through structured learning roadmaps.

Given the following user profile, generate a roadmap of EXACTLY FIVE learning steps. Each step should include:
- A title (`step`)
- An estimated time in weeks (`estimated_time`)
- A detailed explanation (`description`)
- A list of learning outcomes (`learning_outcomes`)

Be clear and realistic. The roadmap should be practical and aligned with the user's background, skills, goals, and weekly time commitment.

User Profile:
Name: {profile.name}
ID: {profile.user_id}
Country: {profile.country}
Preferred Language: {profile.preferred_language}
Education: {profile.education}
Experience: {profile.experience}
Skills Detected: {profile.skills_detected}
Goals: {profile.goals}
Target Industry: {profile.target_industry}
GitHub: {profile.github_url}
LinkedIn: {profile.linkedin_url}
Financial Capability: {profile.financial_capability}
Weekly Commitment: {profile.weekly_commitment} hours/week
Learning Preferences: {profile.learning_preferences}

Respond only with the structured list of roadmap steps.
"""


# 2
def build_resource_prompt(step_title: str,step_description: str,learning_outcomes: List[str],profile: ClarifiedProfile) -> str:
    outcomes_str = "\n".join(f"- {o}" for o in learning_outcomes)
    prefs = (
        f"Language: {profile.preferred_language or 'English'}\n"
        f"Budget: {profile.financial_capability or 'No preference'}\n"
        f"Weekly Time Available: {profile.weekly_commitment} hours\n"
        f"Learning Preferences: {', '.join(profile.learning_preferences or ['None'])}\n"
    )

    return f"""
You're a smart learning assistant helping to find the best online resources for a learner.

Your task:
Find **3 high-quality resources** tailored to the following step of a learning roadmap.

For each resource, provide:
- A short name
- A valid URL
- Type (one of: video, text, course, book, interactive)
- A short description (1-2 lines max)

Step Info:
Title: {step_title}
Description: {step_description}
Learning Outcomes:
{outcomes_str}

Learner Profile:
{prefs}

Respond ONLY with a structured list of resources.
"""


# nodes:
# 1
def clarifier(state: AgentState, llm: ChatOpenAI) -> AgentState:
    original = state.get("profile", {})
    clarified = {}

    for field in FIELDS:
        value = original.get(field, "")
        result = clarify_field_with_gpt(llm, field, value, original.get("education"), original.get("experience"), original.get("goals"))

        if result.clarity == "clear":
            clarified[field] = result.data
        else:
            clarified[field] = value 

        # Handle list fields safely
        if field in LIST_FIELDS:
            if isinstance(result.data, list):
                clarified[field] = result.data
            elif isinstance(result.data, str):
                clarified[field] = convert_to_list_with_llm(llm, field, result.data)
            else:
                clarified[field] = []
        else:
            clarified[field] = result.data
    print('==============================')
    print('clarifier:')
    print(clarified)
    print('==============================')
    return {
        "clarified_profile": clarified,
        "profile": original
    }


def convert_to_list_with_llm(llm: ChatOpenAI, field: str, value: str) -> List[str]:
    system_prompt = (
        "You're an assistant that turns user input into a list of clean strings.\n"
        "Only return a valid JSON list of strings, like: [\"Python\", \"Machine Learning\"]"
    )
    user_prompt = f"Convert this input for the field '{field}' into a list:\n{value}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    try:
        return json.loads(response.content)
    except:
        return []


# 2
def planner(state: PlannerState, llm: ChatOpenAI) -> PlannerState:
    profile = state["profile"]
    assert isinstance(profile, ClarifiedProfile)
    structured_llm = llm.with_structured_output(PlannerResponse)

    prompt = build_planner_prompt(profile)
    result = structured_llm.invoke(prompt)
    print('==============================')
    print("planner:")
    print([stip.dict() for stip in result.steps])
    print('==============================')
    return {
        "profile": profile,
        "roadmap": [step.dict() for step in result.steps],
        "enriched_roadmap": None
    }

# 3
def enrich_with_resources(state: PlannerState, llm: ChatOpenAI) -> PlannerState:
    roadmap = state.get("roadmap", [])
    profile = state["profile"]
    structured_llm = llm.with_structured_output(ResourceListResponse)
    enriched = []

    for step in roadmap:
        prompt = build_resource_prompt(
            step["step"],
            step["description"],
            step["learning_outcomes"],
            profile
        )
        result = structured_llm.invoke(prompt)
        step["resources"] = [res.dict() for res in result.resources or []]
        enriched.append(step)

    return {
        **state,
        "enriched_roadmap": enriched
    }


# building the graph
# Define the full graph state
class CPDState(TypedDict):
    profile: Dict
    clarified_profile: Dict
    roadmap: Optional[List[Dict]]
    enriched_roadmap: Optional[List[Dict]]


def build_graph(llm: ChatOpenAI):
    # Node: Clarifier
    clarifier_node = RunnableLambda(lambda state: {
        **state,
        "clarified_profile": clarifier({"profile": state["profile"]}, llm)["clarified_profile"]
    })

    # Node: Planner
    planner_node = RunnableLambda(lambda state: {
        **state,
        "roadmap": planner({
            "profile": ClarifiedProfile(**state["clarified_profile"])
        }, llm)["roadmap"]
    })

    # Node: Enricher
    enricher_node = RunnableLambda(lambda state: {
        **state,
        "enriched_roadmap": enrich_with_resources({
            "profile": ClarifiedProfile(**state["clarified_profile"]),
            "roadmap": state["roadmap"]
        }, llm)["enriched_roadmap"]
    })


    # Build the LangGraph
    graph = StateGraph(CPDState)

    graph.add_node("clarifier", clarifier_node)
    graph.add_node("planner", planner_node)
    graph.add_node("enricher", enricher_node)

    graph.set_entry_point("clarifier")
    graph.add_edge("clarifier", "planner")
    graph.add_edge("planner", "enricher")
    graph.add_edge("enricher", END)

    # Compile
    return graph.compile()