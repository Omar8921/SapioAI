from typing import Dict, List, TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda


class StepUpdateResponse(BaseModel):
    steps: List[Dict]

def build_adaptive_prompt(roadmap: List[Dict], current_index: int, actual_duration: int) -> str:
    steps_text = "\n\n".join([
        f"Step {i + 1}:\n"
        f"- Title: {step['step']}\n"
        f"- Description: {step['description']}\n"
        f"- Estimated Time: {step['estimated_time']} weeks"
        for i, step in enumerate(roadmap)
    ])

    return f"""
You are a smart curriculum adaptation assistant.

The learner just completed **Step {current_index + 1}** in **{actual_duration} weeks**.
Below is their current roadmap:

{steps_text}

Your tasks:
1. Update the estimated time of all the following steps that are somewhat related to this completed step (Step {current_index + 1}) based ont he performance in this step. For example, if this step was about Python fundamentals and the user finished it early with high mark, then the estimated time data analytics step is shortened.
2. Do NOT change the step title, description, or outcomes.

⚠️ Return JSON ONLY in this format:
{{
  "steps": [
    {{
      "step": "string",
      "description": "string",
      "learning_outcomes": ["string", ...],
      "estimated_time": int
    }},
    ...
  ]
}}
"""


def adaptive_duration_updater(state: Dict, llm: BaseChatModel) -> Dict:
    roadmap = state["enriched_roadmap"]         # use the enriched roadmap
    actual_duration = state["actual_duration"]
    current_index = state["current_step_index"]

    prompt = build_adaptive_prompt(roadmap, current_index, actual_duration)
    structured_llm = llm.with_structured_output(StepUpdateResponse, method="function_calling")
    result = structured_llm.invoke(prompt)
    updated_list = result.steps  # LLM-updated steps (only durations changed)

    # Merge updated durations into the original enriched roadmap
    original_enriched = state["enriched_roadmap"]
    if updated_list:  # if the LLM returned an updated roadmap
        for i, orig_step in enumerate(original_enriched):
            if i < len(updated_list):
                updated_step = updated_list[i] or {}
                if "estimated_time" in updated_step:
                    orig_step["estimated_time"] = updated_step["estimated_time"]
    # (If updated_list is empty or missing, we leave original_enriched unchanged)

    # Return new state with updated enriched_roadmap
    return {
        "profile": state["profile"],
        "clarified_profile": state["clarified_profile"],
        "roadmap": state["roadmap"],               # keep original roadmap intact
        "enriched_roadmap": original_enriched      # updated in-place with new durations
    }


def build_adaptive_graph(llm: BaseChatModel):
    class AdaptiveState(TypedDict):
        profile: Dict
        clarified_profile: Dict
        roadmap: List[Dict]
        enriched_roadmap: List[Dict]
        current_step_index: int
        actual_duration: int

    updater_node = RunnableLambda(lambda state: adaptive_duration_updater(state, llm))

    graph = StateGraph(AdaptiveState)
    graph.add_node("updater", updater_node)
    graph.set_entry_point("updater")
    graph.add_edge("updater", END)

    return graph.compile()

"""
how to use:

from adaptive_graph import build_adaptive_graph

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
adaptive_agent = build_adaptive_graph(llm)

result = adaptive_agent.invoke(initial_state


we use it after each finish of a step
it takes the same state with exra these: current_step_title, actual_duration_weeks
"""
 

class ResourceUpdateResponse(BaseModel):
    steps: List[Dict]

def build_resource_update_prompt(roadmap: List[Dict], current_index: int, liked: bool) -> str:
    current = roadmap[current_index]
    current_style = ", ".join([res["type"] for res in current.get("resources", [])])

    steps_text = "\n\n".join([
        f"Step {i + 1}:\n"
        f"- Title: {step['step']}\n"
        f"- Resources: {[res['type'] for res in step.get('resources', [])]}"
        for i, step in enumerate(roadmap)
    ])

    return f"""
You are a smart learning assistant that updates learning styles.

User just completed Step {current_index + 1}.
The learning style used was: {current_style}
The user {'liked' if liked else 'did not like'} this style.

Roadmap:
{steps_text}

Your job:
- If the user liked the style, try to keep similar styles in the next step.
- If the user didn’t like it, and the next step uses similar style(s), change it to something else.

⚠️ Respond ONLY in this exact JSON format:

{{
  "steps": [
    {{
      "step": "...",
      "description": "...",
      "learning_outcomes": ["..."],
      "estimated_time": ...,
      "resources": [
        {{
          "name": "...",
          "url": "...",
          "type": "...",
          "description": "..."
        }}
      ]
    }},
    ...
  ]
}}
"""

def resource_style_updater(state: Dict, llm: BaseChatModel) -> Dict:
    roadmap = state["enriched_roadmap"]
    liked = state["liked_style"]
    current_index = state["current_step_index"]

    prompt = build_resource_update_prompt(roadmap, current_index, liked)
    structured_llm = llm.with_structured_output(ResourceUpdateResponse, method="function_calling")
    result = structured_llm.invoke(prompt)

    updated_list = result.steps
    if updated_list:
        for i, step in enumerate(roadmap):
            if i < len(updated_list) and "resources" in updated_list[i]:
                step["resources"] = updated_list[i]["resources"]

    return {
        **state,
        "enriched_roadmap": roadmap
    }


def build_resource_graph(llm: BaseChatModel):
    class ResourceState(TypedDict):
        profile: Dict
        clarified_profile: Dict
        roadmap: List[Dict]
        enriched_roadmap: List[Dict]
        current_step_index: int
        liked_style: bool

    updater_node = RunnableLambda(lambda state: resource_style_updater(state, llm))

    graph = StateGraph(ResourceState)
    graph.add_node("updater", updater_node)
    graph.set_entry_point("updater")
    graph.add_edge("updater", END)

    return graph.compile()