
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, RootModel
from adaptive_graph import build_adaptive_graph, build_resource_graph
from uuid import uuid4
from dotenv import load_dotenv
from typing import List, Dict
import os, uvicorn
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_openai import ChatOpenAI
from main import build_graph, FIELDS, clarify_field_with_gpt

load_dotenv()
OpenAI_KEY = ""

llm = ChatOpenAI(
    model_name='gpt-4o',
    temperature=0.5,
    openai_api_key=OpenAI_KEY,
) 

graph = build_graph(llm)
adaptive_graph = build_adaptive_graph(llm)
resource_graph = build_resource_graph(llm)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

class ClarifyInput(BaseModel):
    session_id: str = None
    message: str
    profile: dict = None


@app.post("/clarify")
async def clarify(input: ClarifyInput):
    session_id = input.session_id or str(uuid4())

    if session_id not in sessions:
        sessions[session_id] = {
            "state": {
                "profile": input.profile or {
                    "name": "Ali",
                    "user_id": "ali_01",
                    "goals": ["become a data scientist"],
                    "weekly_commitment": 10
                },
                "clarified_profile": {},
                "roadmap": None,
                "enriched_roadmap": None
            },
            "history": []
        }

    session = sessions[session_id]
    state = session["state"]
    history = session.get("history", [])

    for field in FIELDS:
        if field in state["clarified_profile"]:
            continue

        value = state["profile"].get(field, "")
        education = state["profile"].get("education", "")
        experience = state["profile"].get("experience", "")
        goals = state["profile"].get("goals", "")

        system_msg = SystemMessage(content=(
            f"You are a profile clarification assistant.\n"
            f"Field to clarify: {field}\n"
            f"User background:\n"
            f"- Education: {education}\n"
            f"- Experience: {experience}\n"
            f"- Goals: {goals}"
        ))

        if input.message:
            history.append(HumanMessage(content=input.message))

        messages = [system_msg] + history[-10:]  # limit context to last 10 exchanges
        result = clarify_field_with_gpt(llm, field, value, education, experience, goals, messages)

        history.append(AIMessage(content=result.data))
        session["history"] = history

        if result.clarity == "clear":
            state["clarified_profile"][field] = result.data
            continue
        else:
            return {
                "reply": result.data,
                "session_id": session_id
            }

    result = graph.invoke(state)
    sessions[session_id]["state"] = result


    return {
        "reply": "🎯 Your roadmap is ready!",
        "roadmap": result,
        "final_profile": result["clarified_profile"],
        "session_id": session_id,
        "enriched_profile": result
    }

class QuizRequest(BaseModel):
    step: str
    outcomes: List[str]

class QuizQuestion(BaseModel):
    question: str
    choices: List[str]

class QuizList(BaseModel):
    questions: List[QuizQuestion]

@app.post("/generate-quiz")
async def generate_quiz(input: QuizRequest):
    prompt = f"""
You are a learning assistant. Generate **exactly 10** multiple-choice questions

Learning Outcomes:
{chr(10).join(f"- {o}" for o in input.outcomes)}

For each question:
- Ask a clear question
- Provide 1 correct answer and 3 distractors
- Format as JSON like this:
{{
  "questions": [
    {{
      "question": "....",
      "choices": ["Correct", "Wrong 1", "Wrong 2", "Wrong 3"]
    }},
    ...
  ]
}}
Respond ONLY with valid JSON.
"""

    llm_quiz = llm.with_structured_output(QuizList, method="function_calling")
    result = llm_quiz.invoke(prompt)
    return {"quiz": result.questions}


class AdaptiveInput(BaseModel):
    session_id: str
    actual_duration: int  # in weeks
    current_step_index: int  # zero-based



@app.post("/update-duration")
async def update_duration(input: AdaptiveInput):
    session = sessions.get(input.session_id)
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    state = session["state"]

    if not state.get("enriched_roadmap"):
        return JSONResponse(
            content={"error": "Missing enriched_roadmap. Clarify must run first."},
            status_code=400
        )

    state["actual_duration"] = input.actual_duration
    state["current_step_index"] = input.current_step_index

    try:
        updated = adaptive_graph.invoke(state)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Adaptive graph failed: {str(e)}"},
            status_code=500
        )

    session["state"]["enriched_roadmap"] = updated["enriched_roadmap"]

    return {
        "message": "Roadmap updated.",
        "updated_roadmap": updated["enriched_roadmap"]
    }


class ResourceStyleInput(BaseModel):
    session_id: str
    current_step_index: int
    liked_style: bool

@app.post("/update-style")
async def update_style(input: ResourceStyleInput):
    session = sessions.get(input.session_id)
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    state = session["state"]
    if not state.get("enriched_roadmap"):
        return JSONResponse(content={"error": "Missing roadmap"}, status_code=400)

    state["liked_style"] = input.liked_style
    state["current_step_index"] = input.current_step_index

    try:
        updated = resource_graph.invoke(state)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": f"Style graph failed: {str(e)}"}, status_code=500)

    session["state"]["enriched_roadmap"] = updated["enriched_roadmap"]
    return {
        "message": "Resources updated.",
        "updated_roadmap": updated["enriched_roadmap"]
    }



if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )