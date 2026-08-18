import os
import getpass
import nest_asyncio
import uvicorn

# =========================
# 1. API KEY SETUP
# =========================
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Paste your Google Gemini API key: ")

# =========================
# 2. CREATE main.py WITH LANGSERVE
# =========================
main_code = """
import os
from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langserve import add_routes

# ============================================================
# GOOGLE GEMINI SETUP
# ============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing.")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

# ============================================================
# LANGGRAPH STATE
# ============================================================
class CodeReviewState(TypedDict, total=False):
    code_snippet: str
    language: str
    bug_analysis: str
    optimized_code: str

# ============================================================
# AGENT 1 - BUG & SECURITY REVIEWER
# ============================================================
def bug_reviewer(state: CodeReviewState):
    prompt = ChatPromptTemplate.from_template('''
You are an expert senior code reviewer. 
Analyze the following {language} code snippet for bugs, performance issues, and security vulnerabilities.

CODE SNIPPET:
{code_snippet}

Provide a concise breakdown of issues found.
''')
    response = (prompt | llm).invoke({
        "language": state["language"],
        "code_snippet": state["code_snippet"]
    })
    return {"bug_analysis": response.content}

# ============================================================
# AGENT 2 - CODE REFACTORER & OPTIMIZER
# ============================================================
def code_optimizer(state: CodeReviewState):
    prompt = ChatPromptTemplate.from_template('''
You are an expert software engineer. 
Based on the code review analysis provided, rewrite and optimize the code to fix all issues.

ORIGINAL CODE:
{code_snippet}

REVIEW ANALYSIS:
{bug_analysis}

Provide the clean, refactored code with brief comments explaining improvements.
''')
    response = (prompt | llm).invoke({
        "code_snippet": state["code_snippet"],
        "bug_analysis": state["bug_analysis"]
    })
    return {"optimized_code": response.content}

# ============================================================
# LANGGRAPH WORKFLOW SETUP
# ============================================================
builder = StateGraph(CodeReviewState)

builder.add_node("bug_reviewer", bug_reviewer)
builder.add_node("code_optimizer", code_optimizer)

builder.set_entry_point("bug_reviewer")
builder.add_edge("bug_reviewer", "code_optimizer")
builder.add_edge("code_optimizer", END)

# Compile graph with input/output schemas compatible with LangServe
graph = builder.compile()

# ============================================================
# FASTAPI APP & LANGSERVE ROUTES
# ============================================================
app = FastAPI(
    title="AI Code Review Assistant",
    description="A 2-agent LangGraph system with LangServe playground.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"status": "running", "project": "AI Code Review Assistant"}

# This adds the LangServe playground route at /agent/playground
add_routes(
    app,
    graph,
    path="/agent"
)
"""

with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_code)

# =========================
# 3. RUN SERVER
# =========================
from main import app

print("Starting FastAPI server with LangServe...")
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
