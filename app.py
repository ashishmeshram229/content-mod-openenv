"""
Content Moderation OpenEnv — FastAPI server
POST /reset  — accepts null body, no body, {} or {"task":"easy"}
POST /step   — submit a moderation action
GET  /state  — current env state
GET  /tasks  — list all tasks
GET  /health — health check
"""
from __future__ import annotations
import json
import os
import traceback
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Content Moderation OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── lazy env loader ──────────────────────────────────────────────────────────
_env = None

def get_env():
    global _env
    if _env is None:
        from env.environment import ContentModerationEnv
        _env = ContentModerationEnv()
    return _env

async def _body(request: Request) -> dict:
    """Read request body safely. Returns {} on any failure."""
    try:
        raw = await request.body()
        if raw and raw.strip() and raw.strip() != b"null":
            return json.loads(raw) or {}
    except Exception:
        pass
    return {}

# ── global error handler — always JSON, never HTML ───────────────────────────
@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    return JSONResponse({"error": str(exc)}, status_code=500)

# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "env_id": "content-moderation-v1", "version": "1.0.0"}

@app.get("/")
async def root():
    return {"name": "Content Moderation OpenEnv", "version": "1.0.0",
            "env_id": "content-moderation-v1", "tasks": ["easy", "medium", "hard"],
            "endpoints": {"reset": "POST /reset", "step": "POST /step",
                          "state": "GET /state", "tasks": "GET /tasks"}}

@app.post("/reset")
async def reset(request: Request):
    """
    Reset — tolerates null body, empty body, {} or {"task":"easy"}.
    The OpenEnv validator sends a null/empty body; this always returns 200.
    """
    body = await _body(request)
    task = body.get("task", "easy") if isinstance(body, dict) else "easy"
    if task not in ("easy", "medium", "hard"):
        task = "easy"
    try:
        obs = get_env().reset(task=task)
        return JSONResponse(obs.dict())
    except Exception as exc:
        return JSONResponse({"error": str(exc), "traceback": traceback.format_exc()},
                            status_code=500)

@app.post("/step")
async def step(request: Request):
    body = await _body(request)
    action_data = body.get("action", body)
    try:
        from env.models import ModerationAction
        action = ModerationAction(**action_data)
        obs, reward, done, info = get_env().step(action)
        return JSONResponse({"observation": obs.dict(), "reward": reward,
                             "done": done, "info": info})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

@app.get("/state")
async def state():
    try:
        return JSONResponse(get_env().state())
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@app.get("/tasks")
async def tasks():
    try:
        from env.tasks import ALL_TASKS
        return {k: {"task_id": v["task_id"], "difficulty": v["difficulty"],
                    "description": v["description"], "num_items": len(v["items"])}
                for k, v in ALL_TASKS.items()}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 7860)), reload=False)