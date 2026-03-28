"""
FastAPI server exposing the ContentModerationEnv as an HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import ContentModerationEnv
from env.models import ModerationAction

app = FastAPI(
    title="Content Moderation OpenEnv",
    description=(
        "A real-world content moderation environment for evaluating AI agents. "
        "Implements the OpenEnv step/reset/state API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per session)
_env = ContentModerationEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"   # "easy" | "medium" | "hard"


class StepRequest(BaseModel):
    action: ModerationAction


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "env_id": ContentModerationEnv.ENV_ID}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    from env.tasks import ALL_TASKS
    return {
        k: {
            "task_id":     v["task_id"],
            "difficulty":  v["difficulty"],
            "description": v["description"],
            "num_items":   len(v["items"]),
        }
        for k, v in ALL_TASKS.items()
    }


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        obs = _env.reset(task=req.task)
        return obs.dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    try:
        obs, reward, done, info = _env.step(req.action)
        return StepResponse(
            observation=obs.dict(),
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state() -> Dict[str, Any]:
    return _env.state()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Content Moderation OpenEnv",
        "version": "1.0.0",
        "env_id": ContentModerationEnv.ENV_ID,
        "tasks": ["easy", "medium", "hard"],
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "docs":  "GET /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
