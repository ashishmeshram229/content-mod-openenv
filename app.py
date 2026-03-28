"""
FastAPI server exposing the ContentModerationEnv as an HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health

Hardened for OpenEnv validator:
- POST /reset accepts no body, empty body {}, or {"task": "easy"}
- All endpoints return 200 with valid JSON on the happy path
- Startup errors are caught and reported cleanly
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup (before any env imports so server always starts)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Lazy-load the environment so import errors don't kill the server
# ---------------------------------------------------------------------------

_env = None
_env_error: Optional[str] = None


def get_env():
    global _env, _env_error
    if _env is not None:
        return _env
    try:
        from env.environment import ContentModerationEnv
        _env = ContentModerationEnv()
        return _env
    except Exception as exc:
        _env_error = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Env init failed: {exc}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"   # "easy" | "medium" | "hard"


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Global exception handler — always return JSON, never HTML 500
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "env_id": "content-moderation-v1", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    try:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """
    Reset the environment.
    Accepts ANY body: null, no body, empty {}, or {"task": "easy"|"medium"|"hard"}
    The validator sends a null body — this handler never rejects on body format.
    """
    task = "easy"  # always-safe default
    try:
        raw = await request.body()
        if raw and len(raw.strip()) > 2:   # more than just "{}" or ""
            import json as _json
            data = _json.loads(raw.decode("utf-8", errors="ignore"))
            if isinstance(data, dict):
                t = data.get("task") or data.get("task_id") or "easy"
                if t in ("easy", "medium", "hard"):
                    task = t
    except Exception:
        pass   # any parse failure → stay with "easy"

    try:
        env = get_env()
        obs = env.reset(task=task)
        return obs.dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")


@app.post("/step")
async def step(request: Request) -> StepResponse:
    """Submit one moderation action."""
    try:
        body_bytes = await request.body()
        import json
        body = json.loads(body_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}")

    # Accept both {"action": {...}} and flat {...}
    action_data = body.get("action", body)

    try:
        from env.models import ModerationAction
        action = ModerationAction(**action_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    try:
        env = get_env()
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.dict(),
            reward=reward,
            done=done,
            info=info,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}")


@app.get("/state")
def state() -> Dict[str, Any]:
    try:
        return get_env().state()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name":    "Content Moderation OpenEnv",
        "version": "1.0.0",
        "env_id":  "content-moderation-v1",
        "tasks":   ["easy", "medium", "hard"],
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "health":"GET /health",
            "docs":  "GET /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)