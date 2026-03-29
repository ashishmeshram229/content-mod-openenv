"""
server/app.py — Required by openenv validate for multi-mode deployment.
Re-exports the main FastAPI app from root app.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

__all__ = ["app"]