"""
FastAPI Routers Package

Exposes all API endpoint routers.
"""

from cerebrum.routers.agents import router as agents
from cerebrum.routers.auth import router as auth
from cerebrum.routers.datasets import router as datasets
from cerebrum.routers.health import router as health
from cerebrum.routers.ml import router as ml
from cerebrum.routers.projects import router as projects
from cerebrum.routers.reports import router as reports
from cerebrum.routers.tasks import router as tasks
from cerebrum.routers.users import router as users
from cerebrum.routers.visualizations import router as visualizations

__all__ = [
    "health",
    "auth",
    "users",
    "projects",
    "datasets",
    "agents",
    "tasks",
    "ml",
    "visualizations",
    "reports",
]
