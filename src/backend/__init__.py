# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""SENTINEL web backend: FastAPI demo console (FYP-I) and the future dashboard API."""

from .app import create_app
from .runner import DemoRunner, RunConfig

__all__ = ["create_app", "DemoRunner", "RunConfig"]
