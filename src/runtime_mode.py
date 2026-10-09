"""Dependency-free backend capability checks; never import local ML libraries."""
import importlib.util
import platform

import userconfig


def intel_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in ("x86_64", "amd64", "i386")


def local_available() -> bool:
    return not intel_mac() and all(importlib.util.find_spec(name) is not None
                                   for name in ("torch", "transformers"))


def api_only() -> bool:
    return (intel_mac() or userconfig.get("JUDGE_BACKEND").strip().lower() in ("api", "cloud")
            or not local_available())
