"""Name -> technique class. Drives the CLI and the eval."""
import importlib

# (module, class) in course-lesson order; imported lazily so the torch-backed ones
# cost nothing unless asked for.
TECHNIQUES = {
    "baseline": ("baseline", "Baseline"),
    "hybrid": ("baseline", "Hybrid"),
}


def get(name: str):
    module, cls = TECHNIQUES[name]
    return getattr(importlib.import_module(f"ragfs.techniques.{module}"), cls)


def names():
    return list(TECHNIQUES)
