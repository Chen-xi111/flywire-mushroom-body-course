"""Build exercise and solution notebooks from tools/lessons/*.py.

Usage (from the repository root):
    python tools/build.py
"""
import importlib.util
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(cells, which):
    nb = new_notebook()
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    for kind, ex, sol in cells:
        src = ex if which == "exercise" else sol
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    return nb


def main():
    for d in ("notebooks", "solutions"):
        (ROOT / d).mkdir(exist_ok=True)
    for path in sorted((ROOT / "tools" / "lessons").glob("[0-9]*.py")):
        mod = load(path)
        name = path.stem + ".ipynb"
        nbformat.write(make(mod.CELLS, "exercise"), ROOT / "notebooks" / name)
        nbformat.write(make(mod.CELLS, "solution"), ROOT / "solutions" / name)
        print("built", name, f"({len(mod.CELLS)} cells)")


if __name__ == "__main__":
    main()
