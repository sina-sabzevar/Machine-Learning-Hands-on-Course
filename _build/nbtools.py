"""Tiny helper to author course notebooks from Python and execute them.

Usage in a build script:
    from nbtools import md, code, build
    cells = [md("# Title"), code("import numpy as np")]
    build(cells, "notebooks/01_x.ipynb")   # writes + executes (outputs kept)
"""
import os, sys, textwrap
import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def md(src):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(src).strip("\n"))

def code(src):
    return nbformat.v4.new_code_cell(textwrap.dedent(src).strip("\n"))

def build(cells, rel_path, execute=True, timeout=900):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if execute:
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                                resources={"metadata": {"path": os.path.dirname(path)}})
        client.execute()
    nbformat.write(nb, path)
    print(f"wrote {rel_path} ({len(cells)} cells, executed={execute})")
