"""Build the standalone Task 8 notebook from its percent-format source."""
import ast
import json
import re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
source = (root / "task8_pipeline.py").read_text(encoding="utf-8")
ast.parse(source)
blocks = re.split(r"^# %%([^\n]*)\n", source, flags=re.MULTILINE)
cells = []
for i, (marker, text) in enumerate(zip(blocks[1::2], blocks[2::2])):
    kind = "markdown" if "[markdown]" in marker else "code"
    if kind == "markdown":
        text = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in text.splitlines())
    cell = {"cell_type": kind, "id": f"task8-{i:02d}", "metadata": {}, "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
                                    "language_info": {"name": "python", "version": "3.12"},
                                    "colab": {"name": "task8.ipynb", "provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
(root / "task8.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
print("Created task8.ipynb:", len(cells), "cells")
