"""Build the self-contained notebook from the reviewable percent-format source."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "task7_pipeline.py").read_text(encoding="utf-8")
blocks = re.split(r"^# %%([^\n]*)\n", source, flags=re.MULTILINE)
cells = []
for number, (marker, text) in enumerate(zip(blocks[1::2], blocks[2::2])):
    cell_type = "markdown" if "[markdown]" in marker else "code"
    if cell_type == "markdown":
        text = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line
                         for line in text.splitlines())
    cell = {"cell_type": cell_type, "id": f"task7-{number:02d}", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}
    # Ensure separate lines stay separate when the file has no trailing newline.
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "task7.ipynb", "provenance": []},
}, "nbformat": 4, "nbformat_minor": 5}
target = root / "task7.ipynb"
backup = root / "task7_original.ipynb"
if target.exists() and not backup.exists():
    backup.write_bytes(target.read_bytes())
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created {target.name}: {len(cells)} cells. Original backup: {backup.name}")
