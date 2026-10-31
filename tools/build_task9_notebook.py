"""Create the standalone Task 9 notebook with embedded project resources."""
import ast
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
owl_path = root / "SmartElectricityOptimizer.owl"
if not owl_path.is_file():
    owl_path.write_bytes((root.parent / "SmartElectricityOptimizer.owl").read_bytes())
ontology = owl_path.read_bytes().decode("utf-8")
task7 = (root / "task7_final_peak_gated_multitask_lstm.py").read_bytes().decode("utf-8")
core = (root / "task9_core.py").read_text(encoding="utf-8")
dashboard = (root / "task9_dashboard.py").read_text(encoding="utf-8")
pipeline = (root / "task9_pipeline.py").read_text(encoding="utf-8")
for text in [core, dashboard, pipeline, task7]:
    ast.parse(text)

def embedded(name, text):
    data = base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f'{name} = zlib.decompress(base64.b64decode({data!r})).decode("utf-8")\nassert hashlib.sha256({name}.encode("utf-8")).hexdigest() == {digest!r}\n'

resources = "import base64, hashlib, zlib\n" + embedded("ONTOLOGY_XML", ontology) + embedded("TASK7_SOURCE_TEXT", task7)
split = core.index("class ForecastBridge:")
pipeline = pipeline.replace("# EMBEDDED_RESOURCES", resources)
pipeline = pipeline.replace("# CORE_DEFINITIONS", core[:split])
pipeline = pipeline.replace("# FORECAST_DEFINITIONS", core[split:])
pipeline = pipeline.replace("# DASHBOARD_DEFINITIONS", dashboard)
ast.parse(pipeline)
parts = re.split(r"^# %%([^\n]*)\n", pipeline, flags=re.MULTILINE)
cells = []
for i, (marker, body) in enumerate(zip(parts[1::2], parts[2::2])):
    kind = "markdown" if "[markdown]" in marker else "code"
    if kind == "markdown":
        body = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in body.splitlines())
    cell = {"cell_type": kind, "id": f"task9-{i:02d}", "metadata": {}, "source": body.strip().splitlines(keepends=True)}
    if kind == "code":
        ast.parse(body)
        cell.update(execution_count=None, outputs=[])
        if any(name in body for name in ["base64.b64decode", "class OntologyKnowledge", "class ForecastBridge", "def render_optimizer_result"]):
            cell["metadata"] = {"jupyter": {"source_hidden": True}}
    cells.append(cell)
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "task9_integrated_optimizer.ipynb", "provenance": []}}}
(root / "task9_integrated_optimizer.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created task9_integrated_optimizer.ipynb: {len(cells)} cells; ontology and Task 7 source embedded.")
