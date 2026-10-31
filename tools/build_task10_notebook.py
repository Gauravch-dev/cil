"""Build a standalone Task 10 evaluation notebook from reproducible sources."""
import ast
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
text = (root / "task10_pipeline.py").read_text(encoding="utf-8")
task9 = (root / "task9_core.py").read_text(encoding="utf-8")
task10 = (root / "task10_core.py").read_text(encoding="utf-8")
visuals = (root / "task10_visuals.py").read_text(encoding="utf-8")
dashboard = (root / "task9_dashboard.py").read_text(encoding="utf-8")
task7 = (root / "task7_final_peak_gated_multitask_lstm.py").read_bytes().decode("utf-8")
ontology = (root / "SmartElectricityOptimizer.owl").read_bytes().decode("utf-8")
task9_workflow = (root / "task9_pipeline.py").read_text(encoding="utf-8")
prepare_node = next(n for n in ast.parse(task9_workflow).body if isinstance(n, ast.FunctionDef) and n.name == "prepare_task7_results")
prepare = ast.get_source_segment(task9_workflow, prepare_node)

def embedded(name, content):
    encoded = base64.b64encode(zlib.compress(content.encode("utf-8"), 9)).decode("ascii")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f'{name} = zlib.decompress(base64.b64decode({encoded!r})).decode("utf-8")\nassert hashlib.sha256({name}.encode("utf-8")).hexdigest() == {digest!r}\n'

preprocessor_names = ["prepare_multivariate", "calendar_features", "build_context", "valid_target_indices"]
preprocessors = "\n\n".join(ast.get_source_segment(task7, n) for n in ast.parse(task7).body
                            if isinstance(n, ast.FunctionDef) and n.name in preprocessor_names)
features = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
resources = "import base64, hashlib, zlib\n" + embedded("ONTOLOGY_XML", ontology) + embedded("TASK7_SOURCE_TEXT", task7)
resources += embedded("TASK7_PREPROCESSOR_SOURCE", preprocessors)
task10 += f'\n\nTASK7_PREPROCESSORS = {{"np": np, "pd": pd, "ELECTRICAL_FEATURES": {features!r}, "TARGET_FEATURE": "Global_active_power", "HISTORY_INPUT_COLUMNS": {["input_"+n for n in features]!r}}}\n'
task10 += 'exec(compile(TASK7_PREPROCESSOR_SOURCE, "frozen_task7_preprocessors.py", "exec"), TASK7_PREPROCESSORS)\n'
for marker, body in {"# EMBEDDED_RESOURCES": resources, "# TASK9_DEFINITIONS": task9,
    "# TASK9_PREPARATION_FUNCTION": prepare, "# TASK10_DEFINITIONS": task10,
    "# VISUAL_DEFINITIONS": visuals, "# DASHBOARD_DEFINITIONS": dashboard}.items():
    text = text.replace(marker, body)
ast.parse(text)
parts = re.split(r"^# %%([^\n]*)\n", text, flags=re.MULTILINE)
cells = []
for i, (marker, body) in enumerate(zip(parts[1::2], parts[2::2])):
    kind = "markdown" if "[markdown]" in marker else "code"
    if kind == "markdown":
        body = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in body.splitlines())
    item = {"cell_type": kind, "id": f"task10-{i:02d}", "metadata": {}, "source": body.strip().splitlines(keepends=True)}
    if kind == "code":
        ast.parse(body)
        item.update(execution_count=None, outputs=[])
        if any(token in body for token in ["base64.b64decode", "class OntologyKnowledge", "def freeze_fingerprint", "def save_eval_figure", "def render_optimizer_result"]):
            item["metadata"] = {"jupyter": {"source_hidden": True}}
    cells.append(item)
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "task10_final_evaluation.ipynb", "provenance": []}}}
(root / "task10_final_evaluation.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
print(f"Created task10_final_evaluation.ipynb: {len(cells)} cells; all required system code/resources embedded.")
