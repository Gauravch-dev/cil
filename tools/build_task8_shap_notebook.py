"""Build the standalone SHAP Task 8 notebook from its percent-format source."""
import ast
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "task8_shap.py").read_text(encoding="utf-8")
ast.parse(source)
blocks = re.split(r"^# %%([^\n]*)\n", source, flags=re.MULTILINE)
cells = []
for i, (marker, body) in enumerate(zip(blocks[1::2], blocks[2::2])):
    kind = "markdown" if "[markdown]" in marker else "code"
    if kind == "markdown":
        body = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in body.splitlines())
    cell = {"cell_type": kind, "id": f"task8-shap-{i:02d}", "metadata": {}, "source": body.strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "task8_shap.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
(root / "task8_shap.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created task8_shap.ipynb with {len(cells)} cells.")

# Preserve the current Task 7 implementation while turning its exported markdown
# strings into notebook markdown cells. Its executable statements are unchanged.
task7_source = (root / "task7_final_peak_gated_multitask_lstm.py").read_text(encoding="utf-8")
task7_tree = ast.parse(task7_source)
task7_lines = task7_source.splitlines(keepends=True)
combined = [{"cell_type": "markdown", "id": "combined-introduction", "metadata": {}, "source": [
    "# Smart Electricity Consumption Optimizer — Tasks 7 and 8\n",
    "Run Task 7, then continue to Task 8 below in this same notebook. Task 8 reads the saved models directly; no ZIP transfer is needed.\n",
    "Task 7 training runs only when you run its cells. If those cells have already completed in this runtime, continue from the Task 8 handoff cell.\n",
    "The Task 7 implementation is preserved. Task 8 adds SHAP, activation heatmaps and model comparisons.\n"]}]

def add_cell(kind, body):
    if not body.strip():
        return
    item = {"cell_type": kind, "id": f"combined-task7-{len(combined):02d}", "metadata": {},
            "source": body.strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    combined.append(item)

previous = 0
for node in task7_tree.body:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        add_cell("code", "".join(task7_lines[previous:node.lineno - 1]))
        body = node.value.value
        if previous == 0 and "# Task 7" in body:
            body = body[body.index("# Task 7"):]
        add_cell("markdown", body)
        previous = node.end_lineno
add_cell("code", "".join(task7_lines[previous:]))

# Check that conversion preserved every executable Task 7 statement exactly.
converted = ast.parse("\n\n".join("".join(cell["source"]) for cell in combined if cell["cell_type"] == "code"))
original_code = ast.Module(body=[node for node in task7_tree.body if not (
    isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))], type_ignores=[])
assert ast.dump(converted, include_attributes=False) == ast.dump(original_code, include_attributes=False)
combined.append({"cell_type": "markdown", "id": "task8-handoff-instructions", "metadata": {}, "source": [
    "# Continue here for Task 8\n",
    "Task 7 must have completed in this runtime. The next cell passes its saved output folder directly to Task 8. No training is performed by Task 8.\n"]})
combined.append({"cell_type": "code", "id": "task8-direct-handoff", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "task7_result_folder = Path(globals().get('OUTPUT_DIR', ''))\n",
    "if not (task7_result_folder / 'manifest.json').is_file():\n",
    "    task7_result_folder = Path(globals().get('TASK7_COMPLETED_OUTPUT_DIR', ''))\n",
    "if not (task7_result_folder / 'manifest.json').is_file():\n",
    "    raise RuntimeError('Complete Task 7 above before starting Task 8.')\n",
    "TASK7_COMPLETED_OUTPUT_DIR = str(task7_result_folder.resolve())\n",
    "print('Task 8 will use Task 7 results directly:', TASK7_COMPLETED_OUTPUT_DIR)\n"]})
combined.extend(cells)
combined_notebook = dict(notebook, cells=combined, metadata=dict(notebook["metadata"]))
combined_notebook["metadata"]["colab"] = {"name": "task7_and_task8_shap.ipynb", "provenance": []}
(root / "task7_and_task8_shap.ipynb").write_text(json.dumps(combined_notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created task7_and_task8_shap.ipynb with {len(combined)} cells; Task 7 code preserved.")
