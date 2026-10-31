"""Validate notebook structure, causal preprocessing, and optionally a real-data smoke run."""
import argparse
import ast
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import nbformat
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
notebook = nbformat.read(root / "task7.ipynb", as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
print("PASS: notebook schema and syntax", flush=True)

source = (root / "task7_pipeline.py").read_text(encoding="utf-8")
tree = ast.parse(source)
function_names = {"prepare_power", "calendar_features", "build_context", "valid_target_indices"}
env = {"np": np, "pd": pd}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in function_names:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "task7_functions", "exec"), env)

# Future perturbations must not change earlier completed bins or their forecasts' inputs.
timestamps = pd.date_range("2020-01-01", periods=20*24*60, freq="min")
minute = pd.Series(1 + np.arange(len(timestamps)) / len(timestamps), index=timestamps)
cut = timestamps[15*24*60]
altered = minute.copy()
altered.loc[altered.index >= cut] += 1000
power = env["prepare_power"](minute)
other = env["prepare_power"](altered)
pd.testing.assert_frame_equal(power.loc[:cut], other.loc[:cut])
context = env["build_context"](power)
context_other = env["build_context"](other)
pd.testing.assert_frame_equal(context.loc[:cut + pd.Timedelta(hours=1)],
                              context_other.loc[:cut + pd.Timedelta(hours=1)])
print("PASS: future changes cannot alter historical preprocessing or forecast inputs", flush=True)

# Missing targets stay missing. Short input gaps are bounded; long gaps are excluded.
with_gap = minute.copy()
gap_start = timestamps[10*24*60]
gap_end = gap_start + pd.Timedelta(hours=3)
with_gap.loc[(with_gap.index >= gap_start) & (with_gap.index < gap_end)] = np.nan
gapped = env["prepare_power"](with_gap)
missing_targets = gapped.loc[gap_start + pd.Timedelta(minutes=15):gap_end]
assert missing_targets["actual_kw"].isna().all()
assert missing_targets["input_kw"].iloc[:4].notna().all()
assert missing_targets["input_kw"].iloc[4:].isna().all()
eligible = env["valid_target_indices"](gapped, env["build_context"](gapped))
assert gapped["observed"].iloc[eligible].all()
for target in eligible:
    sequence = gapped["input_kw"].iloc[target-4-96+1:target-4+1]
    assert len(sequence) == 96 and sequence.notna().all()
    assert gapped.index[target] - sequence.index[-1] == pd.Timedelta(hours=1)
assert power.index[0] == timestamps[0] + pd.Timedelta(minutes=15)
assert power["actual_kw"].iloc[0] == minute.iloc[:15].mean()
print("PASS: completed-bin timestamps, horizon, missing-target exclusion and bounded filling", flush=True)

if args.smoke:
    os.chdir(root)
    runtime = {"__name__": "__main__"}
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code = cell.source.replace('RUN_MODE = "full"', 'RUN_MODE = "smoke"')
        print(f"\nEXECUTING NOTEBOOK CELL {i}", flush=True)
        exec(compile(code, f"task7.ipynb cell {i}", "exec"), runtime)
    validation_record = {"status": "passed", "scope": "full-data preprocessing; sampled two-epoch training",
                         "output_directory": str(runtime["OUTPUT_DIR"]),
                         "full_training_performed": False}
    (root / "task7_validation.json").write_text(json.dumps(validation_record, indent=2), encoding="utf-8")
    print("PASS: end-to-end smoke run, artifact export and model reload", flush=True)
