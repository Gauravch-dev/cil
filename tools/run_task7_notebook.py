"""Execute the self-contained notebook and retain real outputs without a GUI/kernel server."""
import base64
import contextlib
import io
import os
import sys
import time
import traceback
import argparse
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib.pyplot as plt
import nbformat

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--reuse-completed", type=Path)
parser.add_argument("--notebook", default="task7.ipynb")
args = parser.parse_args()
os.chdir(root)
target = root / args.notebook
nb = nbformat.read(target, as_version=4)
env = {"__name__": "__main__"}
current_outputs = []

class Tee:
    def __init__(self, original, name):
        self.original, self.name = original, name

    def write(self, value):
        self.original.write(value)
        self.original.flush()
        if value:
            if current_outputs and current_outputs[-1].output_type == "stream" and current_outputs[-1].name == self.name:
                current_outputs[-1].text += value
            else:
                current_outputs.append(nbformat.v4.new_output("stream", name=self.name, text=value))
        return len(value)

    def flush(self):
        self.original.flush()

    def close(self):
        self.flush()

def capture_display(value):
    data = {"text/plain": str(value)}
    if hasattr(value, "_repr_html_"):
        markup = value._repr_html_()
        if markup:
            data["text/html"] = markup
    current_outputs.append(nbformat.v4.new_output("display_data", data=data, metadata={}))
    sys.__stdout__.write(str(value) + "\n")
    sys.__stdout__.flush()

def capture_figures(*args, **kwargs):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        current_outputs.append(nbformat.v4.new_output(
            "display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                                  "text/plain": "Matplotlib figure"}, metadata={}))
        plt.close(figure)

plt.show = capture_figures
execution_count = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    execution_count += 1
    current_outputs = []
    cell.execution_count = execution_count
    print(f"\n--- Executing cell {i} ({execution_count}) ---", flush=True)
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(Tee(sys.stdout, "stdout")), contextlib.redirect_stderr(Tee(sys.stderr, "stderr")):
            code = cell.source
            if args.reuse_completed and "histories, fit_records, trained_paths = {}, [], {}" in code:
                start = code.index('for kind, config in [("mlp", basic_config)]')
                end = code.index('\nvalidation_results = ', start)
                reuse = '''
for config in [basic_config, *LSTM_CONFIGS]:
    name = config["name"]
    original_path = REUSE_DIR / f"{name}.keras"
    history_frame = pd.read_csv(REUSE_DIR / f"{name}_history.csv", index_col="Epoch")
    network = keras.models.load_model(original_path)
    histories[name] = history_frame
    history_frame.to_csv(OUTPUT_DIR / f"{name}_history.csv")
    path = OUTPUT_DIR / f"{name}.keras"
    network.save(path)
    trained_paths[name] = path
    val_prediction, clipped_fraction = predict_kw(network, val_indices)
    validation_predictions[name] = val_prediction
    fit_records.append({"Model": name, "Kind": "mlp" if name == "basic_mlp" else "lstm",
                        **regression_metrics(actual_val, val_prediction),
                        "Epochs": len(history_frame),
                        "Best_epoch": int(history_frame["val_mse"].idxmin()),
                        "Parameters": int(network.count_params()),
                        "Training_seconds": None, "Clipped_fraction": clipped_fraction})
    print(f"Reused completed full-data training: {name}, {len(history_frame)} epochs, best epoch {history_frame['val_mse'].idxmin()}.")
    print("Training duration unavailable in the recovered run; original training histories preserved.")
'''
                env["REUSE_DIR"] = args.reuse_completed.resolve()
                code = code[:start] + reuse + code[end:]
            exec(compile(code, f"{target.name} cell {i}", "exec"), env)
        if "display" in env:
            env["display"] = capture_display
    except Exception as exc:
        current_outputs.append(nbformat.v4.new_output("error", ename=type(exc).__name__,
                                                     evalue=str(exc), traceback=traceback.format_exc().splitlines()))
        cell.outputs = current_outputs
        nbformat.write(nb, target)
        raise
    cell.outputs = current_outputs
    nbformat.write(nb, target)
    print(f"Cell completed in {time.perf_counter()-started:.1f}s", flush=True)
nbformat.validate(nb)
print("FULL NOTEBOOK EXECUTION COMPLETED:", env["OUTPUT_DIR"], flush=True)
