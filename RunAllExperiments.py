# Run all benchmark experiments and generate paper figures sequentially.
import subprocess
import sys
import time
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("VSS & EVPI", [os.path.join("experiments", "vss_and_evpi.py"), "--xlsx", "results.xlsx"]),
    ("Sensitivity & Robustness", [os.path.join("experiments", "sensitivity_analysis.py")]),
    ("Comparative Experiment", [os.path.join("experiments", "comparative_experiment.py")]),
    ("Generate Figures", ["Plot.py"]),
]


# Execute a single experiment script and log its wall-clock time.
def run(label, cmd_args):
    print(f"\n{'='*60}")
    print(f"[{label}] Starting: {' '.join(cmd_args)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, *cmd_args], cwd=ROOT)
    elapsed = time.time() - start
    status = "completed" if result.returncode == 0 else f"failed (exit code {result.returncode})"
    print(f"\n[{label}] {status} in {elapsed:.1f}s")
    return result.returncode


if __name__ == "__main__":
    overall_start = time.time()
    failed = []
    sens_script = os.path.join("experiments", "sensitivity_analysis.py")

    for label, cmd_args in SCRIPTS:
        if cmd_args[0] == "Plot.py" and sens_script in failed:
            print("\n[Generate Figures] Skipped because sensitivity_analysis.py failed.")
            failed.append("Plot.py")
            continue
        code = run(label, cmd_args)
        if code != 0:
            failed.append(cmd_args[0])

    total = time.time() - overall_start
    print(f"\n{'='*60}")
    if failed:
        print(f"Finished with failures: {', '.join(failed)}")
    else:
        print("All experiments completed successfully.")
    print(f"Total elapsed time: {total:.1f}s")
