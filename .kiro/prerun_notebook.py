#!/usr/bin/env python3
"""Pre-execute a Jupyter notebook and embed cell outputs so it opens with results
already visible (no "Run All" needed).

Dependency-light on purpose: stdlib + matplotlib + numpy (the packages confirmed
present in this environment). It runs each code cell in a shared namespace and captures:
  - stdout            -> stream output
  - matplotlib figures -> image/png display_data (base64)
  - a trailing bare expression -> its repr (like a Jupyter execute_result)
  - exceptions        -> error output (and it keeps going)

Scope note: good for numpy/matplotlib teaching notebooks. It does NOT run a real
Jupyter kernel, so rich HTML reprs (e.g. pandas tables) won't render unless printed.

Usage:
    python3 .kiro/prerun_notebook.py path/to/notebook.ipynb
"""
import sys
import io
import ast
import json
import base64
import contextlib
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _exec_cell(src, ns):
    """Exec a cell; if its last statement is a bare expression, eval it and print
    its repr so it shows up as an output (mimicking Jupyter's execute_result)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        exec(compile(src, "<cell>", "exec"), ns)
        return
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        exec(compile(tree, "<cell>", "exec"), ns)
        val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
        if val is not None:
            print(repr(val))
    else:
        exec(compile(tree, "<cell>", "exec"), ns)


def run(path):
    with open(path) as f:
        nb = json.load(f)

    ns = {}
    count = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        count += 1
        src = "".join(cell.get("source", []))
        cell["execution_count"] = count
        outputs = []

        plt.close("all")
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                _exec_cell(src, ns)
        except Exception as e:  # keep going, record the error like Jupyter does
            outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": traceback.format_exc().splitlines(),
            })

        text = stdout.getvalue()
        if text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": text.splitlines(keepends=True),
            })

        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", bbox_inches="tight", dpi=110)
            outputs.append({
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(buf.getvalue()).decode("ascii")},
                "metadata": {},
            })
        plt.close("all")

        cell["outputs"] = outputs

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"executed {count} code cells -> {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 .kiro/prerun_notebook.py <notebook.ipynb>")
    run(sys.argv[1])
