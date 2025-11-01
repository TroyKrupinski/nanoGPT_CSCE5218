
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

RUN_RE = re.compile(
    r"b(?P<block>\d+)_l(?P<layer>\d+)_h(?P<head>\d+)_e(?P<emb>\d+)_bs(?P<bs>\d+)_it(?P<iter>\d+)_do(?P<do>\d+)"
)

def parse_run_name(name: str) -> Optional[Dict[str, int]]:
    m = RUN_RE.fullmatch(name)
    if not m:
        return None
    d = {k:int(v) for k,v in m.groupdict().items()}
    # convert dropout like "01" -> 0.1, "02" -> 0.2
    d["dropout"] = float(d.pop("do")) / 10.0
    d["block_size"] = d.pop("block")
    d["n_layer"] = d.pop("layer")
    d["n_head"] = d.pop("head")
    d["n_embd"] = d.pop("emb")
    d["batch_size"] = d.pop("bs")
    d["max_iters"] = d.pop("iter")
    return d

def approx_param_count(n_layer:int, n_embd:int, n_head:int, vocab_size:int=65) -> int:
    """
    Very rough GPT param estimate just for relative comparison.
    Not used for reporting exact numbers.
    """
    # per-block params ~ 12 * d^2 (attn qkv, proj, mlp 4*d^2, etc.), rough
    per_block = 12 * (n_embd ** 2)
    blocks = n_layer * per_block
    # token embeddings + lm head (tied) ~ vocab * d
    emb = vocab_size * n_embd
    return blocks + emb

def read_metrics_csv(p: Path) -> Tuple[List[int], List[float], List[float]]:
    iters, train, val = [], [], []
    if not p.exists():
        return iters, train, val
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("note") == "eval":
                try:
                    iters.append(int(row["iter"]))
                    train.append(float(row["train_loss"]))
                    val.append(float(row["val_loss"]))
                except:
                    pass
    return iters, train, val

def read_summary(p: Path) -> Dict[str, str]:
    info = {}
    if not p.exists():
        return info
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            if ":" in ln:
                k,v = ln.split(":", 1)
                info[k.strip()] = v.strip()
    return info

def plot_series(x, ys: Dict[str, List[float]], xlabel:str, ylabel:str, title:str, out_png:Path):
    plt.figure()
    for label, y in ys.items():
        if len(x) == len(y) and len(x) > 0:
            plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(ys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def bar_plot(labels: List[str], vals: List[float], title: str, ylabel: str, out_png: Path, rotate=60):
    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=rotate, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Aggregate NanoGPT Group1 runs")
    ap.add_argument("--root_out", type=str, default="out", help="Root directory containing run subfolders")
    ap.add_argument("--topk", type=int, default=5, help="How many top runs (by best val) to overlay")
    args = ap.parse_args()

    root = Path(args.root_out)
    if not root.exists():
        print(f"Root not found: {root}")
        return

    agg_dir = root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    master_rows = []
    # header:
    master_cols = ["run_name","block_size","n_layer","n_head","n_embd","batch_size","max_iters","dropout",
                   "best_val","final_val","final_train","tokens_per_iter","seconds","exit_code","approx_params"]
    master_csv = agg_dir / "master_aggregate.csv"

    for sub in sorted([d for d in root.iterdir() if d.is_dir()]):
        cfg = parse_run_name(sub.name)
        if not cfg:
            continue

        metrics = sub / "metrics.csv"
        summary = sub / "summary.txt"

        iters, train_loss, val_loss = read_metrics_csv(metrics)
        info = read_summary(summary)

        best_val = min(val_loss) if val_loss else None
        final_val = val_loss[-1] if val_loss else None
        final_train = train_loss[-1] if train_loss else None
        tokens_per_iter = info.get("tokens_per_iter", "")
        seconds = info.get("duration_sec", "")
        exit_code = info.get("exit_code", "")

        approx_params = approx_param_count(cfg["n_layer"], cfg["n_embd"], cfg["n_head"])

        master_rows.append([
            sub.name,
            cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"],
            cfg["batch_size"], cfg["max_iters"], cfg["dropout"],
            best_val if best_val is not None else "",
            final_val if final_val is not None else "",
            final_train if final_train is not None else "",
            tokens_per_iter, seconds, exit_code, approx_params
        ])

    # write master CSV
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(master_cols)
        for r in master_rows:
            w.writerow(r)

    # sort for plots
    filtered = [r for r in master_rows if r[8] != ""]  # has best_val
    filtered.sort(key=lambda r: float(r[8]))  # by best_val asc

    # Plot: best val vs model size
    if filtered:
        labels = [r[0] for r in filtered]
        best_vals = [float(r[8]) for r in filtered]
        approx_params = [int(r[14]) for r in filtered]
        # scatter (standalone)
        plt.figure()
        plt.scatter(approx_params, best_vals)
        plt.xscale("log")
        plt.xlabel("approx params (log scale)")
        plt.ylabel("best val loss")
        plt.title("Best val loss vs approx model size")
        plt.tight_layout()
        plt.savefig(agg_dir / "best_val_vs_modelsize.png")
        plt.close()

        # bar chart of best val by run (sorted)
        bar_plot(labels, best_vals, "Best validation loss by run (lower is better)", "best val loss",
                 agg_dir / "best_val_by_run.png")

        # Overlay val curves for top-k runs (one chart per run, to avoid clutter)
        topk = min(args.topk, len(filtered))
        for i in range(topk):
            run_name = filtered[i][0]
            subdir = root / run_name
            iters, train_loss, val_loss = read_metrics_csv(subdir / "metrics.csv")
            if iters and val_loss:
                plot_series(
                    iters,
                    {"val": val_loss, "train": train_loss} if train_loss else {"val": val_loss},
                    "iteration",
                    "loss",
                    f"Top-{i+1}: val/train loss - {run_name}",
                    agg_dir / f"top{i+1:02d}_val_curve_{run_name}.png"
                )

    print(f"Aggregation complete.\n- Master CSV: {master_csv}\n- Plots in: {agg_dir}")

if __name__ == "__main__":
    main()
