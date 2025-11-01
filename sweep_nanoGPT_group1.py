
import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

PY_EXE = sys.executable
TRAIN_SCRIPT = "train.py"
DATASET = "shakespeare_char"
EVAL_INTERVAL = 200
USE_WANDB = False
COMPILE_FLAG = False
DRY_RUN = False

RE_EVAL = re.compile(r"^step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")
RE_ITER = re.compile(r"^iter\s+(\d+):\s+loss\s+([0-9.]+),\s+time\s+([0-9.]+)ms,\s+mfu\s+([0-9.]+)%")
RE_TOKENS = re.compile(r"^tokens per iteration will be:\s+([0-9,]+)")

@dataclass
class RunConfig:
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    batch_size: int = 16
    max_iters: int = 2000
    dropout: float = 0.2

    def run_name(self) -> str:
        do = str(self.dropout).replace('.', '')
        if do.endswith('0'):
            do = do[:-1]
        return f"b{self.block_size}_l{self.n_layer}_h{self.n_head}_e{self.n_embd}_bs{self.batch_size}_it{self.max_iters}_do{do}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_csv_row(path: Path, header: List[str], row: List):
    path_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not path_exists:
            w.writerow(header)
        w.writerow(row)

def plot_series(x, ys: Dict[str, List[float]], xlabel: str, ylabel: str, title: str, out_png: Path):
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

def parse_args():
    ap = argparse.ArgumentParser(description="NanoGPT Group-1 sweep harness (block_size=64, n_layer=4)")
    ap.add_argument("--train_py", type=str, default=TRAIN_SCRIPT, help="Path to train.py")
    ap.add_argument("--root_out", type=str, default="out", help="Root output directory")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()

def build_grid() -> List[RunConfig]:
    configs: List[RunConfig] = []
    block_size = 64
    n_layer = 4
    for n_head in (4, 8):
        for n_embd in (128, 256):
            if n_embd % n_head != 0:
                continue
            for batch_size in (8, 16):
                for max_iters in (1000, 2000):
                    for dropout in (0.1, 0.2):
                        configs.append(RunConfig(block_size, n_layer, n_head, n_embd, batch_size, max_iters, dropout))
    assert len(configs) == 32, f"Expected 32 combos, got {len(configs)}"
    return configs

def build_equals_args(kv: Dict[str, str]) -> List[str]:
    """Return ['--k=v', '--a=b', ...] so configurator.py can parse them."""
    args = []
    for k, v in kv.items():
        args.append(f"--{k}={v}")
    return args

def train_one(cfg: RunConfig, train_py: str, root_out: Path, dry_run: bool=False) -> Tuple[int, Optional[float]]:
    run_dir = root_out / cfg.run_name()
    ensure_dir(run_dir)

    kv = {
        "dataset": DATASET,
        "block_size": str(cfg.block_size),
        "n_layer": str(cfg.n_layer),
        "n_head": str(cfg.n_head),
        "n_embd": str(cfg.n_embd),
        "batch_size": str(cfg.batch_size),
        "max_iters": str(cfg.max_iters),
        "dropout": str(cfg.dropout),
        "eval_interval": str(EVAL_INTERVAL),
        "compile": "False" if not COMPILE_FLAG else "True",
        "out_dir": str(run_dir),
        "wandb_log": "False" if not USE_WANDB else "True",
    }

    cmd = [PY_EXE, train_py] + build_equals_args(kv)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n=== Launching: {cfg.run_name()} ===\n{cmd_str}\n")

    if dry_run:
        return 0, None

    stdout_log = run_dir / "train_stdout.log"
    metrics_csv = run_dir / "metrics.csv"
    best_val = None
    token_per_iter = None
    t_start = time.time()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    with stdout_log.open("w", encoding="utf-8") as logf:
        header = ["iter","train_loss","val_loss","train_iter_loss","time_ms","mfu_pct","note"]
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.01)
                continue
            logf.write(line)
            logf.flush()
            s = line.strip()

            m = RE_TOKENS.match(s)
            if m:
                token_per_iter = int(m.group(1).replace(",", ""))
            m = RE_EVAL.match(s)
            if m:
                it = int(m.group(1)); tr = float(m.group(2)); va = float(m.group(3))
                write_csv_row(metrics_csv, header, [it, tr, va, "", "", "", "eval"])
                if (best_val is None) or (va < best_val):
                    best_val = va
            m = RE_ITER.match(s)
            if m:
                it = int(m.group(1)); tr_it = float(m.group(2)); t_ms = float(m.group(3)); mfu = float(m.group(4))
                write_csv_row(metrics_csv, header, [it, "", "", tr_it, t_ms, mfu, "iter"])

    exit_code = proc.wait()
    dur_s = time.time() - t_start

    (run_dir / "summary.txt").write_text(
        f"run: {cfg.run_name()}\n"
        f"tokens_per_iter: {token_per_iter}\n"
        f"best_val_loss: {best_val}\n"
        f"duration_sec: {dur_s:.2f}\n"
        f"exit_code: {exit_code}\n",
        encoding="utf-8"
    )

    # Per-run plots
    if metrics_csv.exists():
        import csv as _csv
        iters_eval, train_losses, val_losses = [], [], []
        iters_iter, tr_iter_losses = [], []
        times_ms, mfu_vals = [], []
        with metrics_csv.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row["note"] == "eval":
                    iters_eval.append(int(row["iter"]))
                    train_losses.append(float(row["train_loss"]))
                    val_losses.append(float(row["val_loss"]))
                elif row["note"] == "iter":
                    iters_iter.append(int(row["iter"]))
                    tr_iter_losses.append(float(row["train_iter_loss"]))
                    times_ms.append(float(row["time_ms"]))
                    mfu_vals.append(float(row["mfu_pct"]))

        def plot_series(x, ys: Dict[str, List[float]], xlabel, ylabel, title, out_png: Path):
            plt.figure()
            for label, y in ys.items():
                if len(x) == len(y) and len(x) > 0:
                    plt.plot(x, y, label=label)
            plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
            if len(ys) > 1: plt.legend()
            plt.tight_layout(); plt.savefig(out_png); plt.close()

        if len(iters_eval) > 0:
            plot_series(iters_eval, {"train": train_losses, "val": val_losses},
                        "iteration", "loss",
                        f"Loss curves - {cfg.run_name()}",
                        run_dir / "loss_curves.png")
        if len(iters_iter) > 0:
            plot_series(iters_iter, {"train_iter_loss": tr_iter_losses},
                        "iteration", "train loss (per-iter)",
                        f"Per-iter train loss - {cfg.run_name()}",
                        run_dir / "train_iter_loss.png")
            plot_series(iters_iter, {"time_ms": times_ms},
                        "iteration", "time (ms)",
                        f"Step time - {cfg.run_name()}",
                        run_dir / "step_time.png")
            plot_series(iters_iter, {"mfu_%": mfu_vals},
                        "iteration", "MFU (%)",
                        f"Model FLOP Utilization - {cfg.run_name()}",
                        run_dir / "mfu.png")

    return exit_code, best_val

def main():
    args = parse_args()
    root_out = Path(args.root_out).resolve()
    ensure_dir(root_out)
    grid = build_grid()

    master_csv = root_out / "master_summary_group1.csv"
    write_csv_row(master_csv,
                  ["run_name","block_size","n_layer","n_head","n_embd","batch_size","max_iters","dropout","best_val_loss","exit_code","seconds"],
                  ["_header_written_only_once_"])

    t0 = time.time()
    for i, cfg in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {cfg.run_name()}")
        exit_code, best_val = train_one(cfg, args.train_py, root_out, dry_run=(args.dry_run or DRY_RUN))
        dur = time.time() - t0
        write_csv_row(master_csv,
                      ["run_name","block_size","n_layer","n_head","n_embd","batch_size","max_iters","dropout","best_val_loss","exit_code","seconds"],
                      [cfg.run_name(), cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.batch_size, cfg.max_iters, cfg.dropout, best_val if best_val is not None else "", exit_code, f"{dur:.2f}"])

    print("Sweep complete.")

if __name__ == "__main__":
    main()
