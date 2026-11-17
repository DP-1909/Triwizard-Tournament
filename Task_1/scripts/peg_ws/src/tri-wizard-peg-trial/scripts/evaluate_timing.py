#!/usr/bin/env python3
"""Evaluation: success rate and timing from the autonomy logs.

    python3 scripts/evaluate_timing.py                 # all runs in logs/
    python3 scripts/evaluate_timing.py --last 3        # only the last 3 runs
    python3 scripts/evaluate_timing.py --out logs/summary.md

Reads logs/trials.csv (one row per peg) and logs/runs.csv (one row per run),
both written by the autonomy node. No ROS needed.
"""
import argparse
import csv
import os
import statistics as st


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    runs = read(os.path.join(a.log_dir, "runs.csv"))
    trials = read(os.path.join(a.log_dir, "trials.csv"))
    if a.last:
        runs = runs[-a.last:]
    ids = {r["run_id"] for r in runs}
    trials = [t for t in trials if t["run_id"] in ids]
    if not runs:
        print("no runs logged yet")
        return

    lines = ["# Evaluation summary", "", "| run | pegs | success | total time (s) | e-stop |", "|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['run_id']} | {r['pegs']} | {r['success']} | {r['total_time_s']} | {r['estop']} |")

    n = len(trials)
    ok = sum(int(t["success"]) for t in trials)
    first_try = sum(1 for t in trials if t["success"] == "1" and t["attempts"] == "1")
    times = [float(t["duration_s"]) for t in trials if t["success"] == "1"]
    full = [float(r["total_time_s"]) for r in runs if r["success"] == r["pegs"] and r["pegs"] != "0"]
    lines += ["", "## Overall", "",
              f"- Pegs attempted: {n}",
              f"- Success rate: {ok}/{n} = {100 * ok / max(n, 1):.1f} %",
              f"- First-try success (zero-failure bonus): {first_try}/{n}",
              f"- Mean time per successful peg: {st.mean(times):.1f} s" if times else "- Mean time per peg: n/a",
              f"- Best full run (all pegs placed): {min(full):.1f} s" if full else "- Best full run: n/a (no run placed every peg)",
              "", "## Per shape", "", "| shape | success | mean time (s) | mean yaw error (deg) |", "|---|---|---|---|"]
    for shape in sorted({t["shape"] for t in trials}):
        ts = [t for t in trials if t["shape"] == shape]
        good = [float(t["duration_s"]) for t in ts if t["success"] == "1"]
        yaw = [abs(float(t["yaw_error_deg"])) for t in ts]
        lines.append(f"| {shape} | {sum(int(t['success']) for t in ts)}/{len(ts)} | "
                     f"{st.mean(good):.1f} | {st.mean(yaw):.1f} |" if good else
                     f"| {shape} | 0/{len(ts)} | - | {st.mean(yaw):.1f} |")
    fails = [t for t in trials if t["success"] != "1"]
    if fails:
        lines += ["", "## Failures", ""] + [f"- {t['run_id']} {t['color']} {t['shape']}: {t['note']}" for t in fails]

    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
