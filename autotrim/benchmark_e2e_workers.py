#!/usr/bin/env python3
"""Sweep --workers for e2e batch and report wall vs cpu-sum speedup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-id", default="RCbQVAISMcU")
    p.add_argument("--calibration-json", type=Path, required=True)
    p.add_argument("--max-clips", type=int, default=4)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--extract-fps", type=float, default=2.0)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = (args.output_root or (REPO_ROOT / "autotrim/_runs" / args.source_id / "e2e_worker_sweep")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    batch = REPO_ROOT / "autotrim/batch_source_e2e.py"
    cal = args.calibration_json.expanduser().resolve()

    rows: list[dict] = []
    print(f"sweep: {args.source_id} clips={args.max_clips} workers={args.workers}\n")

    for w in args.workers:
        run_dir = out_root / f"workers_{w}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            py,
            str(batch),
            "--source-id",
            args.source_id,
            "--calibration-json",
            str(cal),
            "--max-clips",
            str(args.max_clips),
            "--workers",
            str(w),
            "--extract-fps",
            str(args.extract_fps),
            "--output-dir",
            str(run_dir),
        ]
        if args.no_download:
            cmd.append("--no-download")

        print(f"=== workers={w} ===", flush=True)
        t0 = perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        outer_wall = round(perf_counter() - t0, 1)

        summary_path = run_dir / "summary.json"
        if proc.returncode != 0 or not summary_path.is_file():
            print(f"  FAILED rc={proc.returncode}\n", flush=True)
            rows.append({"workers": w, "status": "failed", "returncode": proc.returncode})
            continue

        s = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "workers": w,
            "status": "ok",
            "n_clips": s["n_clips"],
            "wall_sec": s.get("wall_sec", outer_wall),
            "cpu_sec_sum": s["timings_sec"]["total"],
            "extract_sum": s["timings_sec"]["extract"],
            "infer_sum": s["timings_sec"]["infer"],
            "ffmpeg_sum": s["timings_sec"]["ffmpeg"],
            "mean_clip_sec": s["mean_per_clip_sec"],
            "parallel_speedup": s.get("parallel_speedup"),
            "sec_per_clip_wall": round(s.get("wall_sec", outer_wall) / s["n_clips"], 2),
        }
        rows.append(row)
        print(
            f"  wall={row['wall_sec']:.1f}s  clip-sum={row['cpu_sec_sum']:.1f}s  "
            f"speedup={row['parallel_speedup']}x  sec/clip(wall)={row['sec_per_clip_wall']:.2f}\n",
            flush=True,
        )

    report = {
        "source_id": args.source_id,
        "max_clips": args.max_clips,
        "extract_fps": args.extract_fps,
        "runs": rows,
    }
    if rows and rows[0].get("status") == "ok":
        baseline = next((r for r in rows if r["workers"] == 1 and r["status"] == "ok"), rows[0])
        for r in rows:
            if r.get("status") == "ok" and baseline.get("wall_sec"):
                r["wall_vs_serial"] = round(baseline["wall_sec"] / r["wall_sec"], 2)

    out_path = out_root / "sweep_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"{'workers':>8} {'wall(s)':>10} {'clip-sum(s)':>12} {'speedup':>10} {'vs serial':>10}")
    print("-" * 60)
    baseline_wall = None
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['workers']:>8}  FAILED")
            continue
        if r["workers"] == 1:
            baseline_wall = r["wall_sec"]
        vs = f"{baseline_wall / r['wall_sec']:.2f}x" if baseline_wall else "-"
        print(
            f"{r['workers']:>8} {r['wall_sec']:>10.1f} {r['cpu_sec_sum']:>12.1f} "
            f"{r['parallel_speedup']:>9.2f}x {vs:>10}"
        )
    print("=" * 60)
    print(f"wrote {out_path}")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
