#!/usr/bin/env python

import argparse
import pickle
from pathlib import Path
import json
import os
from datetime import datetime
from typing import List
import importlib.util
import time

from src.dspi.core.Point import Point, Clu_Point
from src.dspi.algorithms.rangequery import (
    perform_range_queries,
    create_test_points_from_file,
)
from src.dspi.algorithms.ChooseRef import ChooseRef
from src.dspi.algorithms.CalculateIValue import CalculateIValue

import src.dspi.core.Point as _old_point_mod
import sys
sys.modules["Point"] = _old_point_mod

import src.dspi.core.Reference as _old_ref_mod
sys.modules["Reference"] = _old_ref_mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DSPI runner")

    p.add_argument(
        "--data",
        default="Skewed_4d_100_processed_data.pkl",
        help="processed dataset file under data/processed/",
    )
    p.add_argument(
        "--query_file",
        default="testpoint4d.txt",
        help="query file under data/query/",
    )
    p.add_argument(
        "--radius",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10, 0.20, 0.30],
        help="radius list",
    )
    p.add_argument("--index", default="DSPI", help="index name used in output paths")
    p.add_argument("-k", type=int, default=100, help="samples per radius (unused)")

    # build mode (two ways):
    p.add_argument("--build", action="store_true", help="run index build pipeline")
    # A) use existing cluster/pivot files
    p.add_argument("--cluster_dir", default=None, help="directory with cluster_*.txt")
    p.add_argument("--pivot_main", default=None, help="main pivot file, e.g., ref.txt or pivot.txt")
    p.add_argument("--pivot_dir", default=None, help="directory with pivot_*.txt or ref_*.txt")
    # B) generate cluster/pivot via a builder module placed next to this script
    p.add_argument("--builder", default=None, help="builder python file in the same directory (e.g., Kcenter.py, transformer.py)")
    p.add_argument("--builder_func", default="gen_eqdata", help="function in builder module to call")
    p.add_argument("--data_dir", default=None, help="raw data directory passed to builder")
    p.add_argument("--out_dir", default=None, help="output directory for builder results")

    p.add_argument("--num_clusters", type=int, default=0, help="number of clusters")
    p.add_argument("--num_ref", type=int, default=3, help="main+sub reference count (3 => 1 main + 2 sub)")
    p.add_argument("--out_name", default="graph_processed_data.pkl", help="output pkl under data/processed/")

    return p.parse_args()


def load_pivots_from_file(filename: Path) -> List[Point]:
    pivots: List[Point] = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = [s for s in line.strip().replace(" ", ",").split(',') if s]
            coords = [float(x) for x in parts]
            pivots.append(Point(coords))
    return pivots


def load_additional_pivots(num_clu: int, base_path: Path) -> List[List[Point]]:
    all_oth_pivots: List[List[Point]] = []
    for i in range(num_clu):
        fname = base_path / f"ref_{i}.txt"
        if not fname.exists():
            fname = base_path / f"pivot_{i}.txt"
        cur: List[Point] = []
        if fname.exists():
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [s for s in line.strip().replace(" ", ",").split(',') if s]
                    coords = [float(x) for x in parts]
                    cur.append(Point(coords))
        all_oth_pivots.append(cur)
    return all_oth_pivots


def read_cluster_points(path: Path) -> List[Point]:
    pts: List[Point] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.replace(" ", ",").split(',')
            try:
                floats = [float(x) for x in parts]
                coords = floats
            except ValueError:
                coords = [float(x) for x in parts[1:]]
            pts.append(Point(coords))
    return pts


def bench_mode(args: argparse.Namespace, root: Path) -> None:
    data_path = root / "data" / "processed" / args.data
    query_path = root / "data" / "query" / args.query_file

    with data_path.open("rb") as f:
        ds = pickle.load(f)

    ref_set = ds["all_refSet"]
    test_pts = create_test_points_from_file(query_path)
    dataset_name = Path(args.data).stem

    shared_models_dir = root / "runs" / dataset_name / args.index / "models"
    shared_models_dir.mkdir(parents=True, exist_ok=True)

    for r in args.radius:
        print(f"\nRadius = {r}")
        metrics = perform_range_queries(
            test_pts,
            ref_set,
            r,
            ds["all_data"],
        )

        run_dir = root / "runs" / dataset_name / args.index / f"r_{str(r).replace('.', 'p')}"
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset": args.data,
            "query_file": args.query_file,
            "radius": r,
            "index": args.index,
            "k": args.k,
            "metrics": metrics,
        }
        with (models_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        try:
            with (run_dir / "refset_snapshot.pkl").open("wb") as f:
                pickle.dump(ref_set, f)
        except Exception as e:
            with (shared_models_dir / "log.file").open("a", encoding="utf-8") as lf:
                lf.write(f"{datetime.utcnow().isoformat()}Z save snapshot failed: {e}\n")

        with (shared_models_dir / "log.file").open("a", encoding="utf-8") as lf:
            lf.write(
                f"{datetime.utcnow().isoformat()}Z bench dataset={args.data} index={args.index} query={args.query_file} "
                f"radius={r} avg_n={metrics['avg_n']:.2f} avg_time_ms={metrics['avg_time_ms']:.2f}\n"
            )


def maybe_run_builder(args: argparse.Namespace) -> Path | None:
    """Optionally run a builder module placed next to DSPI.py. Returns output dir if executed."""
    if not args.builder:
        return None
    if not args.data_dir or not args.out_dir or args.num_clusters <= 0:
        raise SystemExit("builder mode requires --builder --data_dir --out_dir --num_clusters")
    scripts_dir = Path(__file__).resolve().parent
    builder_path = scripts_dir / args.builder
    if not builder_path.exists():
        raise SystemExit(f"builder not found: {builder_path}")
    spec = importlib.util.spec_from_file_location("builder_mod", str(builder_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    fn = getattr(mod, args.builder_func, None)
    if fn is None:
        raise SystemExit(f"function '{args.builder_func}' not found in {builder_path}")
    fn(str(Path(args.data_dir)), str(Path(args.out_dir)), int(args.num_clusters))
    return Path(args.out_dir)


def build_mode(args: argparse.Namespace, root: Path) -> None:
    generated_out = maybe_run_builder(args)

    if generated_out is not None:
        cluster_dir = generated_out
        pivot_main = generated_out / "ref.txt"
        if not pivot_main.exists():
            pivot_main = generated_out / "pivot.txt"
        pivot_dir = generated_out
        dataset_name = generated_out.name
    else:
        if not args.cluster_dir or not args.pivot_main or not args.pivot_dir or args.num_clusters <= 0:
            raise SystemExit("build requires either explicit --cluster_dir --pivot_main --pivot_dir --num_clusters or a --builder with --data_dir --out_dir --num_clusters")
        cluster_dir = Path(args.cluster_dir)
        pivot_main = Path(args.pivot_main)
        pivot_dir = Path(args.pivot_dir)
        dataset_name = cluster_dir.name

    out_path = root / "data" / "processed" / args.out_name

    shared_models_dir = root / "runs" / dataset_name / args.index / "models"
    shared_models_dir.mkdir(parents=True, exist_ok=True)

    pivots = load_pivots_from_file(pivot_main)
    oth_pivots = load_additional_pivots(args.num_clusters, pivot_dir)

    all_data = []
    all_refSet = []
    build_time_ms = 0.0

    for i in range(args.num_clusters):
        cluster_file = cluster_dir / f"cluster_{i}.txt"
        if not cluster_file.exists():
            continue
        cluster_points = read_cluster_points(cluster_file)

        start = time.perf_counter()
        ref_point = ChooseRef(args.num_ref - 1, cluster_points, pivots[i], oth_pivots[i], 1)
        cal = CalculateIValue(cluster_points, ref_point.main_pointSet)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        build_time_ms += elapsed_ms

        all_data.append(cal.cluster)
        all_refSet.append(cal.mainRef_Point)

    data_to_save = {
        "all_data": all_data,
        "all_refSet": all_refSet,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(data_to_save, f)

    build_dir = root / "runs" / dataset_name / args.index / "build"
    models_dir = build_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_clusters": str(cluster_dir),
        "pivot_main": str(pivot_main),
        "pivot_dir": str(pivot_dir),
        "num_clusters": args.num_clusters,
        "num_ref": args.num_ref,
        "out_pkl": str(out_path.relative_to(root)),
        "build_time_ms_total": build_time_ms,
        "generated_by_builder": generated_out is not None,
        "builder": args.builder,
        "builder_func": args.builder_func,
    }
    with (models_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    with (shared_models_dir / "log.file").open("a", encoding="utf-8") as lf:
        lf.write(
            f"{datetime.utcnow().isoformat()}Z build dataset={dataset_name} index={args.index} "
            f"clusters={args.num_clusters} time_ms={build_time_ms:.2f} out={out_path} builder={args.builder or ''}\n"
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.build:
        build_mode(args, root)
    else:
        bench_mode(args, root)


if __name__ == "__main__":
    main()