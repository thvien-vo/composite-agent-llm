"""
Fixed-thickness "stiffest laminate" (objective max_stiffness), step 2: the 16 problems (8 load cases x 2 ply
thicknesses) at the default ply count.

Writes deliverables/optimization/stiffness_step2/ only (results/, stiffness_step1/ and figures_v2/ are read only):
  default_ply_count.csv      N of the 16 problems: QI rule ("QI_ks"), else the fewest plies of the objective min_plies
                             ("min_plies fallback", from results/cases_summary.csv, checked)
  verify_tiebreak.csv        compliance tie-break vs enumeration of every valid design (C1, C2, C8; 16 and 24 plies)
  results.csv / results.json one row per problem: max_stiffness (exact), max_margin, QI with the same N, old fewest-plies
                             design, MATLAB check
  topk.csv                   the k best designs of the 8x8 model with their 16x16 value (buckling problems)
  ga_runs.csv, ga_summary.csv, ga_history.json   GA and GA + local search, 20 seeds, best-so-far histories
  tradeoff_C6_N24_t0125.npz, tradeoff_C6_N24_t0183.npz   (lambda 8x8, strength margin) of every valid 24-ply design
  matlab_check.json, regression_demo.json, summary.md

Model: IM7/8552, simply supported 0.5 m x 0.5 m plate, Ritz double sine series 8 x 8 terms in the optimisation (proofs
are "proven in the 8x8 Ritz model"), reported buckling factor = 16 x 16 terms (with the polynomial Ritz p = 16 and the
8 x 8 value next to it), shear checked for +Nxy and -Nxy, safety factor 1.5 on the strength margin and on lambda.

Run (from python/):  ..\\venv\\Scripts\\python.exe -m optimization.experiments.stiffness_step2 [--workers 2]
                     [--problem-workers 1] [--seeds 20] [--budget 10000] [--skip-matlab] [--reuse]
                     [--stage-free-gb 4] [--min-free-gb 2]

Memory: stages run one after the other, each with its own worker pool that is closed at the end of the stage:
problems (--problem-workers, default 1: the large C5 problem runs alone), GA and GA + local search (--workers, at most
2), the C6 enumeration (--workers), then MATLAB and the demo regression in the main process with no worker alive.
Before a stage starts, the free physical memory must be at least --stage-free-gb (else the script waits up to
--wait-min minutes, then stops). Every finished task is appended to <stage>.partial.jsonl, so --reuse continues where
a run stopped. Before each task the free memory is checked: below --min-free-gb no new task is started, the running
ones finish and are saved, and the script stops (exit code 3); below half of it the workers are terminated at once.
Restart with --reuse. Every stage start is logged in run_log.jsonl (free memory, workers, tasks already saved).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from optimization import stiffness as st
from optimization.api import optimize
from optimization.evaluate import Problem, analyze_design
from optimization.experiments import stiffness_step1 as s1
from optimization.experiments.cases import CASES, SAFETY_FACTOR, THICKNESSES
from optimization.experiments.common import ROOT, _default
from optimization.ga import GAConfig, GeneticAlgorithm
from optimization.rules import full_laminate, notation

OUT = os.path.join(ROOT, "deliverables", "optimization", "stiffness_step2")
RESULTS = os.path.join(ROOT, "deliverables", "optimization", "results")
STEP1 = os.path.join(ROOT, "deliverables", "optimization", "stiffness_step1")
WORKERS = 2
MAX_WORKERS = 2            # the machine has run out of memory with more
PROBLEM_WORKERS = 1        # the exact searches (the C5 problem with 56 plies is the largest) run one at a time
EXACT_TIME = 3600.0        # max_stiffness, per problem (all k of the top-k rule)
MM_TIME = 600.0            # max_margin, per problem
CASE = {c["id"]: c for c in CASES}
TRADEOFF = ("C6", 24)      # figure 4: every valid design of C6 with 24 plies
TRADEOFF_PARTS = 8
VERIFY = [(c, tk, n) for n in (16, 24) for c in ("C1", "C2", "C8") for tk in THICKNESSES]
problem = s1.problem       # (case, thickness, N) -> Problem (8 x 8 Ritz terms, both shear signs)


# ------------------------------------------------------------------------------------------------- tasks
def task_verify(args) -> Dict:
    """Compliance tie-break: exact search vs streaming enumeration of every valid design."""
    cid, tk, n = args
    p = problem(cid, tk, n)
    t = time.perf_counter()
    ex = st.exact_stiffness(p)
    te = time.perf_counter() - t
    t = time.perf_counter()
    bf = st.brute_force_merge(p, [st.brute_force(p)])
    tb = time.perf_counter() - t
    rel = abs(ex["ddir"] - bf["ddir"]) / abs(bf["ddir"])
    return {"case": cid, "thickness": tk, "t_mm": THICKNESSES[tk] * 1e3, "N": n, "kind": ex["kind"],
            "direction": ex["direction"], "n_valid_designs": bf["n_valid"], "n_meeting_strength": bf["n_strength_ok"],
            "exact_status": ex["status"], "brute_status": bf["status"], "exact_design": ex["notation"],
            "brute_design": bf["notation"], "same_design": ex["half"] == bf["half"],
            "exact_c_N_per_m": ex["c"], "brute_c_N_per_m": bf["c"], "exact_strength_margin": ex["rf_strength"],
            "brute_strength_margin": bf["rf_strength"], "exact_ddir_Nm": ex["ddir"], "brute_ddir_Nm": bf["ddir"],
            "rel_diff_ddir": rel, "n_designs_tied_on_c_and_margin": bf["n_tied_on_c_rs"],
            "n_arrangements_tied_tuples_exact": ex.get("n_arrangements_tied_tuples"),
            "max_rel_spread_c_within_counts": bf["max_rel_spread_c_in_counts"],
            "max_rel_spread_margin_within_counts": bf["max_rel_spread_rs_in_counts"],
            "match": bool(ex["half"] == bf["half"] and ex["status"] == bf["status"] and rel <= 1e-12
                          and abs(ex["c"] - bf["c"]) <= 1e-12 * abs(bf["c"])),
            "exact_time_s": te, "brute_time_s": tb}


def task_problem(args) -> Dict:
    """max_stiffness (exact search + top-k rule), the record with the 16x16 / polynomial / 8x8 values and the QI
    laminate, and max_margin at the same N."""
    cid, tk, n = args
    p = problem(cid, tk, n)
    t = time.perf_counter()
    r = st.solve_max_stiffness(p, time_limit=EXACT_TIME)
    t_ex = time.perf_counter() - t
    h8 = np.array(r["half_8x8"], dtype=np.int8)
    rec = st.stiffness_record(
        p, r["half"], status=r["status"], proven=r["proven"], runtime_s=t_ex, gap=r.get("gap"),
        upper_bound=r.get("upper_bound"), n_eigen=r.get("n_eigen"), n_nodes=r.get("n_nodes"), method=r["method"],
        max_rf_strength=r["max_rf_strength"], infeasible_reason=r["infeasible_reason"], k_final=r.get("k_final"),
        k_history=r.get("k_history"), k_rule_met=r.get("k_rule_met"), k_cap_reached=r.get("k_cap_reached"),
        k_note=r.get("k_note"), best_differs=r.get("best_differs"), half_8x8=r.get("half_8x8"),
        notation_8x8=notation(full_laminate(h8[None], True)[0], True), lambda_8x8_best=r.get("lam_8x8_best"),
        rank_8x8_of_selected=r.get("rank_8x8_of_selected"), proven_16x16=r.get("proven_16x16"),
        status_label=st.status_text(r["status"], r["proven"], r.get("gap"), r["kind"]),
        n_count_tuples=r.get("n_count_tuples"), n_tuples_strength_ok=r.get("n_tuples_strength_ok"),
        n_arrangements_tied_tuples=r.get("n_arrangements_tied_tuples"))
    rec["lambda_analyze_design"] = analyze_design(p, r["full"])["lambda"]      # the same number as rec["lambda"]
    t = time.perf_counter()
    mm = optimize(None, problem=p, objective="max_margin", method="exhaustive", time_limit=MM_TIME)
    mm_rec = dict(mm.stiffness, runtime_s=time.perf_counter() - t, status=mm.status)
    return {"case": cid, "thickness": tk, "N": n, "max_stiffness": rec, "max_margin": mm_rec,
            "top_table": r.get("top_table"), "kind": r["kind"]}


def _same_values(a: st.Key, b: st.Key) -> bool:
    """Same feasibility, tier and objective / strength margin / bending stiffness values (within 1e-9)."""
    return (a.feasible == b.feasible and a.tier == b.tier and st._close(a.primary, b.primary, st.RTOL)
            and st._close(a.secondary, b.secondary, st.RTOL) and st._close(a.tertiary, b.tertiary, st.RTOL))


def task_ga(args) -> Dict:
    """One GA or GA + local search run (8x8 model) and its best-so-far history (change points only)."""
    cid, tk, n, method, seed, budget, target, opt_half = args
    p = problem(cid, tk, n)
    cfg = GAConfig(budget=budget, local_search=(method == "ga_ls"), objective="max_stiffness")
    tkey = st.Key(bool(target[0]), int(target[1]), float(target[2]), float(target[3]), tuple(target[4]),
                  float(target[5]))
    r = GeneticAlgorithm(p, cfg, seed).run(target=tkey)
    half = [int(x) for x in r["best_genes"] if x >= 0]
    key = st.make_key(r["kind"], p.safety_factor, r["rf_strength"], r["lam"], r["c"], half, r["ddir"])
    hist, prev = [], None
    for h in r["history"]:
        if h["half"] != prev:
            hist.append(h)
            prev = h["half"]
    if r["history"] and hist[-1] is not r["history"][-1]:
        hist.append(r["history"][-1])
    lam16 = (st.report_lambdas(p, np.array([h["half"] for h in hist], dtype=np.int8)).tolist()
             if r["kind"] == "lambda" and hist else [None] * len(hist))
    return {"case": cid, "thickness": tk, "N": n, "method": method, "seed": seed, "kind": r["kind"],
            "lambda_8x8": r["lam"], "lambda_16x16": float(st.report_lambdas(p, np.array([half], np.int8))[0]),
            "c_N_per_m": r["c"], "strength_margin": r["rf_strength"], "ddir_Nm": r["ddir"],
            "feasible": bool(key.feasible), "design": notation(full_laminate(np.array([half], np.int8), True)[0], True),
            "reaches_optimum": st.reaches(key, tkey), "same_values_as_exact": _same_values(key, tkey),
            "same_design_as_exact": half == opt_half, "hit_evals": r["hit_evals"],
            "n_evaluations": r["n_evaluations"], "runtime_s": r["time_s"],
            "history": {"evals": [h["evals"] for h in hist], "objective_8x8": [h["objective"] for h in hist],
                        "tier": [h["tier"] for h in hist], "feasible": [bool(h["feasible"]) for h in hist],
                        "lambda_16x16": lam16, "final_evals": r["n_evaluations"]}}


def task_tradeoff(args) -> Dict:
    """One part of the enumeration of every valid design (lambda 8x8 of EVERY design and strength margin), saved
    to a .npy part file (merged into the .npz by the main process)."""
    cid, tk, n, part, k = args
    p = problem(cid, tk, n)
    t = time.perf_counter()
    bf = st.brute_force(p, part=part, n_parts=k, keep_all=True)
    os.makedirs(os.path.join(OUT, "_parts"), exist_ok=True)
    path = os.path.join(OUT, "_parts", f"tradeoff_{cid}_N{n}_{tk}_part{part}.npy")
    np.save(path, np.stack([bf["all_lam"], bf["all_rs"]]))
    return {"path": path, "n_valid": bf["n_valid"], "n_strength_ok": bf["n_strength_ok"],
            "max_rf_strength": bf["max_rf_strength"], "time_s": time.perf_counter() - t}


# ------------------------------------------------------------------------------------- resumable tasks, output
def run_tasks(workers, fn, tasks, name, min_free, every=0):
    """stiffness_step1.run_tasks with the partial file in OUT of this step, in a worker pool of its own that is
    closed when the stage ends (no worker process outlives its stage)."""
    old = s1.OUT
    s1.OUT = OUT
    pool = ProcessPoolExecutor(max_workers=workers, initializer=s1._env)
    try:
        out = s1.run_tasks(pool, workers, fn, tasks, name, min_free, every)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        s1.OUT = old
    pool.shutdown()
    return out


def n_saved(name: str) -> int:
    """Number of tasks already saved in <name>.partial.jsonl."""
    path = os.path.join(OUT, name + ".partial.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def log_run(**rec) -> None:
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with open(os.path.join(OUT, "run_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=_default) + "\n")


def wait_for_ram(min_gb: float, wait_min: float, where: str) -> float:
    """Free physical memory before a stage: wait (up to wait_min minutes) while it is below min_gb, then stop."""
    t0 = time.time()
    free = s1.free_ram_gb()
    while free < min_gb:
        if time.time() - t0 >= wait_min * 60:
            log_run(stage=where, event="stopped: low memory", free_gb=round(free, 2), required_gb=min_gb)
            raise s1.LowMemory(f"free memory {free:.2f} GB < {min_gb:.2f} GB before {where} "
                               f"(waited {wait_min:.0f} min)")
        print(f"  waiting: free memory {free:.2f} GB < {min_gb:.2f} GB before {where}", flush=True)
        time.sleep(60)
        free = s1.free_ram_gb()
    return free


def write_csv(name: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, default=_default) if isinstance(v, (dict, list, tuple)) else v)
                        for k, v in r.items()})


def write_json(name: str, obj) -> None:
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_default)


def read_json(name: str):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def stage(name: str, reuse: bool, fn, before=None):
    path = os.path.join(OUT, name)
    if reuse and os.path.exists(path):
        print(f"[reuse] {name}", flush=True)
        return read_json(name)
    if before is not None:
        before()
    t = time.perf_counter()
    out = fn()
    write_json(name, out)
    print(f"[done] {name} ({time.perf_counter() - t:.0f} s)", flush=True)
    return out


def finish_stage(name: str) -> None:
    path = os.path.join(OUT, name + ".partial.jsonl")
    if os.path.exists(path):
        os.remove(path)


def read_cases_summary() -> Dict:
    """results/cases_summary.csv (fewest-plies study, shear of one sign), read only."""
    path = os.path.join(RESULTS, "cases_summary.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {(r["case"], r["thickness"]): r for r in csv.DictReader(f)}


def read_both_signs() -> Dict:
    path = os.path.join(RESULTS, "shear_both_signs.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {(r["case"], r["thickness"]): r for r in csv.DictReader(f)}


def _float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS, help="worker processes of the GA and enumeration stages")
    ap.add_argument("--problem-workers", type=int, default=PROBLEM_WORKERS,
                    help="worker processes of the exact searches (default 1)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--skip-matlab", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse the JSON of the stages already done")
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="no new task below this free memory (GB)")
    ap.add_argument("--stage-free-gb", type=float, default=4.0, help="free memory (GB) needed to start a stage")
    ap.add_argument("--wait-min", type=float, default=15.0, help="minutes to wait for --stage-free-gb, then stop")
    ap.add_argument("--stop-after", default=None, help="stop after this stage (defaults, verify, problems, ga, "
                                                       "tradeoff)")
    a = ap.parse_args()
    s1._env()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.makedirs(OUT, exist_ok=True)
    workers = max(1, min(a.workers, MAX_WORKERS))
    a.problem_workers = max(1, min(a.problem_workers, MAX_WORKERS))
    print(f"workers {workers} (exact searches {a.problem_workers}), free memory {s1.free_ram_gb():.1f} GB, "
          f"stage start needs {a.stage_free_gb} GB, no new task below {a.min_free_gb} GB", flush=True)
    try:
        run(a, workers)
    except s1.LowMemory as e:
        print(f"STOPPED (low memory): {e}. Finished results are saved in {OUT}; run again with --reuse to continue.",
              flush=True)
        sys.exit(3)


def default_rows() -> List[Dict]:
    """N of the 16 problems: QI rule, else the min_plies fallback read from cases_summary.csv; the CSV design is
    re-evaluated in this model (both shear signs, 8x8) and the fewest ply count is re-checked by the count-level
    scan when the load cannot buckle the plate (stiffness.min_plies_count); min_plies is run only if the CSV row is
    missing or does not match."""
    cs, both = read_cases_summary(), read_both_signs()
    rows = []
    for c in CASES:
        for tk in THICKNESSES:
            cid = c["id"]
            p = problem(cid, tk)
            d = st.default_ply_count(p)
            old = cs.get((cid, tk))
            row = {"case": cid, "name": c["en"], "thickness": tk, "t_mm": THICKNESSES[tk] * 1e3,
                   "Nx_N_per_m": c["loads"][0], "Ny_N_per_m": c["loads"][1], "Nxy_N_per_m": c["loads"][2],
                   "kind": st.objective_kind(p), "qi_found": d["found"], "qi_deciding": d["qi"],
                   "qi_strength_margin": d["rf_strength"], "qi_lambda_8x8": d["lam"],
                   "qi_scan": "; ".join(f"{x['n']}: rs={x['rf_strength']:.4g} lam8={x['lam']:.4g}" for x in d["scan"]),
                   "min_plies_old_n": int(old["n_opt"]) if old else None,
                   "min_plies_old_design": old["design"] if old else None,
                   "min_plies_old_feasible": (old["feasible"] == "True") if old else None,
                   "min_plies_old_status": old["exact_status"] if old else None,
                   "min_plies_old_proven": (bool(old and old["exact_status"] == "optimal"
                                                 and old["n_lower_bound"] == old["n_opt"])),
                   "min_plies_old_margin": _float(old["margin_opt"]) if old else None,
                   "min_plies_both_signs_n": _float((both.get((cid, tk)) or {}).get("both_n")),
                   "min_plies_both_signs_status": (both.get((cid, tk)) or {}).get("both_status")}
            if d["found"]:
                row.update(N=d["n"], N_from="QI_ks", N_detail=d["qi"], N_proven=None, status="ok")
            else:
                src, n, proven, detail, status = None, None, None, None, None
                if old is not None:
                    ok_model = (old["model"] == "ritz_full" and abs(float(old["ply_thickness_mm"]) - row["t_mm"]) < 1e-9
                                and (float(old["Nx"]), float(old["Ny"]), float(old["Nxy"])) == tuple(c["loads"]))
                    n_old = int(old["n_opt"])
                    match = False
                    if ok_model and row["min_plies_old_feasible"]:
                        half = parse_half(old["design"])
                        m = st.evaluate_half(st.fixed_problem(p, n_old), half)
                        row["min_plies_old_recheck_strength_margin"] = m["rf_strength"]
                        row["min_plies_old_recheck_lambda_8x8"] = m["lam"]
                        match = 2 * len(half) == n_old and m["rf_strength"] >= SAFETY_FACTOR and m["lam"] >= SAFETY_FACTOR
                    elif ok_model:
                        match = True        # the CSV says infeasible; checked below when the load cannot buckle
                    row["min_plies_csv_matches_model"] = bool(match)
                    if match:
                        src, n, proven = "cases_summary.csv", (n_old if row["min_plies_old_feasible"] else None), \
                            row["min_plies_old_proven"] if row["min_plies_old_feasible"] else True
                # count-level re-check (exact) when the load cannot buckle the plate; else min_plies only on mismatch
                if not st.can_buckle(p) or src is None:
                    m2 = st.min_plies_count(p)
                    row["min_plies_recheck_n"] = m2["n"]
                    row["min_plies_recheck_proven"] = m2["proven"]
                    row["min_plies_recheck_method"] = m2["method"]
                    if src is None:
                        src, n, proven = "min_plies run (" + m2["method"] + ")", m2["n"], m2["proven"]
                    row["min_plies_recheck_agrees"] = (m2["n"] == n)
                if n is not None:
                    detail = f"min_plies {n} plies ({'proven' if proven else 'not proven'}; source: {src})"
                    status = "ok"
                else:
                    detail = f"no design of 8-64 plies meets the constraints (source: {src})"
                    status = "INFEASIBLE: no QI laminate up to 64 plies and no min_plies design up to 64 plies"
                row.update(N=n, N_from="min_plies fallback" if n is not None else "-", N_detail=detail,
                           N_proven=proven, status=status)
            rows.append(row)
    return rows


def parse_half(design: str) -> np.ndarray:
    """Half laminate (outer -> mid-plane) of a notation like [90/0_2/-45/45]s."""
    body = design.strip()
    assert body.startswith("[") and body.endswith("]s"), design
    code = {"0": 0, "45": 1, "-45": 2, "90": 3}
    out = []
    for tok in body[1:-2].split("/"):
        ang, _, k = tok.partition("_")
        out += [code[ang]] * (int(k) if k else 1)
    return np.array(out, dtype=np.int8)


def run(a, workers: int) -> None:
    T = time.perf_counter()
    mf = a.min_free_gb
    nw = {"verify_tiebreak": workers, "problems": a.problem_workers, "ga_runs": workers, "tradeoff": workers}

    def tasks(fn, items, name, every=0):
        return run_tasks(nw[name], fn, items, name, mf, every)

    def before(name, n_tasks=None):
        def f():
            free = wait_for_ram(a.stage_free_gb, a.wait_min, name)
            saved = n_saved(name)
            log_run(stage=name, event="start", free_gb=round(free, 2), workers=nw.get(name, 0),
                    tasks=n_tasks, already_saved=saved)
            print(f"[start] {name}: free memory {free:.1f} GB, {nw.get(name, 0)} worker(s)"
                  + (f", {saved} task(s) already saved" if saved else ""), flush=True)
        return f

    def staged(name, fn, n_tasks=None):
        out = stage(name + ".json", a.reuse, fn, before(name, n_tasks))
        finish_stage(name)
        return out

    # 1) default ply count (main process)
    defaults = staged("default_ply_count", default_rows)
    write_csv("default_ply_count.csv", defaults)
    if a.stop_after == "defaults":
        return
    probs_def = [(r["case"], r["thickness"], r["N"]) for r in defaults if r["N"]]

    # 2) compliance tie-break vs enumeration
    verify = staged("verify_tiebreak", lambda: tasks(task_verify, VERIFY, "verify_tiebreak"), len(VERIFY))
    write_csv("verify_tiebreak.csv", verify)
    if a.stop_after == "verify":
        return

    # 3) the problems (one at a time by default; the large pure-shear problems first)
    order = sorted(probs_def, key=lambda x: -x[2] if x[0] == "C5" else 0)
    probs = staged("problems", lambda: tasks(task_problem, order, "problems"), len(order))
    by_key = {(r["case"], r["thickness"]): r for r in probs}
    if a.stop_after == "problems":
        return

    # 4) GA and GA + local search, 20 seeds, every problem
    def s_ga():
        items = []
        for cid, tk, n in probs_def:
            r = by_key[(cid, tk)]
            s = r["max_stiffness"]
            # target: the optimum of the 8x8 model (what the GA optimises); for an incomplete search the best known
            half = s["half_8x8"] if s.get("half_8x8") else s["half"]
            m = st.evaluate_half(problem(cid, tk, n), half)
            key = st.make_key(s["kind"], SAFETY_FACTOR, m["rf_strength"], m["lam"], m["c"], half,
                              m["D"][st.main_load_direction(CASE[cid]["loads"])] [st.main_load_direction(
                                  CASE[cid]["loads"])])
            tgt = [key.feasible, key.tier, key.primary, key.secondary, list(key.lex), key.tertiary]
            for method in ("ga", "ga_ls"):
                for seed in range(a.seeds):
                    items.append((cid, tk, n, method, seed, a.budget, tgt, list(half)))
        return tasks(task_ga, items, "ga_runs", every=20)
    ga_runs = staged("ga_runs", s_ga, 2 * a.seeds * len(probs_def))
    if a.stop_after == "ga":
        return

    # 5) figure 4 data: every valid 24-ply design of C6 (streaming, parts saved as .npy, merged into .npz)
    def s_tradeoff():
        cid, n = TRADEOFF
        items = [(cid, tk, n, i, TRADEOFF_PARTS) for tk in THICKNESSES for i in range(TRADEOFF_PARTS)]
        res = tasks(task_tradeoff, items, "tradeoff", every=4)
        out = {}
        for tk in THICKNESSES:
            parts = [r for it, r in zip(items, res) if it[1] == tk]
            arr = np.concatenate([np.load(r["path"]) for r in parts], axis=1)
            lam8, rs = arr[0], arr[1]
            out[tk] = tradeoff_summary(cid, tk, n, lam8, rs, sum(r["time_s"] for r in parts))
            np.savez_compressed(os.path.join(OUT, f"tradeoff_{cid}_N{n}_{tk}.npz"), lambda_8x8=lam8.astype(np.float32),
                                strength_margin=rs.astype(np.float32),
                                pareto_strength_margin=np.array(out[tk]["pareto_strength_margin"]),
                                pareto_lambda_8x8=np.array(out[tk]["pareto_lambda_8x8"]))
            for r in parts:
                os.remove(r["path"])
        try:
            os.rmdir(os.path.join(OUT, "_parts"))
        except OSError:
            pass
        return out
    tradeoff = staged("tradeoff", s_tradeoff, len(THICKNESSES) * TRADEOFF_PARTS)
    if a.stop_after == "tradeoff":
        return
    # no worker process from here on: every pool was closed at the end of its stage (MATLAB and the demo run alone)

    # 6) MATLAB check of the max_stiffness designs (strength from clt_engine.m), saved after every design
    def s_matlab():
        part = os.path.join(OUT, "matlab_check.partial.json")
        out = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else {}
        if a.skip_matlab:
            return out
        from optimization import matlab_check
        try:
            for (cid, tk, n), obj in [(x, o) for o in ("max_stiffness", "max_margin") for x in probs_def]:
                mk = f"{cid}|{tk}|{n}" + ("" if obj == "max_stiffness" else "|max_margin")
                if mk in out:
                    continue
                s1.check_ram(mf, f"the MATLAB check of {mk}")
                p = problem(cid, tk, n)
                full = full_laminate(np.array([by_key[(cid, tk)][obj]["half"]], dtype=np.int8), True)[0]
                try:
                    v = matlab_check.verify_design(p, full.tolist())
                    out[mk] = {"pass": v["pass"], "rf_matlab": v["rf_matlab"], "rf_python": v["rf_python"],
                               "e_rf": v["e_rf"], "e_clt": max(v["clt"]["e_ABD"], v["clt"]["e_stress"],
                                                               v["clt"]["e_FI"])}
                except Exception as e:      # MATLAB not available: reported, not hidden
                    out[mk] = {"pass": None, "error": f"{type(e).__name__}: {e}"}
                with open(part, "w", encoding="utf-8") as f:
                    json.dump(s1._plain(out), f, indent=1)
        finally:
            try:
                matlab_check.close()
            except Exception:
                pass
        return out
    matlab = stage("matlab_check.json", a.reuse, s_matlab, before("matlab_check", 2 * len(probs_def)))
    if os.path.exists(os.path.join(OUT, "matlab_check.partial.json")):
        os.remove(os.path.join(OUT, "matlab_check.partial.json"))

    # 7) demo regression (fewest plies, default objective)
    def s_demo():
        s1.check_ram(mf, "the demo regression")
        sys.path.insert(0, os.path.join(ROOT, "python"))
        import design
        empty = {"fibre_words": "", "material_name": "", "material": "NONE", "loads": [], "safety_factor": 0,
                 "ply_thickness": {"value": 0, "unit": "mm"}, "plate_a": {"value": 0, "unit": "m"},
                 "plate_b": {"value": 0, "unit": "m"}, "ply_count": 0, "ply_count_min": 0, "ply_count_max": 0,
                 "objective_words": "", "shear_direction": "", "angles": []}
        text = "Design the lightest IM7/8552 laminate for a shear load Nxy = 40 kN/m."
        spec = design.build_spec(empty, text)
        r = design.solve(spec, check_matlab=False)
        expected = ["[-45/45_2/-45/45/-45_2/45/90/0_2/90]s", 24, 1.699]
        got = [r["notation"], r["n_plies"], round(r["margin"], 3)]
        return {"request": text, "objective": spec.objective, "shear_both": spec.shear_both,
                "notation": r["notation"], "n_plies": r["n_plies"], "margin": r["margin"],
                "status": r["search"]["status"], "proven": r["search"]["proven"], "expected": expected,
                "pass": got == expected}
    demo = stage("regression_demo.json", a.reuse, s_demo, before("regression_demo", 1))

    # 8) tables and summary
    rows, topk_rows = results_rows(defaults, by_key, matlab)
    fig4 = tradeoff_figure_data(tradeoff, rows)
    write_json("tradeoff_figure.json", fig4)
    for r in probs:
        for x in r.get("top_table") or []:
            topk_rows.append({"case": r["case"], "thickness": r["thickness"], "N": r["N"],
                              **{k: v for k, v in x.items() if k != "half"}})
    write_csv("results.csv", rows)
    write_json("results.json", {"rows": rows, "problems": probs})
    write_csv("topk.csv", topk_rows)
    ga_rows = [{k: v for k, v in r.items() if k != "history"} for r in ga_runs]
    write_csv("ga_runs.csv", ga_rows)
    write_json("ga_history.json", [{k: r[k] for k in ("case", "thickness", "N", "method", "seed", "kind", "history")}
                                   for r in ga_runs])
    ga_sum = ga_summary(ga_runs, by_key)
    write_csv("ga_summary.csv", ga_sum)
    verify_old = step1_designs()
    write_summary(defaults, verify, rows, probs, ga_sum, tradeoff, matlab, demo, verify_old,
                  time.perf_counter() - T, ga_runs, fig4)
    print(f"all done ({time.perf_counter() - T:.0f} s this run)", flush=True)


def tradeoff_summary(cid, tk, n, lam8, rs, cpu) -> Dict:
    """Stiffest (largest lambda8 meeting both constraints), strongest (largest min(strength margin, lambda8)), the QI
    laminate and the Pareto front (largest strength margin and lambda8) of every valid design."""
    sf = SAFETY_FACTOR
    p = problem(cid, tk, n)
    feas = (rs >= sf) & (lam8 >= sf)
    i_s = int(np.argmax(np.where(feas, lam8, -np.inf)))
    margin = np.minimum(rs, lam8)
    i_m = int(np.argmax(margin))
    q = st.evaluate_half(p, st.qi_half(n))
    # Pareto front (maximise both): for every strength-margin value the largest lambda, then the non-dominated ones
    order = np.lexsort((-lam8, -rs))
    front, best = [], -np.inf
    for i in order:
        if lam8[i] > best * (1 + 1e-12):
            front.append(int(i))
            best = lam8[i]
    ex = st.solve_max_stiffness(p)
    mm = optimize(None, problem=p, objective="max_margin", method="exhaustive", time_limit=MM_TIME)
    return {"case": cid, "thickness": tk, "N": n, "n_valid": int(len(rs)), "n_feasible": int(feas.sum()),
            "cpu_s": cpu,
            "stiffest_lambda_8x8": float(lam8[i_s]), "stiffest_strength_margin": float(rs[i_s]),
            "strongest_lambda_8x8": float(lam8[i_m]), "strongest_strength_margin": float(rs[i_m]),
            "strongest_margin": float(margin[i_m]),
            "qi_lambda_8x8": q["lam"], "qi_strength_margin": q["rf_strength"],
            "exact_stiffest_lambda_8x8": ex["lam_8x8_best"], "exact_stiffest_design_8x8": notation(
                full_laminate(np.array([ex["half_8x8"]], np.int8), True)[0], True),
            "exact_reported_design": ex["notation"], "max_margin_design": mm.notation,
            "max_margin_margin_8x8": mm.stiffness["margin"],
            "enumeration_equals_exact": bool(abs(lam8[i_s] - ex["lam_8x8_best"]) <= 1e-9 * ex["lam_8x8_best"]),
            "enumeration_equals_max_margin": bool(abs(margin[i_m] - mm.stiffness["margin"]) <= 1e-9 * margin[i_m]),
            "pareto_strength_margin": [float(rs[i]) for i in front], "pareto_lambda_8x8": [float(lam8[i]) for i in front]}


def tradeoff_figure_data(tradeoff, rows) -> Dict:
    """Figure 4 data, from saved results only (tradeoff_C6_N24_*.npz, tradeoff.json, results rows); nothing is
    searched again. Strongest = the max_margin result (objective and tie-break unchanged): results.csv when N = 24
    is the default N, else the max_margin result at N = 24 of tradeoff.json; the one design is evaluated to place
    it and its (strength margin, lambda 8x8) pair is checked against the enumeration file. Every design whose margin
    min(strength margin, lambda 8x8) equals the largest one ties with it: count and lambda 8x8 range. Groups: the
    designs of one strength-margin value (one ply-count tuple) form one vertical line, lambda 8x8 min to max. Pareto
    points of tradeoff.json (largest strength margin and lambda 8x8)."""
    cid, n = TRADEOFF
    out = {}
    for tk, t in tradeoff.items():
        z = np.load(os.path.join(OUT, f"tradeoff_{cid}_N{n}_{tk}.npz"))
        lam32, rs32 = z["lambda_8x8"], z["strength_margin"]
        lam, rs = lam32.astype(np.float64), rs32.astype(np.float64)
        margin = np.minimum(rs, lam)
        best = float(margin.max())
        tie = margin >= best * (1 - 1e-6)          # float32 file; exact equality gives the same count here
        row = next((r for r in rows if r["case"] == cid and r.get("N (-)") == n
                    and abs(r["t (mm)"] - THICKNESSES[tk] * 1e3) < 1e-9), None)
        design = row["max_margin sequence"] if row else t["max_margin_design"]
        if design != t["max_margin_design"]:
            raise RuntimeError(f"max_margin design of results.csv and tradeoff.json differ ({tk})")
        p = problem(cid, tk, n)
        h = parse_half(design)
        e = st.evaluate_half(p, h)
        groups = []
        for v in np.unique(rs32):
            sel = rs32 == v
            groups.append({"strength_margin": float(v), "lambda_8x8_min": float(lam[sel].min()),
                           "lambda_8x8_max": float(lam[sel].max()), "n_designs": int(sel.sum())})
        i_first = int(np.argmax(margin))
        out[tk] = {
            "case": cid, "thickness": tk, "t_mm": THICKNESSES[tk] * 1e3, "N": n,
            "default_N": next((r["N (-)"] for r in rows if r["case"] == cid
                               and abs(r["t (mm)"] - THICKNESSES[tk] * 1e3) < 1e-9), None),
            "n_valid": int(len(rs)), "n_feasible": t["n_feasible"],
            "strongest_source": ("results.csv (max_margin at the default N)" if row else
                                 "tradeoff.json (max_margin at N = 24, the same objective and tie-break)"),
            "strongest_design": design, "strongest_strength_margin": e["rf_strength"],
            "strongest_lambda_8x8": e["lam"], "strongest_lambda_16x16": float(st.report_lambdas(p, h[None])[0]),
            "strongest_margin": min(e["rf_strength"], e["lam"]),
            "strongest_pair_in_enumeration": bool(np.any((lam32 == np.float32(e["lam"]))
                                                         & (rs32 == np.float32(e["rf_strength"])))),
            "strongest_lambda_8x8_results_csv": row["max_margin lambda 8x8 (-)"] if row else None,
            "max_margin_value": best, "n_tied_for_strongest": int(tie.sum()),
            "n_tied_exact_float32": int((margin == margin.max()).sum()),
            "tied_lambda_8x8_min": float(lam[tie].min()), "tied_lambda_8x8_max": float(lam[tie].max()),
            "tied_strength_margins": sorted({float(v) for v in rs32[tie]}),
            "first_enumerated_at_max_margin": {"strength_margin": float(rs[i_first]), "lambda_8x8": float(lam[i_first]),
                                               "note": "tradeoff.json 'strongest_*' (shown by the earlier figure)"},
            "stiffest_strength_margin": t["stiffest_strength_margin"], "stiffest_lambda_8x8": t["stiffest_lambda_8x8"],
            "qi_strength_margin": t["qi_strength_margin"], "qi_lambda_8x8": t["qi_lambda_8x8"],
            "pareto": [[a, b] for a, b in zip(t["pareto_strength_margin"], t["pareto_lambda_8x8"])],
            "groups": groups}
    return out


def results_rows(defaults, by_key, matlab):
    rows, topk = [], []
    for d in defaults:
        cid, tk = d["case"], d["thickness"]
        base = {"case": cid, "load case": d["name"], "Nx (N/m)": d["Nx_N_per_m"], "Ny (N/m)": d["Ny_N_per_m"],
                "Nxy (N/m)": d["Nxy_N_per_m"], "t (mm)": d["t_mm"], "N (-)": d["N"], "N from": d["N_from"],
                "N detail": d["N_detail"] if d["N_from"] != "QI_ks" else f"smallest feasible QI {d['qi_deciding']}",
                "objective": "max lambda" if d["kind"] == "lambda" else "min c"}
        old = {"min_plies old N (-)": d["min_plies_old_n"], "min_plies old sequence": d["min_plies_old_design"],
               "min_plies old proven": d["min_plies_old_proven"],
               "min_plies old feasible": d["min_plies_old_feasible"],
               "min_plies both shear signs N (-)": d["min_plies_both_signs_n"]}
        r = by_key.get((cid, tk))
        if r is None:
            rows.append({**base, "status": d["status"], **old})
            continue
        s, m = r["max_stiffness"], r["max_margin"]
        n = r["N"]
        p = problem(cid, tk, n)
        # reference: the old fewest-plies design, re-evaluated in this model (both shear signs) at its own N
        if d["min_plies_old_design"]:
            oh = parse_half(d["min_plies_old_design"])
            po = problem(cid, tk, 2 * len(oh))
            om = st.evaluate_half(po, oh)
            odet = analyze_design(po, full_laminate(oh[None], True)[0].tolist())
            old.update({"min_plies old lambda 16x16 (-)": odet["lambda"], "min_plies old c (N/m)": om["c"],
                        "min_plies old strength margin (-)": om["rf_strength"]})
        mk = f"{cid}|{tk}|{n}"
        ml = matlab.get(mk) or {}
        qi = s.get("qi") or {}
        qi_ok = bool(qi.get("feasible"))
        vs = s.get("improvement_vs_qi_pct")
        lam_check = (abs(s["lambda_poly16"] / s["lambda_16x16"] - 1) if s["kind"] == "lambda" else None)
        rows.append({
            **base, "status": s["status_label"],
            "max_stiffness sequence": s["notation"],
            "lambda 16x16 (-)": s["lambda_16x16"], "lambda poly p=16 (-)": s["lambda_poly16"],
            "lambda 8x8 (-)": s["lambda_8x8"], "rel diff poly16 vs 16x16 (-)": lam_check,
            "c (N/m)": s["c_N_per_m"], "strength margin (-)": s["strength_margin"],
            "D11 (N.m)": s["D11_Nm"], "D22 (N.m)": s["D22_Nm"], "D66 (N.m)": s["D66_Nm"],
            "D16 (N.m)": s["D16_Nm"], "D26 (N.m)": s["D26_Nm"], "main direction": s["main_direction"],
            "QI sequence": qi.get("notation"), "QI feasible": qi_ok, "QI lambda 16x16 (-)": qi.get("lambda_16x16"),
            "QI lambda 8x8 (-)": qi.get("lambda_8x8"), "QI c (N/m)": qi.get("c_N_per_m"),
            "QI strength margin (-)": qi.get("strength_margin"),
            "vs QI (%)": vs if qi_ok else None,
            # rounded ONCE from the unrounded value above; summary, table and figures all print this column
            "vs QI 1 decimal (%)": round(vs, 1) if (qi_ok and vs is not None) else None,
            "vs QI meaning": (s.get("improvement_vs_qi_meaning") if qi_ok else
                              f"no QI with the same N: {qi.get('notation')} fails the constraints (strength margin "
                              f"{qi.get('strength_margin', float('nan')):.4g})" if qi else "no QI with the same N"),
            "max_margin sequence": m["notation"], "max_margin lambda 16x16 (-)": m["lambda_16x16"],
            "max_margin lambda 8x8 (-)": m["lambda_8x8"], "max_margin c (N/m)": m["c_N_per_m"],
            "max_margin strength margin (-)": m["strength_margin"],
            "max_margin margin min(strength, lambda 8x8) (-)": m["margin"], "max_margin status": m["status"],
            "max_margin D11 (N.m)": m["D11_Nm"], "max_margin D22 (N.m)": m["D22_Nm"], "max_margin D66 (N.m)": m["D66_Nm"],
            "same design stiffest / strongest": s["half"] == m["half"],
            "same design up to +45/-45 swap": s["half"] in (m["half"], [{1: 2, 2: 1}.get(x, x) for x in m["half"]]),
            **old,
            "proven": s["proven"], "proven in": s.get("proven_in"), "gap (-)": s.get("gap"),
            "k final": s.get("k_final"), "k rule met": s.get("k_rule_met"), "k cap reached": s.get("k_cap_reached"),
            "best 8x8 differs from best 16x16": s.get("best_differs"), "best 8x8 sequence": s.get("notation_8x8"),
            "lambda 8x8 of the 8x8 best (-)": s.get("lambda_8x8_best"),
            "rank 8x8 of the reported design": s.get("rank_8x8_of_selected"),
            "lambda analyze_design (-)": s.get("lambda_analyze_design"),
            "time max_stiffness (s)": s["runtime_s"], "time max_margin (s)": m["runtime_s"],
            "MATLAB pass": ml.get("pass"), "MATLAB strength margin (-)": s1._ml_strength(ml),
            "MATLAB max rel error RF (-)": max(ml["e_rf"].values()) if ml.get("e_rf") else None,
            "MATLAB error": ml.get("error"),
            "max_margin MATLAB pass": (matlab.get(mk + "|max_margin") or {}).get("pass"),
            "max_margin MATLAB strength margin (-)": s1._ml_strength(matlab.get(mk + "|max_margin"))})
    return rows, topk


def ga_summary(ga_runs, by_key) -> List[Dict]:
    out = []
    keys = []
    for r in ga_runs:
        k = (r["case"], r["thickness"], r["N"], r["method"])
        if k not in keys:
            keys.append(k)
    for cid, tk, n, method in keys:
        rr = [x for x in ga_runs if (x["case"], x["thickness"], x["N"], x["method"]) == (cid, tk, n, method)]
        hits = [x["hit_evals"] for x in rr if x["hit_evals"] is not None]
        s = by_key[(cid, tk)]["max_stiffness"]
        out.append({"case": cid, "thickness": tk, "t (mm)": THICKNESSES[tk] * 1e3, "N": n, "method": method,
                    "runs": len(rr), "exact result": s["status_label"],
                    "target": ("optimum of the 8x8 model" if s["proven"] else "best known (not proven)"),
                    "reach optimum (full budget)": sum(x["reaches_optimum"] for x in rr),
                    "same values (objective, margin, D)": sum(x["same_values_as_exact"] for x in rr),
                    "same sequence as exact 8x8 best": sum(x["same_design_as_exact"] for x in rr),
                    "hit within 1000 evals": sum(h <= 1000 for h in hits),
                    "hit within 2000 evals": sum(h <= 2000 for h in hits),
                    "median evals to hit": float(np.median(hits)) if hits else None,
                    "feasible": sum(x["feasible"] for x in rr),
                    "evals mean": float(np.mean([x["n_evaluations"] for x in rr])),
                    "time mean (s)": float(np.mean([x["runtime_s"] for x in rr]))})
    return out


def step1_designs() -> Dict:
    """Designs of step 1 for the same (case, thickness, N), for the 'Changes from step 1' section."""
    out = {}
    for name in ("verify_16.csv", "verify_24.csv"):
        path = os.path.join(STEP1, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    out[(r["case"], r["thickness"], int(r["N"]))] = {"design": r["exact_design"],
                                                                      "lambda": _float(r["exact_lambda"])}
    path = os.path.join(STEP1, "problems.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("N"):
                    out[(r["case"], r["thickness"], int(r["N"]))] = {"design": r["design"],
                                                                      "lambda": _float(r.get("lambda"))}
    return out


# ------------------------------------------------------------------------------------------------ summary
def _f(x, nd=4):
    return s1._f(x, nd)


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return "?"


def lambda_triplet(cid, tk, n, half) -> Dict:
    """lambda 8x8, 16x16 and polynomial p = 16 and the strength margin of one design."""
    p = problem(cid, tk, n)
    h = np.asarray(half, dtype=np.int8)
    m = st.evaluate_half(p, h)
    return {"notation": notation(full_laminate(h[None], True)[0], True), "lambda_8x8": m["lam"],
            "lambda_16x16": float(st.report_lambdas(p, h[None])[0]), "lambda_poly16": st.lambda_poly(p, m["D"]),
            "strength_margin": m["rf_strength"]}


def infeasible_notes(defaults) -> List[str]:
    """Problems with no N: comparison with the old fewest-plies result (cases_summary.csv, read only)."""
    out = []
    for d in defaults:
        if d["N"]:
            continue
        p = problem(d["case"], d["thickness"])
        m = st.min_plies_count(p)
        last = m["scan"][-1] if m.get("scan") else None
        txt = (f"- **{d['case']}, t = {d['t_mm']:.5g} mm: infeasible.** No [0/45/-45/90]_ks up to 64 plies meets the "
               f"constraints (QI strength margins: {d['qi_scan']}).")
        if last is not None:
            txt += (f" Largest strength margin of any valid design (count-level scan, exact): "
                    f"{last['max_strength_margin']:.4f} at {last['n']} plies < {SAFETY_FACTOR}.")
        if d["min_plies_old_design"]:
            oh = parse_half(d["min_plies_old_design"])
            om = st.evaluate_half(problem(d["case"], d["thickness"], 2 * len(oh)), oh)
            txt += (f" Old fewest-plies study (cases_summary.csv): status '{d['min_plies_old_status']}', feasible "
                    f"{d['min_plies_old_feasible']}, best design {d['min_plies_old_design']} ({2 * len(oh)} plies) "
                    f"with margin {d['min_plies_old_margin']:.4f}; re-evaluated in this model (both shear signs): "
                    f"strength margin {om['rf_strength']:.4f}. shear_both_signs.csv: "
                    f"{d['min_plies_both_signs_status']}. "
                    + ("The old and the new results agree (no feasible design of at most 64 plies)."
                       if not d["min_plies_old_feasible"] and not m["found"] else
                       "**The old and the new results DISAGREE: to be investigated (conclusion not changed).**"))
        out += [txt, ""]
    return out


def key_points(defaults, probs, ga_runs, tradeoff) -> List[str]:
    """The points to read first: problems with no N, searches that did not finish (best known design, bound, gap,
    best GA result) and problems whose best design by the 16 x 16 value is not the 8 x 8 optimum (both sequences)."""
    T = THICKNESSES
    probs_ok = [pr for pr in probs if pr["max_stiffness"]["proven"]]
    L = ["## Key points", "",
         f"- {len(defaults)} problems; {len(probs)} have a default N and were solved, "
         f"{len(defaults) - len(probs)} infeasible; {len(probs_ok)} proven optima, "
         f"{len(probs) - len(probs_ok)} INCOMPLETE (not proven)."]
    for d in defaults:
        if d["N"]:
            continue
        agree = (d["min_plies_old_feasible"] is False and d.get("min_plies_recheck_n") is None
                 and bool(d.get("min_plies_recheck_agrees")))
        L.append(f"- **{d['case']}, t = {d['t_mm']:.5g} mm: INFEASIBLE.** No [0/45/-45/90]_ks and no design of 8-64 "
                 f"plies meets both constraints (count-level scan of the largest strength margin, exact). Old "
                 f"fewest-plies study: results/cases_summary.csv '{d['min_plies_old_status']}' (best margin "
                 f"{d['min_plies_old_margin']:.4f} < {SAFETY_FACTOR} with 64 plies), results/shear_both_signs.csv "
                 f"'{d['min_plies_both_signs_status']}': "
                 + ("the new result agrees with the old one." if agree else
                    "**the new result DISAGREES with the old one (to be investigated).**")
                 + " Details below.")
    for pr in probs:
        s = pr["max_stiffness"]
        if s["proven"]:
            continue
        ga = [g for g in ga_runs if (g["case"], g["thickness"]) == (pr["case"], pr["thickness"])]
        best_ga = max((g["lambda_8x8"] for g in ga), default=None)
        txt = (f"- **{pr['case']}, t = {T[pr['thickness']] * 1e3:.5g} mm, N = {pr['N']}: {s['status_label']}.** The "
               f"exact search reached its time limit of {EXACT_TIME:.0f} s (total {s['runtime_s']:.0f} s). Best design "
               f"found: {s['notation']}, "
               f"lambda 8x8 = {s['lambda_8x8']:.6f} (16x16 = {s['lambda_16x16']:.6f}); upper bound of the 8x8 model "
               f"{s['upper_bound']:.6f}, gap {100 * s['gap']:.3f} % in the 8x8 model. Not proven: the optimum lies "
               f"between these two values of lambda 8x8.")
        if best_ga is not None:
            better = best_ga > s["lambda_8x8"] * (1 + 1e-9)
            txt += (f" GA and GA + local search ({len(ga)} runs) compared with this best known design: largest "
                    f"lambda 8x8 found {best_ga:.6f}, "
                    + ("**better than the exact search's best design**." if better else
                       "no run found a better design."))
        L.append(txt)
    for pr in probs:
        s = pr["max_stiffness"]
        if not s.get("best_differs"):
            continue
        a = lambda_triplet(pr["case"], pr["thickness"], pr["N"], s["half"])
        b = lambda_triplet(pr["case"], pr["thickness"], pr["N"], s["half_8x8"])
        txt = (f"- **{pr['case']}, t = {T[pr['thickness']] * 1e3:.5g} mm, N = {pr['N']}: the best design by the 16 x "
               f"16 value is not the 8 x 8 optimum.** Best by the 8 x 8 model: {b['notation']} (lambda 8x8 = "
               f"{b['lambda_8x8']:.6f}, 16x16 = {b['lambda_16x16']:.6f}, strength margin {b['strength_margin']:.4f}). "
               f"Best by the 16 x 16 value (reported): {a['notation']} (lambda 8x8 = {a['lambda_8x8']:.6f}, 16x16 = "
               f"{a['lambda_16x16']:.6f}, strength margin {a['strength_margin']:.4f}).")
        t = tradeoff.get(pr["thickness"]) if (pr["case"], pr["N"]) == TRADEOFF else None
        if t:
            txt += (f" The enumeration of all {t['n_valid']:,} valid designs (figure 4) gives the same largest "
                    f"lambda 8x8: {_f(t['enumeration_equals_exact'])}.")
        L.append(txt)
    return L + [""]


def run_log_lines() -> List[str]:
    path = os.path.join(OUT, "run_log.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    out = ["", "## Run log (run_log.jsonl)", "",
           "Free physical memory at the start of each stage (a stage starts only with at least 4 GB free), worker "
           "processes of the stage and the tasks already saved by an earlier, stopped run (reused with --reuse).", "",
           "| time | stage | event | free memory (GB) | workers | tasks | already saved |", "|---|---|---|---|---|---|---|"]
    for r in recs:
        out.append(f"| {r['time']} | {r['stage']} | {r['event']} | {r.get('free_gb', '-')} | {r.get('workers', '-')} "
                   f"| {r.get('tasks', '-')} | {r.get('already_saved', '-')} |")
    return out


def write_summary(defaults, verify, rows, probs, ga_sum, tradeoff, matlab, demo, old, total, ga_runs=(), fig4=None):
    import scipy
    T = THICKNESSES
    L = ["# Fixed thickness, stiffest laminate (objective max_stiffness): step 2", "",
         "IM7/8552, simply supported 0.5 m x 0.5 m plate, safety factor 1.5 on the strength margin min(RF Max Stress, "
         "RF Tsai-Wu) and on the buckling factor lambda, shear checked for +Nxy and -Nxy, symmetric balanced laminates "
         "of 0/+45/-45/90 plies with the design rules of the project. Objective: largest lambda; when the load cannot "
         "buckle the plate (C1, C2, C8), smallest in-plane compliance c = N^T A^-1 N [N/m].", "",
         "- **Search model and proofs**: Ritz double sine series 8 x 8 terms. 'proven' means proven in the 8x8 Ritz "
         "model (the compliance problems involve no Ritz model: count-level search, exact).",
         "- **Reported lambda** (tables, figures except figure 4, analyze_design): Ritz double sine series 16 x 16 "
         "terms; the polynomial Ritz solution p = 16 and the 8 x 8 value are given next to it. The k best designs of "
         "the 8 x 8 model (k = 20 first) are re-evaluated with 16 x 16 terms; k is doubled while (lambda8_1 - "
         "lambda8_k) / lambda8_1 < the largest relative 8x8-16x16 difference in the list (up to 500), and the design "
         "reported is the best of the list by the 16 x 16 value. The 16 x 16 basis contains the 8 x 8 one, so "
         "lambda16 <= lambda8 for every design and no design outside the list can be better by the 16 x 16 value.",
         "- **Ties of the compliance objective**: (1) smallest c, (2) largest strength margin, (3) largest bending "
         "stiffness in the main load direction (largest of |Nx|, |Ny|, |Nxy|: D11, D22 or D66), (4) lexicographically "
         "smallest sequence (0 < +45 < -45 < 90 from the outer ply). Within 1e-9 (relative) two values are equal.",
         "- **Default N**: smallest [0/45/-45/90]_ks (N <= 64) meeting both constraints ('QI_ks'); if none, the "
         "fewest plies of the objective min_plies ('min_plies fallback'); if neither exists the problem is "
         "infeasible.", ""]
    L += key_points(defaults, probs, ga_runs, tradeoff)
    # default N
    L += ["## Default ply count", "",
          "| case | t (mm) | objective | N | N from | detail | old fewest plies (cases_summary.csv) |",
          "|---|---|---|---|---|---|---|"]
    for d in defaults:
        L.append(f"| {d['case']} | {d['t_mm']:.5g} | {'lambda' if d['kind'] == 'lambda' else 'c'} | {_f(d['N'])} | "
                 f"{d['N_from']} | {d['N_detail']} | {_f(d['min_plies_old_n'])} "
                 f"({d['min_plies_old_status']}{', proven' if d['min_plies_old_proven'] else ''}) |")
    L += ["", "cases_summary.csv is the fewest-plies study with shear of one sign. For C8 (no buckling, balanced "
          "laminates) the strength margin is the same for +Nxy and -Nxy, so its ply count holds for both signs; it "
          "is re-checked here by the count-level scan of the largest strength margin over every N (exact) and by "
          "results/shear_both_signs.csv.", ""]
    L += infeasible_notes(defaults)
    # tie-break verification
    L += ["## Compliance tie-break: exact search vs enumeration of every valid design", "",
          "c and the strength margin were compared design by design with the first design of the same ply counts: "
          "largest relative difference over every valid design below (0 = identical numbers).", "",
          "| case | t (mm) | N | valid designs | designs tied on c and margin | status | direction | exact design | "
          "enumeration design | same | D (N.m) exact / enumeration | spread of c / margin within counts | match |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in verify:
        L.append(f"| {r['case']} | {r['t_mm']:.5g} | {r['N']} | {r['n_valid_designs']:,} | "
                 f"{r['n_designs_tied_on_c_and_margin'] if r['exact_status'] == 'optimal' else '- (infeasible)'} | "
                 f"{r['exact_status']} | {r['direction']} | "
                 f"{r['exact_design']} | {r['brute_design']} | {_f(r['same_design'])} | {r['exact_ddir_Nm']:.10g} / "
                 f"{r['brute_ddir_Nm']:.10g} | {r['max_rel_spread_c_within_counts']:.1e} / "
                 f"{r['max_rel_spread_margin_within_counts']:.1e} | {_f(r['match'])} |")
    # main table
    L += ["", "## The 16 problems (max_stiffness, exact search, default N)", "",
          "lambda: 16 x 16 value (reported); vs QI: change against [0/45/-45/90]_ks with the same N (lambda: "
          "increase of the 16 x 16 value, c: decrease). Strength margin = min(RF Max Stress, RF Tsai-Wu).", "",
          "vs QI is computed once from the unrounded values (results.csv column 'vs QI (%)') and rounded once to one "
          "decimal (column 'vs QI 1 decimal (%)'); this summary, the table and the figures of "
          "deliverables/figures_stiffness/ all print that column.", "",
          "| case | t (mm) | N | N from | stiffest sequence | lambda 16x16 | lambda poly p=16 | lambda 8x8 | c (N/m) | "
          "strength margin | vs QI (%) | status | time (s) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "max_stiffness sequence" not in r:
            L.append(f"| {r['case']} | {r['t (mm)']:.5g} | - | {r['N from']} | - | - | - | - | - | - | - | "
                     f"**{r['status']}** | - |")
            continue
        vs = (f"{r['vs QI 1 decimal (%)']:+.1f} ({'lambda' if r['objective'] == 'max lambda' else 'c'})"
              if r["vs QI 1 decimal (%)"] is not None else "no QI with the same N")
        L.append(f"| {r['case']} | {r['t (mm)']:.5g} | {r['N (-)']} | {r['N from']} | {r['max_stiffness sequence']} | "
                 f"{_f(r['lambda 16x16 (-)'], 5)} | {_f(r['lambda poly p=16 (-)'], 5)} | {_f(r['lambda 8x8 (-)'], 5)} | "
                 f"{_f(r['c (N/m)'], 6)} | {_f(r['strength margin (-)'])} | {vs} | {r['status']} | "
                 f"{r['time max_stiffness (s)']:.1f} |")
    # QI and D
    L += ["", "QI with the same N and bending stiffness:", "",
          "| case | t (mm) | N | QI | QI lambda 16x16 | QI c (N/m) | QI strength margin | QI feasible | D11 / D22 / D66 "
          "(N.m) stiffest | main direction |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "max_stiffness sequence" not in r:
            continue
        L.append(f"| {r['case']} | {r['t (mm)']:.5g} | {r['N (-)']} | {r['QI sequence']} | "
                 f"{_f(r['QI lambda 16x16 (-)'], 5)} | {_f(r['QI c (N/m)'], 6)} | {_f(r['QI strength margin (-)'])} | "
                 f"{_f(r['QI feasible'])} | {r['D11 (N.m)']:.4g} / {r['D22 (N.m)']:.4g} / {r['D66 (N.m)']:.4g} | "
                 f"{r['main direction']} |")
    # top-k
    L += ["", "## Top-k rule (8x8 -> 16x16)", "",
          "| case | t (mm) | N | k final | gap of the list | largest 8x8-16x16 difference | rule met | best 8x8 = "
          "best 16x16 | 8x8 rank of the reported design |", "|---|---|---|---|---|---|---|---|---|"]
    for pr in probs:
        s = pr["max_stiffness"]
        h = s.get("k_history") or []
        if not h:
            if s["kind"] == "lambda" and not s["proven"]:
                L.append(f"| {pr['case']} | {T[pr['thickness']] * 1e3:.5g} | {pr['N']} | - | - | - | no list: exact "
                         f"search INCOMPLETE, the best known design is reported | - | - |")
            continue
        L.append(f"| {pr['case']} | {T[pr['thickness']] * 1e3:.5g} | {pr['N']} | {s['k_final']} | "
                 f"{h[-1]['gap']:.3e} | {h[-1]['e_max']:.3e} | {_f(s['k_rule_met'])}"
                 f"{' (k cap reached)' if s.get('k_cap_reached') else ''} | {_f(not s['best_differs'])} | "
                 f"{s['rank_8x8_of_selected']} |")
    diff = [pr for pr in probs if pr["max_stiffness"].get("best_differs")]
    L += [""]
    if not diff:
        L += ["- In every problem the best design by the 16 x 16 value is the 8 x 8 optimum."]
    for pr in diff:
        s = pr["max_stiffness"]
        L += [f"- {pr['case']} t = {T[pr['thickness']] * 1e3:.5g} mm, N = {pr['N']}: the best design by the 16 x 16 "
              f"value is not the 8 x 8 optimum; the 16 x 16 one is reported.", "",
              "  | design | sequence | lambda 8x8 | lambda 16x16 | lambda poly p=16 | strength margin | 8x8 rank | "
              "16x16 rank |", "  |---|---|---|---|---|---|---|---|"]
        for label, half in (("reported (best by 16x16)", s["half"]), ("8 x 8 optimum", s["half_8x8"])):
            v = lambda_triplet(pr["case"], pr["thickness"], pr["N"], half)
            row = next((x for x in pr.get("top_table") or [] if x["half"] == list(half)), {})
            L.append(f"  | {label} | {v['notation']} | {v['lambda_8x8']:.6f} | {v['lambda_16x16']:.6f} | "
                     f"{v['lambda_poly16']:.6f} | {v['strength_margin']:.4f} | {_f(row.get('rank_8x8'))} | "
                     f"{_f(row.get('rank_16x16'))} |")
        L.append("")
    # stiffest vs strongest
    L += ["", "## Stiffest (max_stiffness) vs strongest (max_margin) at the same N", "",
          "max_margin maximises margin = min(strength margin, lambda 8x8) (objective of the fewest-plies study, "
          "unchanged); its lambda is also given with 16 x 16 terms. Once lambda 8x8 reaches the strength margin of "
          "the ply counts a larger lambda does not count, so many designs can tie; the exact search keeps the first "
          "one it finds (see figure 4 below for the number of ties). With shear of both signs a design and its mirror "
          "(+45 and -45 swapped) have the same values; the two objectives break such ties differently, so 'same up "
          "to swap' is given.", "",
          "| case | t (mm) | N | stiffest | lambda 16x16 | c (N/m) | strength margin | strongest | lambda 16x16 | "
          "c (N/m) | strength margin | strongest status | same (up to +45/-45 swap) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "max_stiffness sequence" not in r:
            continue
        L.append(f"| {r['case']} | {r['t (mm)']:.5g} | {r['N (-)']} | {r['max_stiffness sequence']} | "
                 f"{_f(r['lambda 16x16 (-)'], 5)} | {_f(r['c (N/m)'], 6)} | {_f(r['strength margin (-)'])} | "
                 f"{r['max_margin sequence']} | {_f(r['max_margin lambda 16x16 (-)'], 5)} | "
                 f"{_f(r['max_margin c (N/m)'], 6)} | {_f(r['max_margin strength margin (-)'])} | "
                 f"{r['max_margin status']} | {_f(r['same design up to +45/-45 swap'])} |")
    # old fewest plies
    L += ["", "## Old fewest-plies design of each case (reference)", "",
          "From results/cases_summary.csv (shear of one sign), re-evaluated here with both shear signs at its own N.",
          "", "| case | t (mm) | N old | sequence | lambda 16x16 | c (N/m) | strength margin | fewest plies with both "
          "signs (shear_both_signs.csv) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['case']} | {r['t (mm)']:.5g} | {_f(r.get('min_plies old N (-)'))} | "
                 f"{r.get('min_plies old sequence') or '-'} | {_f(r.get('min_plies old lambda 16x16 (-)'), 5)} | "
                 f"{_f(r.get('min_plies old c (N/m)'), 6)} | {_f(r.get('min_plies old strength margin (-)'))} | "
                 f"{_f(r.get('min_plies both shear signs N (-)'))} |")
    # GA
    L += ["", "## GA and GA + local search (20 seeds, budget 10 000 designs, 8x8 model)", "",
          "Hit = the run's best design reaches the objective value of the exact 8 x 8 optimum (lambda8 or c within a "
          "relative 1e-9) with the same feasibility and tier; 'same values' also requires the same strength margin "
          "and (compliance) the same bending stiffness.", "",
          "| case | t (mm) | N | method | hit (full budget) | hit <= 1 000 | hit <= 2 000 | same values | same sequence | "
          "median evaluations to hit | target |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in ga_sum:
        L.append(f"| {g['case']} | {g['t (mm)']:.5g} | {g['N']} | {g['method']} | "
                 f"{g['reach optimum (full budget)']}/{g['runs']} | {g['hit within 1000 evals']}/{g['runs']} | "
                 f"{g['hit within 2000 evals']}/{g['runs']} | {g['same values (objective, margin, D)']}/{g['runs']} | "
                 f"{g['same sequence as exact 8x8 best']}/{g['runs']} | {_f(g['median evals to hit'])} | "
                 f"{g['target']} |")
    # tradeoff
    L += ["", "## Stiffness-strength trade-off, C6 with 24 plies (figure 4)", "",
          "'Strongest' is the max_margin design (objective and code unchanged). max_margin maximises margin = "
          "min(strength margin, lambda 8x8). When lambda 8x8 is at least the strength margin of the ply counts, the "
          "margin equals that strength margin and a larger lambda adds nothing, so every arrangement of the best ply "
          "counts with lambda 8x8 >= its strength margin ties. The exact search keeps the first design it finds with "
          "the largest margin (no further tie-break). Figure 4 marks that design: the max_margin sequence of "
          "results.csv (the one of figure 3) when N = 24 is the default N, else the max_margin result at N = 24 "
          "(tradeoff.json). Tie counts and lambda ranges are counted in the enumeration files (float32 values; "
          "exact equality and a relative 1e-6 give the same counts).", ""]
    for tk, t in tradeoff.items():
        g = (fig4 or {}).get(tk) or {}
        dn = g.get("default_N")
        note = (f"default N of C6 for this thickness" if dn == TRADEOFF[1] else
                f"NOT the default N of C6 for this thickness (default {dn})")
        txt = (f"- t = {T[tk] * 1e3:.5g} mm ({note}): {t['n_valid']:,} valid designs, {t['n_feasible']:,} meet both "
               f"constraints; stiffest lambda8 = {t['stiffest_lambda_8x8']:.6g} (strength margin "
               f"{t['stiffest_strength_margin']:.4g}); QI lambda8 = {t['qi_lambda_8x8']:.4g}, strength margin "
               f"{t['qi_strength_margin']:.4g}; enumeration = exact search: {_f(t['enumeration_equals_exact'])}, = "
               f"max_margin (margin value): {_f(t['enumeration_equals_max_margin'])}; CPU {t['cpu_s']:.0f} s.")
        if g:
            txt += (f" Strongest (max_margin, {g['strongest_source']}): {g['strongest_design']}, strength margin "
                    f"{g['strongest_strength_margin']:.4f}, lambda8 {g['strongest_lambda_8x8']:.4f} (16x16 "
                    f"{g['strongest_lambda_16x16']:.4f}), margin {g['strongest_margin']:.6f}. "
                    f"{g['n_tied_for_strongest']:,} designs tie for the strongest; their lambda8 ranges from "
                    f"{g['tied_lambda_8x8_min']:.4f} to {g['tied_lambda_8x8_max']:.4f}. Pareto points (strength "
                    f"margin, lambda8): " + ", ".join(f"({a:.4f}, {b:.4f})" for a, b in g["pareto"]) + ".")
        L.append(txt)
    # MATLAB, lambda check
    n_ml = sum(1 for v in matlab.values() if v.get("pass"))
    L += ["", "## Checks", "",
          f"- MATLAB clt_engine.m (strength recomputed, compared with Python) of the {len(matlab)} designs "
          f"(max_stiffness and max_margin): {n_ml}/{len(matlab)} pass.",
          "- lambda: 16 x 16 sine series vs polynomial Ritz p = 16, largest relative difference over the buckling "
          f"problems: {max([r['rel diff poly16 vs 16x16 (-)'] for r in rows if r.get('rel diff poly16 vs 16x16 (-)') is not None] or [0]):.2e}.",
          "- analyze_design gives the same lambda as the results file for every design (column 'lambda "
          "analyze_design (-)' of results.csv equals 'lambda 16x16 (-)'): "
          f"{_f(all(r.get('lambda analyze_design (-)') == r.get('lambda 16x16 (-)') for r in rows if 'max_stiffness sequence' in r))}.",
          f"- Demo regression (fewest plies, default objective): \"{demo['request']}\" -> {demo['notation']}, "
          f"{demo['n_plies']} plies, margin {demo['margin']:.4f} -> {'PASS' if demo['pass'] else 'FAIL'}."]
    # changes from step 1
    L += ["", "## Changes from step 1", "",
          "- Compliance (tension-only) problems: ties on c and the strength margin are now broken by the bending "
          "stiffness of the main load direction before the sequence, so their stacking sequences can change (c and "
          "the strength margin cannot: they depend on the ply counts only). Same (case, t, N) as step 1:"]
    for r in rows:
        if r["objective"] != "min c" or "max_stiffness sequence" not in r:
            continue
        o = old.get((r["case"], {round(v * 1e3, 6): k for k, v in T.items()}[round(r["t (mm)"], 6)], r["N (-)"]))
        if o:
            L.append(f"  - {r['case']} t = {r['t (mm)']:.5g} mm, N = {r['N (-)']}: step 1 {o['design']} -> step 2 "
                     f"{r['max_stiffness sequence']}{' (unchanged)' if o['design'] == r['max_stiffness sequence'] else ''}")
    for v in verify:
        o = old.get((v["case"], v["thickness"], v["N"]))
        if o:
            L.append(f"  - {v['case']} t = {v['t_mm']:.5g} mm, N = {v['N']} (verification): step 1 {o['design']} -> "
                     f"step 2 {v['exact_design']}{' (unchanged)' if o['design'] == v['exact_design'] else ''}")
    L += ["- lambda is now reported with 16 x 16 terms (step 1: 8 x 8, the optimisation model) and the reported "
          "design of a buckling problem is the best of the k best 8 x 8 designs by the 16 x 16 value.",
          "- Default N: when no QI laminate up to 64 plies meets the constraints (C8), N is the fewest plies of the "
          "objective min_plies (step 1: infeasible, no N).",
          "- deliverables/optimization/stiffness_step1/ is kept unchanged as the record of step 1."]
    # reproducibility
    L += ["", "## Reproducibility", "",
          f"- Code: git commit {_git_hash()} (the step-2 code itself is in the commit that adds this folder); run "
          "from python/: `..\\venv\\Scripts\\python.exe -m optimization.experiments.stiffness_step2` (add --reuse "
          "to continue a stopped run).",
          f"- Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}; "
          f"{platform.system()} {platform.release()}.",
          "- GA seeds 0-19, budget 10 000 distinct designs per run; at most 2 worker processes.",
          "- tradeoff_C6_N24_*.npz (lambda 8x8 and strength margin of every valid design, float32) are written by "
          "the 'tradeoff' stage of the same script.",
          f"- Time of the last invocation (stages reused from earlier, stopped runs not counted; see the stage times "
          f"in the tables and the run log): {total:.0f} s."]
    L += run_log_lines()
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
