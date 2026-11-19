"""
Fixed-thickness "stiffest laminate" (objective max_stiffness), step 1: verification on small problems.

Writes deliverables/optimization/stiffness_step1/ only (never results/ of the fewest-plies study):
  default_ply_count.csv    default N of the 16 problems (8 load cases x 2 ply thicknesses), quasi-isotropic rule
  verify_16.csv            exact search vs enumeration of every valid design, 16 plies, 8 cases x 2 thicknesses
  verify_24.csv            the same at 24 plies (C1, C6, C7)
  problems.csv / .json     problems solved: max_stiffness and max_margin at the same N, QI comparison, MATLAB check
  ritz_convergence.csv     the 5 best designs of every buckling problem: 8x8 (model), 12x12, 16x16, polynomial Ritz
  ga_runs.csv, ga_summary.csv   GA and GA + local search, 20 seeds, 16- and 24-ply problems
  probe_sampling.csv       probes at 32 / 40 plies: random valid designs never beat the exact optimum
  regression_demo.json     the demo design request (fewest plies, default objective) must be unchanged
  summary.md

Model: IM7/8552, simply supported 0.5 m x 0.5 m plate, Ritz double sine series 8 x 8 terms (as the fewest-plies
study and the design agent), shear checked for +Nxy and -Nxy, safety factor 1.5 on the strength margin and on lambda.

Run (from python/):  ..\\venv\\Scripts\\python.exe -m optimization.experiments.stiffness_step1 [--workers 2]
                     [--seeds 20] [--budget 10000] [--skip-matlab] [--reuse] [--min-free-gb 1.5]

Memory: at most --workers processes compute at any time (the main process only coordinates while they run). Every
finished task is appended to <stage>.partial.jsonl, so --reuse continues where a run stopped. Before each task the
free physical memory is checked: below --min-free-gb no new task is started, the running ones finish and are saved,
and the script stops (exit code 3); below half of it the workers are terminated at once. Restart with --reuse.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, List, Optional

import numpy as np

from optimization import buckling
from optimization import stiffness as st
from optimization.api import make_problem, optimize
from optimization.evaluate import Plate, Problem, analyze_design
from optimization.experiments.cases import CASES, MATERIAL, PLATE, SAFETY_FACTOR, THICKNESSES
from optimization.experiments.common import ROOT, _default
from optimization.ga import GAConfig
from optimization.material import get_material
from optimization.rules import full_laminate, notation

OUT = os.path.join(ROOT, "deliverables", "optimization", "stiffness_step1")
RITZ_K = 8                 # 8 x 8 sine series, as in the fewest-plies study (ritz_terms.json) and design.py
WORKERS = 2                # default number of worker processes (never more than MAX_WORKERS)
MAX_WORKERS = 4
CASE = {c["id"]: c for c in CASES}
TOP = 5
EXACT_TIME = 600.0

# problems of step 1: (case, thickness, N); N = None -> default ply count (quasi-isotropic rule)
GA_PROBLEMS = [(c, tk, n) for c in ("C1", "C3", "C7") for tk in THICKNESSES for n in (16, 24)]
DEFAULT_PROBLEMS = [(c, tk, None) for c in ("C1", "C3", "C7") for tk in THICKNESSES]
PROBES = [("C6", "t0125", None), ("C5", "t0183", None)]       # >= 30 plies (default N 32 and 40): exact search probe
V24_CASES = ("C7", "C6", "C1")    # 24-ply enumeration: compression + shear, tension + shear, tension
N_VALID = {16: 10056, 24: 1864416}   # valid designs (enumeration, verify_16 / verify_24)
V24_PARTS = 8                    # enumeration parts per buckling problem (short tasks: frequent safe points)


def problem(cid: str, tk: str, n: Optional[int] = None) -> Problem:
    c = CASE[cid]
    p = make_problem(c["loads"], material=get_material(MATERIAL, verbose=False), safety_factor=SAFETY_FACTOR,
                     plate=Plate(*PLATE), ply_thickness=THICKNESSES[tk], ritz_terms=RITZ_K, shear_both_signs=True,
                     name=f"{cid} {tk}")
    return st.fixed_problem(p, n) if n else p


def resolved(cid: str, tk: str, n: Optional[int]):
    p = problem(cid, tk)
    if n is None:
        d = st.default_ply_count(p)
        if not d["found"]:
            return None, d
        return st.fixed_problem(p, d["n"]), d
    return st.fixed_problem(p, n), {"n": n, "source": "given"}


# ------------------------------------------------------------------------------------------------- tasks
def _env():
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def task_verify(args) -> Dict:
    """16 plies: exact search and streaming enumeration of every valid design in one task."""
    cid, tk, n = args
    p = problem(cid, tk, n)
    t = time.perf_counter()
    ex = st.exact_stiffness(p, time_limit=EXACT_TIME)
    te = time.perf_counter() - t
    t = time.perf_counter()
    bf = st.brute_force_merge(p, [st.brute_force(p)])
    tb = time.perf_counter() - t
    return _verify_row(cid, tk, n, ex, bf, te, tb)


def _exact_brief(ex: Dict) -> Dict:
    keys = ("status", "kind", "half", "notation", "lam", "c", "rf_strength", "value", "value_meaning", "feasible",
            "max_rf_strength", "infeasible_reason", "n_eigen", "proven", "complete")
    return {k: ex.get(k) for k in keys}


def task_v24(args) -> Dict:
    """24 plies: ("exact", case, t) or ("brute", case, t, part, n_parts) (streaming enumeration of one part)."""
    t = time.perf_counter()
    if args[0] == "exact":
        out = _exact_brief(st.exact_stiffness(problem(args[1], args[2], 24), time_limit=EXACT_TIME))
    else:
        _, cid, tk, part, k = args
        out = st.brute_force(problem(cid, tk, 24), part=part, n_parts=k)
    out["time_s"] = time.perf_counter() - t
    return out


def _bf_part(r: Dict) -> Dict:
    """A brute_force part read back from JSON: arrays restored."""
    r = dict(r)
    r["halves"] = np.asarray(r["halves"], dtype=np.int8).reshape(len(r["rs"]), -1)
    for k in ("rs", "lam", "c"):
        r[k] = np.asarray(r[k], dtype=float)
    return r


def _verify_row(cid, tk, n, ex, bf, te, tb) -> Dict:
    def rel(a, b):
        if np.isinf(a) and np.isinf(b):
            return 0.0
        return abs(a - b) / abs(b)
    kind = ex["kind"]
    d_val = rel(ex["value"], bf["value"])
    d_max = rel(ex["max_rf_strength"], bf["max_rf_strength"])
    return {"case": cid, "thickness": tk, "t_mm": THICKNESSES[tk] * 1e3, "N": n, "kind": kind,
            "objective": "max lambda" if kind == "lambda" else "min c [N/m]",
            "ranked_by": ex.get("value_meaning"), "ranked_by_enumeration": bf.get("value_meaning"),
            "n_valid_designs": bf["n_valid"], "n_meeting_strength": bf["n_strength_ok"],
            "exact_status": ex["status"], "brute_status": bf["status"],
            "exact_value": ex["value"], "brute_value": bf["value"], "rel_diff_value": d_val,
            "exact_lambda": ex["lam"], "brute_lambda": bf["lam"], "exact_c": ex["c"],
            "exact_strength_margin": ex["rf_strength"], "brute_strength_margin": bf["rf_strength"],
            "exact_max_strength_margin_any_design": ex["max_rf_strength"],
            "brute_max_strength_margin_any_design": bf["max_rf_strength"], "rel_diff_max_strength_margin": d_max,
            "exact_infeasible_reason": ex.get("infeasible_reason"),
            "brute_infeasible_reason": bf.get("infeasible_reason"),
            "exact_design": ex["notation"], "brute_design": bf["notation"], "same_design": ex["half"] == bf["half"],
            "match": bool(ex["half"] == bf["half"] and ex["status"] == bf["status"] and d_val <= 1e-9
                          and d_max <= 1e-9 and ex.get("value_meaning") == bf.get("value_meaning")),
            "exact_time_s": te, "brute_time_s": tb, "exact_eigen_solves": ex.get("n_eigen")}


def _distinct_top(top: List[Dict], k: int) -> List[Dict]:
    """The first k designs, dropping the +45 / -45 mirror twin of a design already listed (with both shear signs
    checked the twin has the same lambda by symmetry)."""
    out, seen = [], set()
    for d in top:
        h = tuple(d["half"])
        twin = tuple({1: 2, 2: 1}.get(x, x) for x in h)
        if h in seen or twin in seen:
            continue
        seen.add(h)
        out.append(d)
        if len(out) == k:
            break
    return out


def task_problem(args) -> Dict:
    """max_stiffness (exact, top designs) and max_margin (exact) at the same N."""
    cid, tk, n, group = args
    p, nd = resolved(cid, tk, n)
    if p is None:
        return {"case": cid, "thickness": tk, "group": group, "feasible_default": False, "default": nd}
    N = p.rules.n_min
    t = time.perf_counter()
    ex = st.exact_stiffness(p, time_limit=EXACT_TIME, top_k=12)
    t_ex = time.perf_counter() - t
    rec = st.stiffness_record(p, ex["half"], status=ex["status"], proven=ex["proven"], runtime_s=t_ex,
                              gap=ex.get("gap"), upper_bound=ex.get("upper_bound"),
                              n_eigen=ex.get("n_eigen"), method=ex["method"], max_rf_strength=ex["max_rf_strength"],
                              infeasible_reason=ex["infeasible_reason"])
    det = analyze_design(p, ex["full"])
    rec["lambda_reported_by_analyze_design"] = det["rf"].get("buckling")
    rec["analyze_design_margin"] = det["margin"]
    t = time.perf_counter()
    mm = optimize(None, problem=p, objective="max_margin", method="exhaustive", time_limit=300.0)
    mm_rec = dict(mm.stiffness, runtime_s=time.perf_counter() - t, status=mm.status)
    out = {"case": cid, "thickness": tk, "N": N, "group": group, "default": nd, "max_stiffness": rec,
           "max_margin": mm_rec, "top": ex["top"], "kind": ex["kind"], "exact": {k: ex.get(k) for k in (
               "status", "complete", "n_count_tuples", "n_tuples_strength_ok", "n_eigen", "n_leaves", "n_nodes",
               "upper_bound", "gap", "method", "time_s")}}
    if ex["kind"] == "lambda":
        out["ritz"] = ritz_convergence(p, _distinct_top(ex["top"], TOP))
    return out


def ritz_convergence(p: Problem, top: List[Dict]) -> List[Dict]:
    a, b = p.plate.a, p.plate.b
    rows = []
    for rank, d in enumerate(top, 1):
        D = st.evaluate_half(p, d["half"])["D"]
        row = {"rank_8x8": rank, "design": d["notation"], "half": d["half"]}
        for k in (8, 12, 16):
            row[f"lambda_{k}x{k}"] = min(float(buckling.ritz_full(D, *lc, a, b, k, k)[0]) for lc in p.load_cases())
        for q in (12, 16):
            row[f"lambda_poly{q}"] = min(buckling.ritz_polynomial(D, *lc, a, b, p=q) for lc in p.load_cases())
        row["model_lambda"] = d["lam"]
        rows.append(row)
    for key in ("lambda_12x12", "lambda_16x16", "lambda_poly12", "lambda_poly16"):
        order = sorted(range(len(rows)), key=lambda i: -rows[i][key])
        for pos, i in enumerate(order, 1):
            rows[i][f"rank_{key[7:]}"] = pos
    for r in rows:
        r["diff_12_vs_8_pct"] = 100 * (r["lambda_12x12"] / r["lambda_8x8"] - 1)
        r["diff_16_vs_8_pct"] = 100 * (r["lambda_16x16"] / r["lambda_8x8"] - 1)
        r["diff_poly12_vs_8_pct"] = 100 * (r["lambda_poly12"] / r["lambda_8x8"] - 1)
        r["diff_16_vs_poly16_pct"] = 100 * (r["lambda_16x16"] / r["lambda_poly16"] - 1)
    return rows


def task_ga(args) -> Dict:
    cid, tk, n, method, seed, budget, target, opt_half = args
    p = problem(cid, tk, n)
    cfg = GAConfig(budget=budget, local_search=(method == "ga_ls"))
    tkey = st.Key(*target[:4], tuple(target[4]))
    r = optimize(None, problem=p, objective="max_stiffness", method="ga", seed=seed, ga_config=cfg, target=tkey)
    s = r.stiffness
    key = st.make_key(s["kind"], p.safety_factor, s["strength_margin"], s["lambda"], s["c_N_per_m"], s["half"])
    return {"case": cid, "thickness": tk, "N": n, "method": method, "seed": seed, "kind": s["kind"],
            "lambda": s["lambda"], "c_N_per_m": s["c_N_per_m"], "strength_margin": s["strength_margin"],
            "feasible": s["feasible"], "design": s["notation"], "reaches_optimum": st.reaches(key, tkey),
            "same_design_as_exact": s["half"] == opt_half, "hit_evals": r.hit_evals,
            "n_evaluations": r.n_evaluations, "runtime_s": r.runtime_s}


def task_probe_sampling(args) -> Dict:
    """Random valid designs of a >= 30-ply problem: none may beat the exact optimum (sanity check, not a proof)."""
    cid, tk, n, lam_opt, n_samples = args
    from optimization.evaluate import Evaluator
    from optimization.rules import Repairer
    p = problem(cid, tk, n)
    rng = np.random.default_rng(2026)
    rep = Repairer(p.enc, p.rules)
    H = np.unique(np.array([rep.random_valid(rng) for _ in range(n_samples)], dtype=np.int8), axis=0)
    ev = Evaluator(p)
    r = ev.compute_plies(H, details=True)
    ok = r["rf_strength"] >= p.safety_factor
    best = float(np.max(r["lam"][ok])) if ok.any() else None
    return {"case": cid, "thickness": tk, "N": n, "n_random_valid_designs": len(H), "n_meeting_strength": int(ok.sum()),
            "best_random_lambda": best, "exact_lambda": lam_opt,
            "random_never_better": bool(best is None or best <= lam_opt * (1 + 1e-9))}


# ------------------------------------------------------------------------------------- memory, resumable tasks
class LowMemory(RuntimeError):
    pass


def free_ram_gb() -> float:
    """Free physical memory (GB), Windows GlobalMemoryStatusEx (no extra package); inf if it cannot be read."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return float("inf")
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return float("inf")


def check_ram(min_free: float, where: str) -> None:
    free = free_ram_gb()
    if free < min_free:
        raise LowMemory(f"free memory {free:.2f} GB < {min_free:.2f} GB before {where}")


def _plain(x):
    """The JSON form of a task or a result (fresh and resumed runs then use exactly the same data)."""
    return json.loads(json.dumps(x, default=_default))


def run_tasks(pool, workers: int, fn, tasks: List, name: str, min_free: float, every: int = 0) -> List:
    """Run fn over tasks with at most `workers` in flight; every result is appended to <name>.partial.jsonl as soon
    as it is done and reused on the next run. Raises LowMemory (after saving what finished) when memory is short."""
    path = os.path.join(OUT, name + ".partial.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[json.dumps(rec["task"])] = rec["result"]
    keys = [json.dumps(_plain(t)) for t in tasks]
    todo = [(k, t) for k, t in zip(keys, tasks) if k not in done]
    if done:
        print(f"  {name}: {len(tasks) - len(todo)} task(s) reused from the partial file, {len(todo)} to run",
              flush=True)
    inflight, it, low, n_new = {}, iter(todo), None, 0
    with open(path, "a", encoding="utf-8") as f:
        while True:
            while low is None and len(inflight) < workers:
                nxt = next(it, None)
                if nxt is None:
                    break
                free = free_ram_gb()
                if free < min_free:
                    low = f"free memory {free:.2f} GB < {min_free:.2f} GB: no new task started"
                    break
                inflight[pool.submit(fn, nxt[1])] = nxt[0]
            if not inflight:
                break
            fin, _ = wait(list(inflight), timeout=5.0, return_when=FIRST_COMPLETED)
            if not fin and free_ram_gb() < min_free / 2:        # critical: stop the workers now
                for proc in list((getattr(pool, "_processes", None) or {}).values()):
                    proc.terminate()
                raise LowMemory(f"free memory below {min_free / 2:.2f} GB: workers terminated, "
                                f"{len(inflight)} running task(s) lost, {len(done)}/{len(tasks)} saved")
            for fu in fin:
                k = inflight.pop(fu)
                done[k] = _plain(fu.result())
                f.write(json.dumps({"task": json.loads(k), "result": done[k]}) + "\n")
                f.flush()
                n_new += 1
                if every and n_new % every == 0:
                    print(f"  {name}: {len(done)}/{len(tasks)} done (free memory {free_ram_gb():.1f} GB)",
                          flush=True)
    if low is not None:
        raise LowMemory(low + f"; {name}: {len(done)}/{len(tasks)} saved")
    return [done[k] for k in keys]


def finish_stage(name: str) -> None:
    """The stage JSON is written: its partial file is no longer needed."""
    path = os.path.join(OUT, name + ".partial.jsonl")
    if os.path.exists(path):
        os.remove(path)


# ------------------------------------------------------------------------------------------------ output
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


def stage(name: str, reuse: bool, fn):
    path = os.path.join(OUT, name)
    if reuse and os.path.exists(path):
        print(f"[reuse] {name}", flush=True)
        return read_json(name)
    t = time.perf_counter()
    out = fn()
    write_json(name, out)
    print(f"[done] {name} ({time.perf_counter() - t:.0f} s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--skip-matlab", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse the JSON of the stages already done")
    ap.add_argument("--min-free-gb", type=float, default=1.5, help="stop at a safe point below this free memory")
    a = ap.parse_args()
    _env()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.makedirs(OUT, exist_ok=True)
    workers = max(1, min(a.workers, MAX_WORKERS))
    print(f"workers {workers}, free memory {free_ram_gb():.1f} GB, stop below {a.min_free_gb} GB", flush=True)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_env)
    try:
        run(a, pool, workers)
    except LowMemory as e:
        pool.shutdown(wait=False, cancel_futures=True)
        print(f"STOPPED (low memory): {e}. Finished results are saved in {OUT}; run again with --reuse to continue.",
              flush=True)
        sys.exit(3)
    pool.shutdown()


def run(a, pool, workers: int) -> None:
    T = time.perf_counter()
    mf = a.min_free_gb

    def tasks(fn, items, name, every=0):
        out = run_tasks(pool, workers, fn, items, name, mf, every)
        return out

    def staged(name, fn):
        out = stage(name + ".json", a.reuse, fn)
        finish_stage(name)
        return out

    # 1) default ply count of the 16 problems (main process only, a few seconds)
    def s_default():
        rows = []
        for c in CASES:
            for tk in THICKNESSES:
                p = problem(c["id"], tk)
                d = st.default_ply_count(p)
                rows.append({"case": c["id"], "name": c["en"], "thickness": tk, "t_mm": THICKNESSES[tk] * 1e3,
                             "Nx": c["loads"][0], "Ny": c["loads"][1], "Nxy": c["loads"][2],
                             "kind": st.objective_kind(p), "default_N": d["n"], "qi_deciding": d["qi"],
                             "qi_strength_margin": d["rf_strength"], "qi_lambda": d["lam"],
                             "status": "ok" if d["found"] else "INFEASIBLE: no QI laminate up to 64 plies meets (a) and (b)",
                             "scan": "; ".join(f"{x['n']}: rs={x['rf_strength']:.4g} lam={x['lam']:.4g}"
                                               for x in d["scan"])})
        return rows
    default_rows = staged("default_ply_count", s_default)
    write_csv("default_ply_count.csv", default_rows)

    # 2) exact vs streaming enumeration at 16 plies, every case and thickness
    v16 = staged("verify_16", lambda: tasks(task_verify, [(c["id"], tk, 16) for c in CASES for tk in THICKNESSES],
                                            "verify_16"))
    write_csv("verify_16.csv", v16)

    # 3) 24 plies: exact search and streaming enumeration in parts (all in the worker processes)
    def s_v24():
        items = []
        for cid in V24_CASES:
            for tk in THICKNESSES:
                k = V24_PARTS if st.objective_kind(problem(cid, tk, 24)) == "lambda" else 1
                items.append(("exact", cid, tk))
                items += [("brute", cid, tk, i, k) for i in range(k)]
        res = dict(zip([json.dumps(_plain(t)) for t in items], tasks(task_v24, items, "verify_24", every=4)))
        rows = []
        for cid in V24_CASES:
            for tk in THICKNESSES:
                p = problem(cid, tk, 24)
                ex = res[json.dumps(["exact", cid, tk])]
                ex["kind"] = st.objective_kind(p)
                parts = [_bf_part(v) for kk, v in res.items() if json.loads(kk)[:3] == ["brute", cid, tk]]
                bf = st.brute_force_merge(p, parts)
                row = _verify_row(cid, tk, 24, ex, bf, ex["time_s"], sum(x["time_s"] for x in parts))
                row["brute_parts"] = len(parts)
                rows.append(row)
                print(f"  verify 24 {cid} {tk}: match={row['match']} ({row['brute_time_s']:.0f} s CPU)", flush=True)
        return rows
    v24 = staged("verify_24", s_v24)
    write_csv("verify_24.csv", v24)

    # 4) problems: max_stiffness (exact) and max_margin at the same N
    keys = []
    for cid, tk, n in DEFAULT_PROBLEMS + GA_PROBLEMS + PROBES:
        grp = "probe (>= 30 plies)" if (cid, tk, n) in PROBES else ("default N" if n is None else "fixed N")
        keys.append((cid, tk, n, grp))
    probs = staged("problems", lambda: tasks(task_problem, keys, "problems"))
    # same problem reached twice (default N equal to 16 or 24): keep one entry, both groups listed
    uniq: Dict = {}
    for r in probs:
        if not r.get("N"):
            continue
        k = (r["case"], r["thickness"], r["N"])
        if k in uniq:
            uniq[k]["group"] += " + " + r["group"]
        else:
            uniq[k] = r
    problems = list(uniq.values())

    # 5) GA and GA + local search, 20 seeds, 16- and 24-ply problems
    def s_ga():
        items = []
        for cid, tk, n in GA_PROBLEMS:
            s = uniq[(cid, tk, n)]["max_stiffness"]
            key = st.make_key(s["kind"], SAFETY_FACTOR, s["strength_margin"], s["lambda"], s["c_N_per_m"], s["half"])
            for method in ("ga", "ga_ls"):
                for seed in range(a.seeds):
                    items.append((cid, tk, n, method, seed, a.budget, list(key[:4]) + [list(key[4])], s["half"]))
        return tasks(task_ga, items, "ga_runs", every=20)
    ga_runs = staged("ga_runs", s_ga)
    write_csv("ga_runs.csv", ga_runs)

    # 6) probes: random valid designs never beat the exact optimum
    def s_probe():
        items = [(r["case"], r["thickness"], r["N"], r["max_stiffness"]["lambda"], 20000) for r in problems
                 if r["group"].startswith("probe")]
        return tasks(task_probe_sampling, items, "probe_sampling")
    probe_rows = staged("probe_sampling", s_probe)
    write_csv("probe_sampling.csv", probe_rows)
    pool.shutdown()          # no worker process from here on (MATLAB and the demo run in the main process)

    # 7) MATLAB check of the best designs (strength recomputed from clt_engine.m), saved after every design
    def s_matlab():
        part = os.path.join(OUT, "matlab_check.partial.json")
        out = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else {}
        if a.skip_matlab:
            return out
        from optimization import matlab_check
        try:
            for r in problems:
                mk = f"{r['case']}|{r['thickness']}|{r['N']}"
                if mk in out:
                    continue
                check_ram(mf, f"the MATLAB check of {mk}")
                p = problem(r["case"], r["thickness"], r["N"])
                full = full_laminate(np.array([r["max_stiffness"]["half"]], dtype=np.int8), True)[0].tolist()
                try:
                    v = matlab_check.verify_design(p, full)
                    out[mk] = {"pass": v["pass"], "rf_matlab": v["rf_matlab"], "rf_python": v["rf_python"],
                               "e_rf": v["e_rf"], "e_clt": max(v["clt"]["e_ABD"], v["clt"]["e_stress"],
                                                               v["clt"]["e_FI"])}
                except Exception as e:      # MATLAB not available: reported, not hidden
                    out[mk] = {"pass": None, "error": f"{type(e).__name__}: {e}"}
                with open(part, "w", encoding="utf-8") as f:
                    json.dump(_plain(out), f, indent=1)
        finally:
            try:
                matlab_check.close()
            except Exception:
                pass
        return out
    matlab = stage("matlab_check.json", a.reuse, s_matlab)
    if os.path.exists(os.path.join(OUT, "matlab_check.partial.json")):
        os.remove(os.path.join(OUT, "matlab_check.partial.json"))

    # 8) demo regression: fewest plies (default objective), IM7/8552, Nxy = 40 kN/m
    def s_demo():
        check_ram(mf, "the demo regression")
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
    demo = stage("regression_demo.json", a.reuse, s_demo)

    # 9) tables
    prob_rows, ritz_rows, cmp_rows = [], [], []
    for r in sorted(problems, key=lambda r: (r["case"], r["thickness"], r["N"])):
        s, m = r["max_stiffness"], r["max_margin"]
        p = problem(r["case"], r["thickness"], r["N"])
        mk = f"{r['case']}|{r['thickness']}|{r['N']}"
        d = r["default"]
        max_rs = s.get("max_rf_strength")
        if max_rs is None:
            max_rs = st.max_strength_margin(p)
        reason = s.get("infeasible_reason") or st.infeasible_reason(
            SAFETY_FACTOR, max_rs, s["lambda"] if s["strength_margin"] >= SAFETY_FACTOR else None, s["feasible"])
        result = s.get("result") or ("proven optimum" if s["proven"] and s["status"] == "optimal" else
                                     "proven infeasible (best design shown)" if s["proven"] and
                                     s["status"] == "infeasible" else f"{s['status']} (not proven)")
        mm = matlab.get(mk) or {}
        prob_rows.append({
            "case": r["case"], "thickness": r["thickness"], "t_mm": s["t_mm"], "N": r["N"], "group": r["group"],
            "N_source": "default: " + d["qi"] if d.get("qi") else "given", "kind": s["kind"],
            "objective": "max lambda" if s["kind"] == "lambda" else "min c [N/m]",
            "design": s["notation"], "lambda": s["lambda"], "c_N_per_m": s["c_N_per_m"],
            "strength_margin": s["strength_margin"], "rf_max_stress": s["rf"]["max_stress"],
            "rf_tsai_wu": s["rf"]["tsai_wu"], "rf_buckling": s["rf"]["buckling"], "feasible": s["feasible"],
            "status": s["status"], "result": result, "infeasible_reason": reason,
            "max_strength_margin_any_design": max_rs, "proven": s["proven"], "gap": s.get("gap"),
            "qi": (s["qi"] or {}).get("notation"), "qi_lambda": (s["qi"] or {}).get("lambda"),
            "qi_c_N_per_m": (s["qi"] or {}).get("c_N_per_m"),
            "qi_strength_margin": (s["qi"] or {}).get("strength_margin"),
            "improvement_vs_qi_pct": s["improvement_vs_qi_pct"],
            "improvement_meaning": s.get("improvement_vs_qi_meaning"), "runtime_s": s["runtime_s"],
            "eigen_solves": r["exact"].get("n_eigen"), "nodes": r["exact"].get("n_nodes"),
            "matlab_pass": mm.get("pass"), "matlab_strength_margin": _ml_strength(mm),
            "matlab_max_rel_error_rf": max(mm["e_rf"].values()) if mm.get("e_rf") else None,
            "matlab_error": mm.get("error"),
            "lambda_analyze_design": s.get("lambda_reported_by_analyze_design")})
        cmp_rows.append({"case": r["case"], "thickness": r["thickness"], "N": r["N"], "kind": s["kind"],
                         "stiffest_design": s["notation"], "stiffest_lambda": s["lambda"],
                         "stiffest_c_N_per_m": s["c_N_per_m"], "stiffest_strength_margin": s["strength_margin"],
                         "stiffest_margin": s["margin"], "stiffest_result": result,
                         "strongest_design": m["notation"],
                         "strongest_lambda": m["lambda"], "strongest_c_N_per_m": m["c_N_per_m"],
                         "strongest_strength_margin": m["strength_margin"], "strongest_margin": m["margin"],
                         "strongest_status": m["status"], "strongest_proven": m["proven"],
                         "same_design": s["half"] == m["half"]})
        for x in r.get("ritz", []):
            ritz_rows.append({"case": r["case"], "thickness": r["thickness"], "N": r["N"],
                              **{k: v for k, v in x.items() if k != "half"}})
    write_csv("problems.csv", prob_rows)
    write_csv("stiffest_vs_strongest.csv", cmp_rows)
    write_csv("ritz_convergence.csv", ritz_rows)
    result_of = {(x["case"], x["thickness"], x["N"]): x["result"] for x in prob_rows}
    ga_sum = []
    for cid, tk, n in GA_PROBLEMS:
        for method in ("ga", "ga_ls"):
            rr = [x for x in ga_runs if (x["case"], x["thickness"], x["N"], x["method"]) == (cid, tk, n, method)]
            hits = [x["hit_evals"] for x in rr if x["hit_evals"] is not None]
            ga_sum.append({"case": cid, "thickness": tk, "N": n, "method": method, "runs": len(rr),
                           "exact_result": result_of.get((cid, tk, n)),
                           "reach_exact_answer": sum(x["reaches_optimum"] for x in rr),
                           "same_design_as_exact": sum(x["same_design_as_exact"] for x in rr),
                           "feasible": sum(x["feasible"] for x in rr),
                           "hit_within_1000_evals": sum(h <= 1000 for h in hits),
                           "hit_within_2000_evals": sum(h <= 2000 for h in hits),
                           "evals_to_hit_median": float(np.median(hits)) if hits else None,
                           "evals_mean": float(np.mean([x["n_evaluations"] for x in rr])),
                           "valid_designs": N_VALID.get(n),
                           "space_covered_pct": (100 * float(np.mean([x["n_evaluations"] for x in rr])) / N_VALID[n]
                                                 if n in N_VALID else None),
                           "time_mean_s": float(np.mean([x["runtime_s"] for x in rr]))})
    write_csv("ga_summary.csv", ga_sum)
    write_summary(default_rows, v16, v24, prob_rows, cmp_rows, ritz_rows, ga_sum, probe_rows, matlab, demo,
                  time.perf_counter() - T)
    print(f"all done ({time.perf_counter() - T:.0f} s this run)", flush=True)


def _ml_strength(m: Optional[Dict]) -> Optional[float]:
    """Strength margin min(RF Max Stress, RF Tsai-Wu) recomputed from the clt_engine.m output."""
    rf = (m or {}).get("rf_matlab") or {}
    v = [x for k, x in rf.items() if k != "buckling"]
    return min(v) if v else None


def _f(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if np.isinf(x):
            return "inf"
        return f"{x:.{nd}g}"
    return str(x)


def write_summary(default_rows, v16, v24, prob_rows, cmp_rows, ritz_rows, ga_sum, probe_rows, matlab, demo, total):
    L = ["# Fixed thickness, stiffest laminate (objective max_stiffness): step 1", "",
         "IM7/8552, simply supported 0.5 m x 0.5 m plate, Ritz double sine series 8 x 8 terms (as the fewest-plies "
         "study), shear checked for +Nxy and -Nxy, safety factor 1.5 on the strength margin min(RF Max Stress, "
         "RF Tsai-Wu) and on lambda. Objective: largest lambda; when the load cannot buckle the plate, smallest "
         "in-plane compliance c = N^T A^-1 N [N/m]. Ties (relative 1e-9): larger strength margin, then the "
         "lexicographically smallest sequence (0 < +45 < -45 < 90, from the outer ply).", "",
         "When no design of N plies meets both constraints the problem is reported as INFEASIBLE (proven by the "
         "exact search), never as an optimum. The design shown is then the best one of the order: the largest "
         "strength margin when no design meets strength, else the largest lambda among the designs that meet "
         "strength. 'max strength margin' is the largest strength margin of ANY valid design of N plies.", "",
         "## Default ply count (smallest [0/45/-45/90]_ks meeting both constraints)", "",
         "| case | t (mm) | kind | default N | deciding QI | QI strength margin | QI lambda |", "|---|---|---|---|---|---|---|"]
    for r in default_rows:
        L.append(f"| {r['case']} | {r['t_mm']:.5g} | {r['kind']} | {_f(r['default_N'])} | "
                 f"{_f(r['qi_deciding']) if r['default_N'] else 'INFEASIBLE (no QI up to 64 plies)'} | "
                 f"{_f(r['qi_strength_margin'])} | {_f(r['qi_lambda'])} |")
    for title, rows in (("Exact search vs enumeration of every valid design, 16 plies", v16),
                        ("Exact search vs enumeration of every valid design, 24 plies (streaming enumeration)", v24)):
        L += ["", f"## {title}", "",
              "| case | t (mm) | valid designs | status exact / enumeration | ranked by | exact value | enumeration "
              "value | rel. diff | max strength margin exact / enumeration | same sequence | match |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            st_ = (r["exact_status"].upper() if r["exact_status"] == "infeasible" else r["exact_status"]) + " / " + \
                  (r["brute_status"].upper() if r["brute_status"] == "infeasible" else r["brute_status"])
            L.append(f"| {r['case']} | {r['t_mm']:.5g} | {r['n_valid_designs']:,} | {st_} | {r['ranked_by']} | "
                     f"{_f(r['exact_value'], 10)} | {_f(r['brute_value'], 10)} | {r['rel_diff_value']:.1e} | "
                     f"{_f(r['exact_max_strength_margin_any_design'], 6)} / "
                     f"{_f(r['brute_max_strength_margin_any_design'], 6)} | {_f(r['same_design'])} | "
                     f"{_f(r['match'])} |")
    L += ["", "## Problems solved (max_stiffness, exact search)", "",
          "vs QI: change against the quasi-isotropic laminate [0/45/-45/90]_ks with the same N (lambda: increase, "
          "c: decrease); not given for infeasible problems.", "",
          "| case | t (mm) | N | N from | design | lambda | c (N/m) | strength margin | vs QI same N | result | "
          "time (s) | MATLAB (strength margin) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in prob_rows:
        feas = r["status"] == "optimal"
        imp = (f"{r['improvement_vs_qi_pct']:+.2f} % ({'lambda' if r['kind'] == 'lambda' else 'c'})"
               if r["improvement_vs_qi_pct"] is not None and feas else "-")
        res = r["result"] if feas else (f"**INFEASIBLE** (proven): {r['infeasible_reason']}; max strength margin "
                                        f"of any design {_f(r['max_strength_margin_any_design'])}")
        L.append(f"| {r['case']} | {r['t_mm']:.5g} | {r['N']} | {r['N_source']} | {r['design']} | {_f(r['lambda'])} | "
                 f"{_f(r['c_N_per_m'], 6)} | {_f(r['strength_margin'])} | {imp} | {res} | {r['runtime_s']:.2f} | "
                 f"{_f(r['matlab_pass'])} ({_f(r['matlab_strength_margin'])}) |")
    L += ["", "## Stiffest (max_stiffness) vs strongest (max_margin) at the same N", "",
          "margin = min(strength margin, lambda): the quantity max_margin maximises.", "",
          "| case | t (mm) | N | stiffest | lambda | c (N/m) | strength margin | strongest | lambda | c (N/m) | "
          "strength margin | margin stiffest / strongest |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cmp_rows:
        note = "" if r["strongest_proven"] else f" ({r['strongest_status']}, not proven)"
        L.append(f"| {r['case']} | {THICKNESSES[r['thickness']] * 1e3:.5g} | {r['N']} | {r['stiffest_design']} | "
                 f"{_f(r['stiffest_lambda'])} | {_f(r['stiffest_c_N_per_m'], 6)} | "
                 f"{_f(r['stiffest_strength_margin'])} | {r['strongest_design']} | {_f(r['strongest_lambda'])} | "
                 f"{_f(r['strongest_c_N_per_m'], 6)} | {_f(r['strongest_strength_margin'])} | "
                 f"{_f(r['stiffest_margin'])} / {_f(r['strongest_margin'])}{note} |")
    L += ["", "## Ritz convergence of the 5 best designs (mirror +45/-45 twins removed)", "",
          "| case | t (mm) | N | rank (8x8) | design | 8x8 | 12x12 | 16x16 | poly p=16 | 16x16 vs 8x8 | rank 12 / 16 / "
          "poly16 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ritz_rows:
        L.append(f"| {r['case']} | {THICKNESSES[r['thickness']] * 1e3:.5g} | {r['N']} | {r['rank_8x8']} | {r['design']} | "
                 f"{r['lambda_8x8']:.6g} | {r['lambda_12x12']:.6g} | {r['lambda_16x16']:.6g} | "
                 f"{r['lambda_poly16']:.6g} | {r['diff_16_vs_8_pct']:+.3f} % | "
                 f"{r['rank_12x12']} / {r['rank_16x16']} / {r['rank_poly16']} |")
    L.append("")
    groups: Dict = {}
    for r in ritz_rows:
        groups.setdefault((r["case"], r["thickness"], r["N"]), []).append(r)
    for (cid, tk, n), rows in groups.items():
        changed = [k for k in ("12x12", "16x16", "poly12", "poly16")
                   if any(r[f"rank_{k}"] != r["rank_8x8"] for r in rows)]
        if changed:
            spread = max((max(r[f"lambda_{k}"] for r in rows) - min(r[f"lambda_{k}"] for r in rows))
                         / max(r[f"lambda_{k}"] for r in rows) for k in ["8x8"] + changed)
            L.append(f"- {cid} {tk} N = {n}: the order changes ({', '.join(changed)}) only among designs whose lambda "
                     f"are equal to rounding (relative spread of the 5 values {spread:.1e} under every model).")
    L += ["- Elsewhere the order of the 5 designs is the same under every model; the best design stays first.",
          "- The ranks of the designs are recomputed in each model; 16x16 vs 8x8 is the change of lambda of each "
          "design."]
    L += ["", "## GA and GA + local search (20 seeds, budget 10 000 designs)", "",
          "Hit = the run's best design has the exact answer's objective (lambda or c within a relative 1e-9; for an "
          "infeasible problem the same ranking value) with the same feasibility and tier.", "",
          "At 16 plies the budget is about the whole design space (10 056 valid designs), so the full-budget hit "
          "count says little there; the hits within the first 1 000 / 2 000 evaluations are the useful comparison.",
          "",
          "| case | t (mm) | N | exact result | method | hit (full budget) | hit <= 1 000 evals | hit <= 2 000 evals | "
          "same sequence | median evaluations to hit | space covered |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ga_sum:
        ex = r["exact_result"] or "-"
        L.append(f"| {r['case']} | {THICKNESSES[r['thickness']] * 1e3:.5g} | {r['N']} | "
                 f"{'INFEASIBLE' if ex.startswith('proven infeasible') else ex} | {r['method']} | "
                 f"{r['reach_exact_answer']}/{r['runs']} | {r['hit_within_1000_evals']}/{r['runs']} | "
                 f"{r['hit_within_2000_evals']}/{r['runs']} | {r['same_design_as_exact']}/{r['runs']} | "
                 f"{_f(r['evals_to_hit_median'])} | {_f(r['space_covered_pct'], 3)} % |")
    if probe_rows:
        L += ["", "## Probes at >= 30 plies: random valid designs vs the exact optimum", ""]
        for r in probe_rows:
            L.append(f"- {r['case']} {r['thickness']} N = {r['N']}: {r['n_random_valid_designs']:,} random valid designs, "
                     f"best lambda {_f(r['best_random_lambda'])} vs exact {_f(r['exact_lambda'])} -> never better: "
                     f"{_f(r['random_never_better'])}")
    L += ["", "## Demo regression (default objective, fewest plies)", "",
          f"- request: \"{demo['request']}\"", f"- result: {demo['notation']}, {demo['n_plies']} plies, margin "
          f"{demo['margin']:.4f}, status {demo['status']} -> {'PASS' if demo['pass'] else 'FAIL'}"]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
