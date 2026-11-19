"""
stiffness.py - fixed ply count, stiffest stacking sequence (objective "max_stiffness").

Problem
-------
Fixed: material, ply thickness t, ply count N, simply supported a x b plate, load case (Nx, Ny, Nxy).
Variables: the stacking sequence (0, +45, -45, 90) under every design rule of rules.py (symmetric, balanced,
>= 10 % of the plies in each family, <= 4 identical plies in a row on the full laminate).
Constraints:
  (a) strength margin  rs = min(RF Max Stress, RF Tsai-Wu) >= safety factor  (depends on the ply counts only);
  (b) buckling factor  lambda >= safety factor  (the Ritz model of the optimisation, buckling_model "ritz_full";
      with shear_both_signs lambda = min over +Nxy and -Nxy). When the load cannot buckle the plate, lambda = inf
      for every laminate and (b) holds (see can_buckle).
Objective, compared in this order:
  1. loads that can buckle the plate: MAXIMISE lambda;
  2. loads that cannot (lambda = inf): MINIMISE the in-plane compliance
         c = N^T a N,   a = A^-1,   N = (Nx, Ny, Nxy) [N/m],  A [N/m]   ->   c [N/m] (= J/m^2)
     c = N . eps0 is twice the membrane strain energy per unit area. With shear_both_signs the larger value over
     +Nxy / -Nxy is used (they are equal for balanced laminates, A16 = A26 = 0). c depends on the ply counts only;
  3. two objective values within a relative 1e-9 are equal; then the larger strength margin wins (within 1e-9),
     then the lexicographically smallest stacking sequence (ply codes 0 = 0 deg, 1 = +45, 2 = -45, 3 = 90, read
     from ply 1 = outer surface to the mid-plane), so that the answer is unique.
Problems with no feasible design follow the same order with the constraints relaxed step by step: among the
designs meeting (a), the largest lambda (resp. the smallest c); if no design meets (a), the largest strength margin.
The status is then "infeasible".

Default ply count (default_ply_count): the smallest quasi-isotropic laminate [0/45/-45/90]_ks (N = 8k,
8 <= N <= 64) that meets (a) and (b) for the load case and the ply thickness; none -> no default (infeasible).

Exact search (exact_stiffness): see the docstring of exact_stiffness for the bounds and why they are valid.
"exact" means: the global optimum of the model that is optimised (Ritz series with problem.ritz_terms terms).
"""
from __future__ import annotations

import time
from dataclasses import replace
from functools import lru_cache
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import buckling
from .evaluate import Evaluator, Problem
from .exhaustive import _FullBounds, _count_tuples, _tuple_data, _valid_mask, arrange
from .rules import PLY, full_laminate, notation, satisfies

RTOL = 1e-9                 # objective values within this relative difference are equal
PRUNE_RTOL = 2e-9           # pruning margin of the branch and bound (keeps every design of the 1e-9 tie window)
COMPLIANCE_UNIT = "N/m"     # c = N^T A^-1 N with N in N/m and A in N/m (= J/m^2)
OBJECTIVE = "max_stiffness"


class NoFeasiblePlyCount(ValueError):
    """No quasi-isotropic laminate with 8 <= N <= 64 meets the constraints: no default ply count."""

    def __init__(self, message: str, scan: List[Dict]):
        super().__init__(message)
        self.scan = scan


# --------------------------------------------------------------------------------------------- problem helpers
def fixed_problem(problem: Problem, n: int) -> Problem:
    """The same problem with the ply count fixed to n (rules.n_min = rules.n_max = n)."""
    return replace(problem, rules=replace(problem.rules, n_min=int(n), n_max=int(n)))


def check_problem(problem: Problem) -> int:
    """The fixed-thickness search needs a symmetric, balanced laminate (ply encoding) with a fixed even ply count;
    returns N."""
    r = problem.rules
    if not (problem.encoding == "ply" and r.symmetric and r.balanced):
        raise ValueError("max_stiffness: symmetric, balanced laminates with the ply encoding")
    if r.n_min != r.n_max:
        raise ValueError("max_stiffness needs a fixed ply count (fixed_plies, or rules.n_min == rules.n_max)")
    n = r.n_min
    if n % 2 or not 8 <= n <= 64:
        raise ValueError(f"fixed ply count {n}: an even number between 8 and 64 is needed")
    if not r.string_lengths(problem.enc):
        raise ValueError(f"no {n}-ply laminate can meet the design rules")
    if "buckling" in problem.criteria and problem.buckling_model != "ritz_full":
        raise ValueError("max_stiffness uses the Ritz buckling model with D16, D26 (buckling_model='ritz_full')")
    return n


def can_buckle(problem: Problem) -> bool:
    """True if the load can buckle the plate in the optimisation model; then lambda is finite for EVERY laminate.
    Why it does not depend on the laminate: lambda = 1 / mu_max, mu_max = largest eigenvalue of (-G) A = mu K A,
    with K (bending stiffness) positive definite and G (load) independent of the laminate. With K = L L^T the
    eigenvalues are those of L^-1 (-G) L^-T, which has the same inertia as -G (Sylvester): a positive mu exists
    for some D iff it exists for every D iff -G has a positive eigenvalue."""
    if "buckling" not in problem.criteria:
        return False
    a, b = problem.plate.a, problem.plate.b
    mn = buckling.terms_for(problem.ritz_terms, a, b)
    for lc in problem.load_cases():
        model = buckling.RitzModel(a, b, *lc, *mn)
        if not model.can_buckle:
            continue
        for blk in model.blocks:
            ev = np.linalg.eigvalsh(blk["mG"])
            if ev.max() > 1e-12 * max(np.abs(ev).max(), 1e-300):
                return True
    return False


def objective_kind(problem: Problem) -> str:
    """'lambda' (maximise the buckling factor) or 'compliance' (minimise c; the load cannot buckle the plate)."""
    return "lambda" if can_buckle(problem) else "compliance"


def qi_half(n: int) -> np.ndarray:
    """Half laminate (outer -> mid-plane) of [0/45/-45/90]_ks, n = 8k."""
    if n % 8:
        raise ValueError("quasi-isotropic [0/45/-45/90]_ks needs n = 8k")
    return np.array([0, 1, 2, 3] * (n // 8), dtype=np.int8)


def evaluate_half(problem: Problem, half: Sequence[int]) -> Dict:
    """Model values of one design (half laminate, outer -> mid-plane): lambda is computed even when the strength
    margin is below the safety factor (no lazy evaluation)."""
    ev = Evaluator(problem)
    ev.lazy = False
    r = ev.compute_plies(np.asarray(half, dtype=np.int8)[None], details=True)
    rf = {c: float(r["rf"][0, i]) for i, c in enumerate(problem.criteria)}
    return {"lam": float(r["lam"][0]), "c": float(r["c"][0]), "rf_strength": float(r["rf_strength"][0]),
            "margin": float(r["margin"][0]), "rf": rf, "A": r["A"][0], "D": r["D"][0]}


def default_ply_count(problem: Problem, n_min: int = 8, n_max: int = 64) -> Dict:
    """Smallest N = 8k in [n_min, n_max] for which [0/45/-45/90]_ks meets the strength and buckling constraints.
    Returns {"found", "n", "qi", "rf_strength", "lam", "scan"}; found = False -> no default ply count."""
    sf = problem.safety_factor
    scan = []
    for n in range(8, 65, 8):
        if not n_min <= n <= n_max:
            continue
        m = evaluate_half(fixed_problem(problem, n), qi_half(n))
        ok = m["rf_strength"] >= sf and m["lam"] >= sf
        scan.append({"n": n, "qi": f"[0/45/-45/90]_{n // 8}s", "rf_strength": m["rf_strength"], "lam": m["lam"],
                     "c": m["c"], "meets": bool(ok)})
        if ok:
            return {"found": True, "n": n, "qi": scan[-1]["qi"], "rf_strength": m["rf_strength"], "lam": m["lam"],
                    "scan": scan}
    return {"found": False, "n": None, "qi": None, "rf_strength": None, "lam": None, "scan": scan}


def resolve_ply_count(problem: Problem, fixed_plies: Optional[int]) -> Tuple[Problem, Dict]:
    """Problem with the ply count fixed: fixed_plies if given, else rules.n_min == n_max if already fixed, else the
    default (quasi-isotropic rule). Raises NoFeasiblePlyCount when there is no default."""
    if fixed_plies is not None:
        p = fixed_problem(problem, fixed_plies)
        check_problem(p)
        return p, {"source": "given", "n": int(fixed_plies)}
    if problem.rules.n_min == problem.rules.n_max:
        check_problem(problem)
        return problem, {"source": "rules", "n": problem.rules.n_min}
    d = default_ply_count(problem)
    if not d["found"]:
        raise NoFeasiblePlyCount("INFEASIBLE: no quasi-isotropic laminate [0/45/-45/90]_ks with 8 <= N <= 64 meets "
                                 "the strength and buckling constraints, so there is no default ply count "
                                 "(give fixed_plies explicitly)", d["scan"])
    p = fixed_problem(problem, d["n"])
    check_problem(p)
    return p, dict(d, source="default (smallest quasi-isotropic laminate meeting the constraints)")


# ---------------------------------------------------------------------------------------------- the order
class Key(NamedTuple):
    """Comparison key of one design: feasible first; then tier 0 (strength met) before tier 1; in tier 0 the
    objective score (lambda, or -c) then the strength margin; in tier 1 the strength margin; then the sequence."""
    feasible: bool
    tier: int
    primary: float
    secondary: float
    lex: Tuple[int, ...]


def score(kind: str, lam, c):
    """Objective to maximise: lambda, or -c."""
    return lam if kind == "lambda" else -np.asarray(c)


def make_key(kind: str, sf: float, rs: float, lam: float, c: float, half: Sequence[int]) -> Key:
    tier = 0 if rs >= sf else 1
    s = float(score(kind, lam, c))
    return Key(bool(rs >= sf and lam >= sf), tier, s if tier == 0 else float(rs), float(rs) if tier == 0 else 0.0,
               tuple(int(x) for x in half if x >= 0))


def _ranked_value(kind: str, sf: float, rs: float, lam: float, c: float) -> Tuple[float, str]:
    """The value that decided the ranking of a design: lambda or c when the strength constraint is met, else the
    strength margin (no design meets strength: the largest strength margin is reported)."""
    if rs >= sf:
        return (float(lam), "lambda") if kind == "lambda" else (float(c), "c [N/m]")
    return float(rs), "strength margin (no design meets the strength constraint)"


def infeasible_reason(sf: float, max_rs: float, lam_best: Optional[float], feasible: bool) -> Optional[str]:
    """Why no design of N plies meets both constraints (None when the best design is feasible).
    max_rs: largest strength margin of any valid design; lam_best: largest lambda among the designs that meet the
    strength constraint (None when there is none)."""
    if feasible:
        return None
    if max_rs < sf:
        return f"no design meets the strength constraint: largest strength margin {max_rs:.4g} < {sf:g}"
    return f"buckling: largest lambda among the designs meeting strength {lam_best:.4g} < {sf:g}"


def _close(x: float, y: float, rtol: float) -> bool:
    return x == y or abs(x - y) <= rtol * max(abs(x), abs(y))


def key_better(a: Key, b: Key, rtol: float = RTOL) -> bool:
    """True if design a is strictly better than design b (values within rtol are equal)."""
    if a.feasible != b.feasible:
        return a.feasible
    if a.tier != b.tier:
        return a.tier < b.tier
    if not _close(a.primary, b.primary, rtol):
        return a.primary > b.primary
    if not _close(a.secondary, b.secondary, rtol):
        return a.secondary > b.secondary
    return a.lex < b.lex


def reaches(key: Key, target: Key, rtol: float = RTOL) -> bool:
    """True if a design reaches the objective value of target (e.g. a proven optimum): same feasibility and tier
    (or better) and an objective value within rtol of it or better (strength margin and sequence not compared)."""
    if (key.feasible, -key.tier) != (target.feasible, -target.tier):
        return (key.feasible, -key.tier) > (target.feasible, -target.tier)
    return key.primary >= target.primary - rtol * abs(target.primary)


def rank_order(res: Dict[str, np.ndarray], sf: float, kind: str) -> np.ndarray:
    """Order (best first) of a batch evaluated with Evaluator(objective='max_stiffness'): the order of Key without
    the 1e-9 tolerance and without the sequence (stable: ties keep their order). Used for GA selection."""
    rs, lam = res["rf_strength"], res["lam"]
    tier = (rs < sf).astype(int)
    feas = (rs >= sf) & (lam >= sf)
    s = score(kind, lam, res["c"])
    primary = np.where(tier == 0, s, rs)
    secondary = np.where(tier == 0, rs, 0.0)
    return np.lexsort((-secondary, -primary, tier, ~feas))


def _window(idx: np.ndarray, values: np.ndarray, rtol: float = RTOL) -> np.ndarray:
    """Members of idx whose value is within rtol of the largest value."""
    vmax = values.max()
    return idx[values >= vmax - rtol * abs(vmax)]


def select(rs, lam, c, halves, kind: str, sf: float) -> int:
    """Index of the best design of a set (the order of the module doc, with the 1e-9 tie window)."""
    rs, lam, c = (np.asarray(v, dtype=float) for v in (rs, lam, c))
    halves = np.asarray(halves)
    s = score(kind, lam, c)
    feas = (rs >= sf) & (lam >= sf)
    t0 = rs >= sf
    if feas.any() or t0.any():
        idx = np.where(feas if feas.any() else t0)[0]
        keep = _window(idx, s[idx])
        keep = _window(keep, rs[keep])
    else:
        idx = np.arange(len(rs))
        keep = _window(idx, rs[idx])
    order = np.lexsort(halves[keep].T[::-1])            # first ply = primary key
    return int(keep[order[0]])


# -------------------------------------------------------------------------------- arrangements of ply counts
def lex_smallest_arrangement(problem: Problem, counts: Sequence[int]) -> Optional[np.ndarray]:
    """Lexicographically smallest half laminate (outer -> mid-plane) with the given counts of the ply codes 0, +45,
    -45, 90 that meets the contiguity rule on the full laminate (runs <= max_contiguous; the run touching the
    mid-plane is mirrored, so it may hold at most max_contiguous // 2 plies of the half); None if there is none."""
    rl, el = problem.rules.run_limits(PLY)
    L = int(sum(counts))
    rl = rl or L
    el = el if el is not None else rl

    @lru_cache(maxsize=None)
    def can(rem: Tuple[int, ...], last: int, run: int) -> bool:
        if sum(rem) == 0:
            return run <= el
        for s in range(4):
            if rem[s] and (s != last or run + 1 <= rl):
                if can(rem[:s] + (rem[s] - 1,) + rem[s + 1:], s, run + 1 if s == last else 1):
                    return True
        return False

    rem, last, run = tuple(int(x) for x in counts), -1, 0
    if not can(rem, last, run):
        return None
    out = []
    for _ in range(L):
        for s in range(4):
            if rem[s] and (s != last or run + 1 <= rl):
                nrem, nrun = rem[:s] + (rem[s] - 1,) + rem[s + 1:], (run + 1 if s == last else 1)
                if can(nrem, s, nrun):
                    out.append(s)
                    rem, last, run = nrem, s, nrun
                    break
    return np.array(out, dtype=np.int8)


def _count_level(problem: Problem, tuples: List[Tuple[int, ...]]) -> Tuple[np.ndarray, np.ndarray]:
    """Strength margin and in-plane compliance of every count tuple (both depend on the counts only)."""
    strength = replace(problem, criteria=tuple(c for c in problem.criteria if c != "buckling"))
    reps = np.array([np.repeat(np.arange(4), c) for c in tuples], dtype=np.int8)
    r = Evaluator(strength).compute_plies(reps, details=True)
    return r["rf_strength"], r["c"]


def max_strength_margin(problem: Problem) -> float:
    """Largest strength margin of any valid design of the fixed ply count (exact: it depends on the ply counts
    only, so the count tuples that can be arranged within the rules are enough)."""
    L = check_problem(problem) // 2
    tuples = [tuple(int(x) for x in c) for c in _count_tuples(problem, L) if arrange(problem, c) is not None]
    return float(np.max(_count_level(problem, tuples)[0]))


def _lam_nocache(ev: Evaluator, Dv: np.ndarray) -> np.ndarray:
    lam = ev.ritz_models[0].lam(Dv)
    for m in ev.ritz_models[1:]:
        lam = np.minimum(lam, m.lam(Dv))
    return lam


# ------------------------------------------------------------------------------------------- exact search
class PrefixBound:
    """Upper bound of lambda for every completion of a partial design (points 2-4 of exact_stiffness).
    vectors: quotient vectors r (each a valid bound lambda(D) <= r . Dvec); convex combinations of pairs of the
    first six are added (also valid). S = sum over the placed positions k of w_k g_j(s_k) (one column per vector)."""

    def __init__(self, fb: _FullBounds, vectors: Sequence[np.ndarray]):
        uniq: List[np.ndarray] = []
        for v in vectors:
            if not any(np.allclose(v, u, rtol=1e-12, atol=0) for u in uniq):
                uniq.append(np.asarray(v, dtype=float))
        base = uniq[:6]
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                uniq.append(0.5 * (base[i] + base[j]))
        self.Rm = np.array(uniq) / fb.factor                         # (J, 6)
        self.G = fb.q6 @ self.Rm.T                                    # (4, J): g_j(s)
        self.perm = np.argsort(-self.G, axis=0, kind="stable")        # symbols by decreasing g_j
        self.Gp = self.G[self.perm, np.arange(self.G.shape[1])[None, :]]
        self.J = self.Rm.shape[0]
        self.w = fb.w                                                 # decreasing, outer -> mid-plane
        self.Wc = np.concatenate([[0.0], np.cumsum(fb.w)])

    def partial(self, halves: np.ndarray, k: int) -> np.ndarray:
        """S of the first k plies of full designs (P, L)."""
        S = np.zeros((len(halves), self.J))
        for i in range(k):
            S += self.w[i] * self.G[halves[:, i]]
        return S

    def bound(self, S: np.ndarray, rem: np.ndarray, k: int) -> np.ndarray:
        """min over the vectors of S + best rearrangement of the remaining plies rem (P, 4) on positions k..L-1."""
        out = S.copy()
        start = np.full((len(rem), self.J), k)
        for m in range(4):
            end = start + rem[:, self.perm[m]]
            out += self.Gp[m][None, :] * (self.Wc[end] - self.Wc[start])
            start = end
        return out.min(axis=1)


class _Pool:
    """Designs found so far whose lambda can still be among the top_k (or in the tie window of the best)."""

    def __init__(self, top_k: int):
        self.k = max(1, int(top_k))
        self.items: Dict[bytes, Tuple[float, float, np.ndarray]] = {}
        self.thr = -np.inf

    def add(self, halves: np.ndarray, lams: np.ndarray, rs: float) -> None:
        for h, lv in zip(halves, lams):
            if lv >= self.thr:
                self.items[h.tobytes()] = (float(lv), float(rs), np.array(h, dtype=np.int8))
        if len(self.items) >= self.k:
            vals = np.sort([v[0] for v in self.items.values()])[::-1]
            kth = vals[self.k - 1]
            self.thr = kth - PRUNE_RTOL * abs(kth)
            self.items = {key: v for key, v in self.items.items() if v[0] >= self.thr}

    def best(self) -> Optional[Tuple[float, float, np.ndarray]]:
        return max(self.items.values(), key=lambda v: v[0]) if self.items else None


def exact_stiffness(problem: Problem, time_limit: float = 600.0, top_k: int = 1, block: int = 1 << 16,
                    verbose: bool = False) -> Dict:
    """Exact answer of the fixed-thickness stiffest-laminate problem (symmetric, balanced, ply encoding, fixed N).

    Compliance (the load cannot buckle the plate): c and the strength margin depend on the ply counts only, so the
    count tuples (n0, n+45, n-45, n90) that satisfy balance and the 10 % rule and can be arranged within the
    contiguity rule (exact DP, exhaustive.arrange) are enumerated: smallest c among the tuples with rs >= safety
    factor, ties by rs, then the lexicographically smallest valid arrangement of the tied tuples. Always complete.

    Buckling (lambda): branch and bound with PROVEN upper bounds of lambda. Why they are valid:
      1. The Ritz factor is lambda(D) = min over mode vectors A with A^T (-G) A > 0 of the Rayleigh quotient
         R_A(D) = A^T K(D) A / A^T (-G) A (lambda = 1 / mu_max of (-G) A = mu K A, K positive definite). With both
         shear signs lambda is the smaller of two such minima. So for ANY fixed A of either load case,
         lambda(D) <= R_A(D) for every laminate.
      2. K(D) is linear in Dvec = (D11, D12, D22, D66, D16, D26), so R_A(D) = r_A . Dvec (buckling.RitzModel.
         quotient_terms); a convex combination of such vectors is again a valid bound (min_j R_j <= sum mu_j R_j).
      3. For a symmetric laminate Dvec = sum_k w_k q6(s_k) (w_k > 0: weight of the mirrored ply pair at position k,
         decreasing from the outer surface to the mid-plane; q6(s): Qbar terms of ply code s), so
         R_A = sum_k w_k g_A(s_k) with g_A(s) = r_A . q6(s).
      4. With the plies of positions 0..k-1 fixed and the multiset M of the remaining plies, every completion has
         R_A <= (fixed part) + (values g_A of M sorted decreasingly) . (remaining weights, decreasing)
         (rearrangement inequality; the contiguity rule only removes completions). The bound of a node is the
         minimum of this over the vectors used; the count-level bound (exhaustive._tuple_data) is the case k = 0.
      5. A node (count tuple, prefix or full design) is discarded only if its bound is below
         (k-th best lambda found) x (1 - 2e-9): no discarded design can be among the top_k or inside the 1e-9 tie
         window of the optimum. Eigenvalues are solved only for full designs whose bound passes this test.
    The vectors r_A come from the critical modes of trial designs of each count tuple and of the best design found.
    Count tuples whose strength margin is below the safety factor are excluded first (rs depends on the counts).
    The search runs depth first over the plies (outer -> mid-plane) in blocks, so the 3 000 000-arrangement cap of
    exhaustive.exact_full does not apply. If the time limit is reached the result is INCOMPLETE: the best design
    found, the largest bound left open ("upper_bound") and the relative gap; it is never reported as proven.

    Returns {"status": "optimal" | "infeasible" | "incomplete", "complete", "proven", "kind", "full", "half",
    "lam", "c", "rf_strength", "value", "upper_bound", "gap", "top", ...}."""
    n = check_problem(problem)
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    sf = problem.safety_factor
    L = n // 2
    kind = objective_kind(problem)
    tuples = [tuple(int(x) for x in c) for c in _count_tuples(problem, L) if arrange(problem, c) is not None]
    if not tuples:
        raise ValueError(f"no {n}-ply count tuple meets the design rules")
    rs_t, c_t = _count_level(problem, tuples)
    base = {"objective": OBJECTIVE, "kind": kind, "n": n, "n_count_tuples": len(tuples),
            "n_tuples_strength_ok": int(np.sum(rs_t >= sf))}
    if kind == "compliance" or not np.any(rs_t >= sf):
        out = _exact_counts(problem, tuples, rs_t, c_t, kind, top_k)
    else:
        out = _exact_bb(problem, L, kind, top_k, deadline, block, verbose)
    out.update(base)
    # the strength margin depends on the ply counts only: its largest value over every valid design
    out["max_rf_strength"] = float(np.max(rs_t))
    out["infeasible_reason"] = (None if out["status"] == "incomplete" else infeasible_reason(
        sf, out["max_rf_strength"], out["lam"] if out["rf_strength"] >= sf else None, out["feasible"]))
    out["time_s"] = time.perf_counter() - t0
    full = full_laminate(np.asarray(out["half"], dtype=np.int8)[None], True)[0]
    assert satisfies(full, problem.rules)
    out["full"] = full.tolist()
    out["notation"] = notation(full, True)
    return out


def _exact_counts(problem: Problem, tuples, rs_t: np.ndarray, c_t: np.ndarray, kind: str, top_k: int) -> Dict:
    sf = problem.safety_factor
    t0 = rs_t >= sf
    if kind == "compliance" and t0.any():
        idx = np.where(t0)[0]
        keep = _window(idx, -c_t[idx])
        keep = _window(keep, rs_t[keep])
        method = "count-level enumeration (c and the strength margin depend on the ply counts only)"
    else:   # no count tuple meets the strength constraint: the design with the largest strength margin
        idx = np.arange(len(tuples))
        keep = _window(idx, rs_t[idx])
        method = "count-level enumeration (no ply-count combination meets the strength constraint)"
    arrs = {int(i): lex_smallest_arrangement(problem, tuples[i]) for i in keep}
    i_best = min(arrs, key=lambda i: tuple(arrs[i].tolist()))
    half = arrs[i_best]
    m = evaluate_half(problem, half)
    feasible = bool(m["rf_strength"] >= sf and m["lam"] >= sf)
    # ranked count tuples (information): tier 0 by c (or rs), then the others by rs
    order = sorted(range(len(tuples)), key=lambda i: (rs_t[i] < sf, c_t[i] if (kind == "compliance" and rs_t[i] >= sf)
                                                      else -rs_t[i]))
    top = []
    for i in order[:max(top_k, 5)]:
        h = lex_smallest_arrangement(problem, tuples[i])
        top.append({"counts": tuples[i], "half": h.tolist(), "c": float(c_t[i]), "rf_strength": float(rs_t[i]),
                    "notation": notation(full_laminate(h[None], True)[0], True)})
    value, meaning = _ranked_value(kind, sf, m["rf_strength"], m["lam"], m["c"])
    return {"status": "optimal" if feasible else "infeasible", "complete": True, "proven": True,
            "feasible": feasible, "half": half.tolist(), "lam": m["lam"], "c": m["c"], "rf_strength": m["rf_strength"],
            "value": value, "value_meaning": meaning, "upper_bound": None, "gap": 0.0, "top": top, "method": method,
            "counts": tuples[i_best], "n_tied_count_tuples": len(keep)}


def _exact_bb(problem: Problem, L: int, kind: str, top_k: int, deadline: float, block: int, verbose: bool) -> Dict:
    sf = problem.safety_factor
    ev = Evaluator(problem)
    fb, data = _tuple_data(problem, ev, L)
    data = sorted((d for d in data if d["rf_strength"] >= sf), key=lambda d: -d["lam_ub"])
    rl, el = problem.rules.run_limits(PLY)
    rl = rl or L
    el = el if el is not None else rl
    pool = _Pool(top_k)
    stats = {"n_eigen": 0, "n_leaves": 0, "n_nodes": 0}

    def solve(halves: np.ndarray, rs: float) -> None:
        if len(halves):
            lam = _lam_nocache(ev, fb.dvec(halves)) / fb.factor
            stats["n_eigen"] += len(halves)
            pool.add(halves, lam, rs)

    # 1) trial designs of every count tuple (valid ones): a good first threshold
    for d in data:
        tr = d["trials"]
        solve(tr[_valid_mask(problem, tr)] if len(tr) else tr, d["rf_strength"])
    glob: List[np.ndarray] = []
    best_seen = None

    def global_vectors() -> List[np.ndarray]:
        nonlocal best_seen
        b = pool.best()
        if b is not None and (best_seen is None or b[2].tobytes() != best_seen):
            best_seen = b[2].tobytes()
            glob.extend(fb.modes(b[2][None]))
            del glob[:-8]
        return glob

    w = fb.w
    open_bounds: List[float] = []
    per_tuple = []
    complete = True
    for ti, d in enumerate(data):
        if d["lam_ub"] < pool.thr:
            per_tuple.append({"counts": d["counts"], "lam_ub": d["lam_ub"], "pruned_at_count_level": True})
            continue
        if time.perf_counter() > deadline:
            complete = False
            open_bounds += [x["lam_ub"] for x in data[ti:] if x["lam_ub"] >= pool.thr]
            break
        pb = PrefixBound(fb, list(d["R"]) + global_vectors())
        G, J, bound = pb.G, pb.J, pb.bound
        rem0 = np.array([d["counts"]], dtype=np.int16)
        S0 = np.zeros((1, J))
        b0 = bound(S0, rem0, 0)
        stack = [(np.zeros((1, 0), np.int8), rem0, np.full(1, -1, np.int16), np.zeros(1, np.int16), S0, b0)]
        done = True
        while stack:
            if time.perf_counter() > deadline:
                done = False
                break
            P, rem, last, run, S, bnd = stack.pop()
            keep = bnd >= pool.thr
            if not keep.any():
                continue
            P, rem, last, run, S = P[keep], rem[keep], last[keep], run[keep], S[keep]
            k = P.shape[1]
            stats["n_nodes"] += len(P)
            parts = []
            for s in range(4):
                ok = (rem[:, s] > 0) & ~((last == s) & (run >= rl))
                nrun = np.where(last == s, run + 1, 1).astype(np.int16)
                if k + 1 == L:
                    ok &= nrun <= el                    # the run touching the mid-plane is mirrored
                if not ok.any():
                    continue
                r2 = rem[ok].copy()
                r2[:, s] -= 1
                parts.append((np.concatenate([P[ok], np.full((int(ok.sum()), 1), s, np.int8)], axis=1), r2,
                              np.full(int(ok.sum()), s, np.int16), nrun[ok], S[ok] + w[k] * G[s][None, :]))
            if not parts:
                continue
            P2 = np.concatenate([p[0] for p in parts])
            r2 = np.concatenate([p[1] for p in parts])
            l2 = np.concatenate([p[2] for p in parts])
            u2 = np.concatenate([p[3] for p in parts])
            S2 = np.concatenate([p[4] for p in parts])
            b2 = bound(S2, r2, k + 1)
            keep = b2 >= pool.thr
            if not keep.any():
                continue
            P2, r2, l2, u2, S2, b2 = P2[keep], r2[keep], l2[keep], u2[keep], S2[keep], b2[keep]
            order = np.argsort(-b2, kind="stable")
            if k + 1 == L:                              # full designs: eigenvalues by decreasing bound
                assert _valid_mask(problem, P2).all()   # the rules hold by construction (counts, runs)
                stats["n_leaves"] += len(P2)
                for pos in range(0, len(order), 512):
                    idx = order[pos:pos + 512]
                    idx = idx[b2[idx] >= pool.thr]
                    if len(idx) == 0:
                        break
                    solve(P2[idx], d["rf_strength"])
                continue
            chunks = [order[i:i + block] for i in range(0, len(order), block)]
            for idx in reversed(chunks):                # highest bounds on top of the stack
                stack.append((P2[idx], r2[idx], l2[idx], u2[idx], S2[idx], b2[idx]))
        if not done:
            complete = False
            open_bounds += [float(x[5].max()) for x in stack if len(x[5]) and x[5].max() >= pool.thr]
            open_bounds += [x["lam_ub"] for x in data[ti + 1:] if x["lam_ub"] >= pool.thr]
            per_tuple.append({"counts": d["counts"], "lam_ub": d["lam_ub"], "complete": False})
            break
        per_tuple.append({"counts": d["counts"], "lam_ub": d["lam_ub"], "complete": True})
        if verbose:
            b = pool.best()
            print(f"  exact_stiffness: tuple {d['counts']} ub={d['lam_ub']:.6g} best={b[0] if b else None:.6g} "
                  f"nodes={stats['n_nodes']:,} eig={stats['n_eigen']:,}")
    items = list(pool.items.values())
    lam = np.array([x[0] for x in items])
    rs = np.array([x[1] for x in items])
    halves = np.array([x[2] for x in items])
    ib = select(rs, lam, np.zeros(len(items)), halves, "lambda", sf)
    best_lam, best_rs, best_half = items[ib]
    m = evaluate_half(problem, best_half)
    assert _close(m["lam"], best_lam, 1e-9), (m["lam"], best_lam)
    feasible = bool(best_lam >= sf and best_rs >= sf)
    ub = max([best_lam] + open_bounds) if not complete else best_lam
    # ranked list (the selected best first, then by lambda, strength margin, sequence; no tolerance)
    order = [ib] + [int(i) for i in np.lexsort(list(halves.T[::-1]) + [-rs, -lam]) if i != ib]
    top = [{"half": halves[i].tolist(), "lam": float(lam[i]), "rf_strength": float(rs[i]),
            "notation": notation(full_laminate(halves[i][None], True)[0], True)} for i in order[:top_k]]
    status = ("optimal" if feasible else "infeasible") if complete else "incomplete"
    return {"status": status, "complete": complete, "proven": complete, "feasible": feasible,
            "half": best_half.tolist(), "lam": best_lam, "c": m["c"], "rf_strength": best_rs, "value": best_lam,
            "value_meaning": "lambda",
            "upper_bound": ub, "gap": (ub - best_lam) / abs(best_lam) if best_lam else None, "top": top,
            "per_tuple": per_tuple, **stats,
            "method": "branch and bound over the plies with proven Rayleigh-quotient / rearrangement bounds"}


# ------------------------------------------------------------------------- brute force (verification only)
def brute_force(problem: Problem, part: int = 0, n_parts: int = 1, top_k: int = 10, chunk: int = 1 << 15) -> Dict:
    """Enumerate EVERY valid design and evaluate it, streaming (verification only, no bound).

    The 4^L ply strings of the half laminate are generated in blocks of `chunk` consecutive base-4 numbers (first
    ply = most significant digit), filtered by the rule check (exhaustive._valid_mask: balance, 10 % rule,
    contiguity with the mirrored mid-plane run), evaluated, and reduced at once to the candidates (_reduce: the
    top_k designs of the order and every design within 2e-9 of the k-th). Only the candidates and a few counters are
    kept, so the memory does not grow with the number of designs. It uses neither the count tuples nor the
    arrangement generator of the exact search. Blocks b with b % n_parts == part form one part. Practical up to
    L = 12 (24 plies: 16.8 million strings, 1 864 416 valid designs).
    Returns the candidates for brute_force_merge, the number of valid designs, how many meet the strength
    constraint and the largest strength margin of any valid design."""
    n = check_problem(problem)
    L = n // 2
    if L > 13:
        raise ValueError(f"brute force over 4^{L} strings is not practical")
    sf = problem.safety_factor
    kind = objective_kind(problem)
    strength = replace(problem, criteria=tuple(c for c in problem.criteria if c != "buckling"))
    evs = Evaluator(strength)
    ev = Evaluator(problem)
    pw = 4 ** np.arange(L - 1, -1, -1, dtype=np.int64)
    total = 4 ** L
    cand = None
    n_valid = n_ok = n_blocks = 0
    max_rs = -np.inf
    for b, s in enumerate(range(0, total, chunk)):
        if b % n_parts != part:
            continue
        n_blocks += 1
        idx = np.arange(s, min(s + chunk, total), dtype=np.int64)
        h = ((idx[:, None] // pw) % 4).astype(np.int8)
        h = h[_valid_mask(problem, h)]
        if not len(h):
            continue
        n_valid += len(h)
        r = evs.compute_plies(h, details=True)
        rs, c = r["rf_strength"], r["c"]
        max_rs = max(max_rs, float(rs.max()))
        ok = np.where(rs >= sf)[0]
        n_ok += len(ok)
        lam = np.full(len(h), np.inf)
        if kind == "lambda":             # lambda is needed only where the strength constraint holds
            for j in range(0, len(ok), 4096):
                sl = ok[j:j + 4096]
                lam[sl] = _lam_nocache(ev, buckling.d_vector(r["D"][sl]))
        new = (h, rs, lam, c)
        if cand is not None:
            new = tuple(np.concatenate([x, y]) for x, y in zip(cand, new))
        cand = _reduce(*new, kind, sf, top_k)
    if cand is None:
        cand = (np.zeros((0, L), np.int8), np.zeros(0), np.zeros(0), np.zeros(0))
    H, rs, lam, c = cand
    return {"n_valid": n_valid, "n_strength_ok": n_ok, "max_rf_strength": max_rs, "n_blocks": n_blocks,
            "halves": H, "rs": rs, "lam": lam, "c": c, "kind": kind}


def _reduce(H, rs, lam, c, kind: str, sf: float, top_k: int):
    """The top_k designs of the order and every design within 2e-9 of the k-th one (same feasibility / tier)."""
    tier = (rs < sf).astype(int)
    feas = (rs >= sf) & (lam >= sf)
    s = score(kind, lam, c)
    primary = np.where(tier == 0, s, rs)
    order = np.lexsort((-primary, tier, ~feas))
    k = order[min(top_k, len(order)) - 1]
    pk = primary[k]
    same = (feas == feas[k]) & (tier == tier[k]) & (primary >= pk - PRUNE_RTOL * abs(pk))
    sel = np.union1d(order[:top_k], np.where(same)[0])
    return H[sel], rs[sel], lam[sel], c[sel]


def brute_force_merge(problem: Problem, parts: List[Dict], top_k: int = 10) -> Dict:
    """Best design of the union of brute_force parts (same order and tie rule as the exact search)."""
    sf = problem.safety_factor
    kind = parts[0]["kind"]
    H = np.concatenate([p["halves"] for p in parts])
    rs = np.concatenate([p["rs"] for p in parts])
    lam = np.concatenate([p["lam"] for p in parts])
    c = np.concatenate([p["c"] for p in parts])
    i = select(rs, lam, c, H, kind, sf)
    feasible = bool(rs[i] >= sf and lam[i] >= sf)
    tier = (rs < sf).astype(int)
    feas = (rs >= sf) & (lam >= sf)
    primary = np.where(tier == 0, score(kind, lam, c), rs)
    order = [i] + [int(j) for j in np.lexsort(list(H.T[::-1]) + [-rs, -primary, tier, ~feas]) if j != i]
    top = [{"half": H[j].tolist(), "lam": float(lam[j]), "c": float(c[j]), "rf_strength": float(rs[j]),
            "notation": notation(full_laminate(H[j][None], True)[0], True)} for j in order[:top_k]]
    lam_i = float(lam[i])
    if kind == "lambda" and rs[i] < sf:     # lambda was not needed for the ranking: computed for the report
        lam_i = evaluate_half(problem, H[i])["lam"]
    value, meaning = _ranked_value(kind, sf, float(rs[i]), lam_i, float(c[i]))
    max_rs = max(p["max_rf_strength"] for p in parts)
    return {"kind": kind, "n_valid": int(sum(p["n_valid"] for p in parts)),
            "n_strength_ok": int(sum(p["n_strength_ok"] for p in parts)), "max_rf_strength": max_rs,
            "half": H[i].tolist(), "lam": lam_i, "c": float(c[i]), "rf_strength": float(rs[i]), "feasible": feasible,
            "value": value, "value_meaning": meaning, "top": top,
            "status": "optimal" if feasible else "infeasible",
            "infeasible_reason": infeasible_reason(sf, max_rs, lam_i if rs[i] >= sf else None, feasible),
            "notation": notation(full_laminate(H[i][None], True)[0], True)}


# ------------------------------------------------------------------------------------------------ reporting
def stiffness_record(problem: Problem, half: Sequence[int], status: str, proven: bool, runtime_s: float,
                     objective: str = OBJECTIVE, **extra) -> Dict:
    """One result line: objective, load case, t, N, sequence, lambda, c, strength margin, reserve factors, the
    quasi-isotropic laminate with the same N (percent increase of lambda or decrease of c), proven or not, time."""
    half = np.asarray(half, dtype=np.int8)
    n = 2 * len(half)
    kind = objective_kind(problem)
    m = evaluate_half(problem, half)
    full = full_laminate(half[None], True)[0]
    rec = {"objective": objective, "case": problem.name, "loads_N_per_m": list(problem.loads),
           "t_mm": problem.ply_thickness * 1e3, "N": n, "kind": kind, "notation": notation(full, True),
           "half": half.tolist(), "lambda": m["lam"], "c_N_per_m": m["c"], "strength_margin": m["rf_strength"],
           "rf": m["rf"], "margin": m["margin"], "feasible": bool(m["rf_strength"] >= problem.safety_factor
                                                                 and m["lam"] >= problem.safety_factor),
           "status": status, "proven": bool(proven), "runtime_s": float(runtime_s),
           "result": ("proven optimum" if proven and status == "optimal" else "proven infeasible (best design shown)"
                      if proven and status == "infeasible" else f"{status} (not proven)"),
           "ritz_terms": buckling.terms_for(problem.ritz_terms, problem.plate.a, problem.plate.b),
           "shear_both_signs": problem.both_signs, "compliance_unit": COMPLIANCE_UNIT}
    if n % 8 == 0:
        q = evaluate_half(problem, qi_half(n))
        rec["qi"] = {"notation": f"[0/45/-45/90]_{n // 8}s", "lambda": q["lam"], "c_N_per_m": q["c"],
                     "strength_margin": q["rf_strength"]}
        if kind == "lambda":
            rec["improvement_vs_qi_pct"] = 100.0 * (m["lam"] / q["lam"] - 1.0)
            rec["improvement_vs_qi_meaning"] = "increase of lambda"
        else:
            rec["improvement_vs_qi_pct"] = 100.0 * (1.0 - m["c"] / q["c"])
            rec["improvement_vs_qi_meaning"] = "decrease of c"
    else:
        rec["qi"] = None
        rec["improvement_vs_qi_pct"] = None
    rec.update(extra)
    return rec


def stiffness_search(problem: Problem, time_limit: float = 30.0, seed: int = 0, budget: int = 10000) -> Dict:
    """Exact search within time_limit; if it cannot finish, GA + local search (fixed seed) and the better of the two
    designs (never reported as proven; the gap to the open bound of the exact search is given)."""
    from .ga import GAConfig, GeneticAlgorithm
    t0 = time.perf_counter()
    check_problem(problem)
    sf = problem.safety_factor
    kind = objective_kind(problem)
    ex = exact_stiffness(problem, time_limit=time_limit)
    exs = {k: ex.get(k) for k in ("status", "complete", "time_s", "method", "lam", "c", "upper_bound", "gap")}
    if ex["complete"]:
        return {"status": ex["status"], "proven": True, "n_proven": True, "full": ex["full"], "exact": exs,
                "ga": None, "objective": OBJECTIVE, "kind": kind, "time_s": time.perf_counter() - t0}
    g = GeneticAlgorithm(problem, GAConfig(budget=budget, local_search=True, objective=OBJECTIVE), seed).run()
    gh = [int(x) for x in g["best_genes"] if x >= 0]
    kg = make_key(kind, sf, g["rf_strength"], g["lam"], g["c"], gh)
    ke = make_key(kind, sf, ex["rf_strength"], ex["lam"], ex["c"], ex["half"])
    half = gh if key_better(kg, ke) else ex["half"]
    full = full_laminate(np.array(half, dtype=np.int8)[None], True)[0]
    feasible = kg.feasible or ke.feasible
    value = max(kg.primary, ke.primary) if kind == "lambda" else None
    gap = ((ex["upper_bound"] - value) / abs(value)) if (kind == "lambda" and ex.get("upper_bound")) else None
    return {"status": "best found" if feasible else "no feasible design found", "proven": False, "n_proven": True,
            "full": full.tolist(), "exact": exs, "objective": OBJECTIVE, "kind": kind, "gap": gap,
            "ga": {"lam": g["lam"], "c": g["c"], "rf_strength": g["rf_strength"], "feasible": g["feasible"],
                   "time_s": g["time_s"], "seed": seed, "budget": budget},
            "time_s": time.perf_counter() - t0}
