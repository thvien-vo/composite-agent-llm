"""Tests of the fixed-thickness objective "max_stiffness" (stiffness.py) and of the fixed_plies option.

Reference answers are computed by enumerating EVERY valid stacking sequence (stiffness.brute_force: all 4^L strings
filtered by the rules for L <= 9) with the same model; the exact search must give the same objective value (relative
1e-9) and the same sequence (tie rules of stiffness.py)."""
import itertools

import numpy as np
import pytest

from optimization import buckling, clt, stiffness as st
from optimization.api import make_problem, optimize
from optimization.evaluate import Evaluator, analyze_design
from optimization.exhaustive import _FullBounds, _tuple_data, exact_full
from optimization.ga import GAConfig, GeneticAlgorithm
from optimization.rules import Rules, full_laminate, satisfies


def _p(im7, loads, n=None, t=0.125e-3, **kw):
    p = make_problem(loads, material=im7, ply_thickness=t, ritz_terms=8, shear_both_signs=True, **kw)
    return st.fixed_problem(p, n) if n else p


def test_details_keep_values_and_add_A_D_c(im7, rng):
    p = make_problem((3e5, -1e4, 2e4), material=im7, ritz_terms=8, shear_both_signs=True)
    ev = Evaluator(p)
    half = rng.integers(0, 4, size=(40, 10)).astype(np.int8)
    a = ev.compute_plies(half)
    b = ev.compute_plies(half, details=True)
    for k in a:
        assert np.array_equal(a[k], b[k])                      # existing values unchanged
    for i in range(5):
        full = list(half[i]) + list(half[i][::-1])
        A, B, D = clt.abd_matrices(np.array([0.0, 45.0, -45.0, 90.0])[full], np.full(len(full), p.ply_thickness),
                                   ev.Q)
        assert np.allclose(b["A"][i], A, rtol=1e-12) and np.allclose(b["D"][i], D, rtol=1e-12)
        c = max(np.array(N) @ np.linalg.inv(A) @ np.array(N) for N in p.load_cases())
        assert b["c"][i] == pytest.approx(c, rel=1e-10)
        assert b["rf_strength"][i] == pytest.approx(min(b["rf"][i, 0], b["rf"][i, 1]), rel=1e-15)
        assert b["lam"][i] == b["rf"][i, 2]
        det = analyze_design(p, full)
        assert det["compliance_c"] == pytest.approx(c, rel=1e-9)
        assert det["strength_margin"] == pytest.approx(min(det["rf"]["max_stress"], det["rf"]["tsai_wu"]))
        assert np.allclose(det["A"], A) and np.allclose(det["D"], D)


@pytest.mark.parametrize("loads, finite", [((1.5e6, 0, 0), False), ((1e6, 5e5, 0), False), ((4e6, 2e6, 1e6), False),
                                           ((-1e4, 0, 0), True), ((0, 0, 3e5), True), ((8e5, 0, 3e5), True),
                                           ((-1e4, 0, 5e3), True)])
def test_can_buckle_decides_lambda_for_every_design(im7, rng, loads, finite):
    p = _p(im7, loads)
    assert st.can_buckle(p) == finite
    ev = Evaluator(p)
    ev.lazy = False
    lam = ev.compute_plies(rng.integers(0, 4, size=(30, 8)).astype(np.int8))["rf"][:, 2]
    assert np.all(np.isfinite(lam)) if finite else np.all(np.isinf(lam))


def test_compliance_depends_on_counts_only(im7, rng):
    p = _p(im7, (1e6, 5e5, 2e5), 16)
    ev = Evaluator(p)
    base = np.array([0, 0, 0, 1, 2, 3, 3, 1], dtype=np.int8)
    perms = np.array([rng.permutation(base) for _ in range(20)])
    r = ev.compute_plies(perms, details=True)
    assert np.allclose(r["c"], r["c"][0], rtol=1e-13) and np.allclose(r["rf_strength"], r["rf_strength"][0], rtol=1e-13)


def test_order_and_tie_rules():
    sf = 1.5
    rs = np.array([3.0, 3.0, 3.0, 2.0, 1.0])
    lam = np.array([2.0, 2.0 * (1 + 5e-10), 2.0 * (1 - 5e-9), 2.0, 9.0])
    H = np.array([[2, 1], [1, 2], [0, 0], [0, 1], [0, 0]])
    # 0 and 1 tie on lambda (5e-10 < 1e-9) and on rs -> the lexicographically smaller sequence (index 1) wins;
    # index 4 has the largest lambda but fails strength
    assert st.select(rs, lam, np.zeros(5), H, "lambda", sf) == 1
    ka = st.make_key("lambda", sf, 3.0, 2.0, 0.0, [2, 1])
    kb = st.make_key("lambda", sf, 3.0, 2.0 * (1 + 5e-10), 0.0, [1, 2])
    assert st.key_better(kb, ka) and not st.key_better(ka, kb)
    kc = st.make_key("lambda", sf, 9.0, 1.4, 0.0, [0, 0])       # infeasible (lambda < 1.5) loses
    assert st.key_better(ka, kc)
    # compliance: smaller c better; equal c -> larger strength margin
    assert st.select([2.0, 2.5, 1.0], [np.inf] * 3, [10.0, 10.0, 5.0], [[1, 1], [2, 2], [0, 0]], "compliance", sf) == 1
    # nothing meets strength -> largest strength margin
    assert st.select([1.2, 1.4], [np.inf] * 2, [1.0, 2.0], [[0, 0], [1, 1]], "compliance", sf) == 1


def test_lex_smallest_arrangement_matches_enumeration(im7):
    """Only the contiguity rule is checked here (the counts are filtered for balance and the 10 % rule before)."""
    from optimization.rules import check_rules
    p = _p(im7, (1e6, 0, 0), 16)
    for counts in [(4, 1, 1, 2), (5, 1, 1, 1), (0, 3, 3, 2), (2, 2, 2, 2), (6, 1, 1, 0), (7, 0, 0, 1)]:
        ms = np.repeat(np.arange(4), counts)
        valid = sorted({tuple(x) for x in itertools.permutations(ms)
                        if check_rules(full_laminate(np.array([x]), True)[0], p.rules)["contiguity"]})
        got = st.lex_smallest_arrangement(p, counts)
        if not valid:
            assert got is None
        else:
            assert tuple(got.tolist()) == valid[0]


def test_prefix_bounds_are_upper_bounds(im7, rng):
    """Every prefix bound of the branch and bound is >= the Ritz factor of every completion."""
    p = _p(im7, (-6e3, 0, 4e3), 16)
    ev = Evaluator(p)
    fb, data = _tuple_data(p, ev, 8)
    for d in data[:6]:
        pb = st.PrefixBound(fb, list(d["R"]))
        ms = np.repeat(np.arange(4), d["counts"])
        H = np.array([rng.permutation(ms) for _ in range(30)], dtype=np.int8)
        lam = st._lam_nocache(ev, fb.dvec(H))
        for k in range(9):
            rem = np.array([[np.sum(h[k:] == s) for s in range(4)] for h in H])
            b = pb.bound(pb.partial(H, k), rem, k)
            assert np.all(lam <= b * (1 + 1e-12))


@pytest.mark.parametrize("loads", [(1.5e6, 0, 0), (-1e4, 0, 0), (0, 0, 4e4), (-1e4, 0, 5e3), (2e5, 0, 6e4),
                                   (8e5, 0, 3e5)])
@pytest.mark.parametrize("n", [12, 16])
def test_exact_equals_brute_force(im7, loads, n):
    """(8e5, 0, 3e5) at 12 plies: no ply-count combination meets strength -> ranked by the strength margin."""
    p = _p(im7, loads, n, t=0.18288e-3)
    ex = st.exact_stiffness(p, time_limit=120)
    bf = st.brute_force_merge(p, [st.brute_force(p)])
    assert ex["complete"] and ex["proven"]
    assert ex["status"] == bf["status"] and ex["value_meaning"] == bf["value_meaning"]
    assert ex["half"] == bf["half"]
    assert ex["value"] == pytest.approx(bf["value"], rel=1e-9)
    assert ex["rf_strength"] == pytest.approx(bf["rf_strength"], rel=1e-9)
    assert ex["max_rf_strength"] == pytest.approx(bf["max_rf_strength"], rel=1e-9)
    assert (ex["infeasible_reason"] is None) == (bf["infeasible_reason"] is None) == (ex["status"] == "optimal")
    if ex["max_rf_strength"] < p.safety_factor:     # infeasible by strength: the design of largest strength margin
        assert ex["rf_strength"] == pytest.approx(ex["max_rf_strength"], rel=1e-12)
    assert ex["lam"] == pytest.approx(bf["lam"], rel=1e-9) or (np.isinf(ex["lam"]) and np.isinf(bf["lam"]))
    if ex["rf_strength"] >= p.safety_factor:
        for k in ("lam", "c"):
            a, b = ex[k], bf[k]
            assert (np.isinf(a) and np.isinf(b)) or a == pytest.approx(b, rel=1e-9)


def test_default_ply_count_and_no_default(im7):
    p = _p(im7, (-1e4, 0, 0))                                      # C3, t = 0.125 mm
    d = st.default_ply_count(p)
    assert d["found"] and d["n"] == 24 and d["qi"] == "[0/45/-45/90]_3s"
    assert [x["meets"] for x in d["scan"]] == [False, False, True]
    assert d["lam"] >= 1.5 and d["rf_strength"] >= 1.5
    r = optimize(None, problem=p, objective="max_stiffness", method="exhaustive")
    assert r.n_plies == 24 and r.stiffness["ply_count"]["n"] == 24 and r.stiffness["proven"]
    assert r.status == "optimal" and r.objective == "max_stiffness"
    assert r.stiffness["lambda"] >= r.stiffness["qi"]["lambda"] and r.stiffness["improvement_vs_qi_pct"] >= 0
    big = _p(im7, (4e6, 2e6, 1e6))                                  # C8, t = 0.125 mm: no QI up to 64 plies
    assert not st.default_ply_count(big)["found"]
    with pytest.raises(st.NoFeasiblePlyCount):
        optimize(None, problem=big, objective="max_stiffness", method="exhaustive")


def test_fixed_plies_infeasible_is_reported(im7):
    p = _p(im7, (-1e4, 0, 5e3))                                     # C7: 16 plies of 0.125 mm cannot meet lambda
    r = optimize(None, problem=p, objective="max_stiffness", fixed_plies=16, method="exhaustive")
    assert r.n_plies == 16 and not r.feasible and r.status == "infeasible" and r.stiffness["proven"]
    assert r.stiffness["lambda"] < 1.5 and r.stiffness["strength_margin"] >= 1.5
    assert any("INFEASIBLE" in w for w in r.warnings)
    p1 = _p(im7, (1.5e6, 0, 0))                                     # C1: 16 plies fail strength
    r1 = optimize(None, problem=p1, objective="max_stiffness", fixed_plies=16, method="exhaustive")
    assert not r1.feasible and r1.status == "infeasible" and r1.stiffness["strength_margin"] < 1.5
    for x in (r, r1):                                               # never called an optimum
        assert x.stiffness["result"].startswith("proven infeasible") and x.stiffness["infeasible_reason"]
    assert r1.stiffness["strength_margin"] == pytest.approx(r1.stiffness["max_rf_strength"], rel=1e-12)
    assert "strength" in r1.stiffness["infeasible_reason"] and "buckling" in r.stiffness["infeasible_reason"]


def test_brute_force_streaming_parts(im7):
    """Streaming enumeration: same answer in one part or in several parts with small blocks; 10 056 valid designs."""
    p = _p(im7, (0, 0, 4e4), 16, t=0.18288e-3)
    one = st.brute_force_merge(p, [st.brute_force(p)])
    parts = st.brute_force_merge(p, [st.brute_force(p, part=i, n_parts=3, chunk=1 << 10) for i in range(3)])
    assert one["n_valid"] == parts["n_valid"] == 10056
    assert one["half"] == parts["half"] and one["value"] == parts["value"]
    assert one["max_rf_strength"] == parts["max_rf_strength"] and one["n_strength_ok"] == parts["n_strength_ok"]


def test_fixed_plies_validation(im7):
    p = _p(im7, (-1e4, 0, 0))
    for bad in (15, 6, 66):
        with pytest.raises(ValueError):
            optimize(None, problem=p, objective="max_stiffness", fixed_plies=bad)
    with pytest.raises(ValueError):
        optimize(None, problem=p, objective="min_plies", fixed_plies=16)
    with pytest.raises(ValueError):
        optimize(None, problem=p, objective="stiffest")
    with pytest.raises(ValueError):                                 # max_margin needs a ply count
        optimize(None, problem=p, objective="max_margin")


def test_max_margin_fixed_plies_api(im7):
    p = _p(im7, (-1e4, 0, 5e3))
    r = optimize(None, problem=p, objective="max_margin", fixed_plies=16, method="exhaustive")
    ref = exact_full(st.fixed_problem(p, 16), time_limit=120)["per_n"][0]
    assert r.n_plies == 16 and r.objective == "max_margin"
    assert r.stiffness["margin"] == pytest.approx(ref["max_margin"], rel=1e-9)
    assert r.stacking_deg == [float(x) for x in np.array([0.0, 45.0, -45.0, 90.0])[ref["full"]]]


def test_default_objective_unchanged(im7):
    r = optimize((-1e4, 0, 0), material=im7, method="ga", seed=0, budget=300, rules=Rules(n_max=24))
    assert r.objective == "min_plies" and r.stiffness is None


@pytest.mark.parametrize("ls", [False, True])
def test_ga_keeps_the_ply_count(im7, ls):
    p = _p(im7, (-1e4, 0, 5e3), 16, t=0.18288e-3)
    ga = GeneticAlgorithm(p, GAConfig(budget=1500, objective="max_stiffness", local_search=ls), seed=3)
    r = ga.run()
    assert all(h["n_plies"] == 16 for h in r["history"])
    assert all(v[0] == 16 for v in ga.ev.cache.values())             # every design analysed had 16 plies
    assert r["n_plies"] == 16 and r["feasible"]


def test_ga_local_search_reaches_proven_optimum(im7):
    p = _p(im7, (-1e4, 0, 5e3), 16, t=0.18288e-3)
    ex = st.exact_stiffness(p)
    target = st.make_key(ex["kind"], p.safety_factor, ex["rf_strength"], ex["lam"], ex["c"], ex["half"])
    r = optimize(None, problem=p, objective="max_stiffness", method="ga", seed=0, target=target,
                 ga_config=GAConfig(budget=3000, local_search=True))
    assert r.stiffness["lambda"] == pytest.approx(ex["lam"], rel=1e-9) and r.hit_evals is not None
    assert r.stiffness["proven"] is False


def test_incomplete_search_is_never_proven(im7):
    """Time limit reached (here: zero, so only the trial designs are evaluated): best design found, open bound, gap,
    status 'incomplete', never proven; the complete search then gives a value inside [best found, bound]."""
    p = _p(im7, (8e5, 0, 3e5), 32)                                   # C6, 32 plies
    r = st.exact_stiffness(p, time_limit=0.0)
    assert r["status"] == "incomplete" and not r["proven"] and not r["complete"]
    assert r["upper_bound"] >= r["lam"] and r["gap"] >= 0
    assert satisfies(r["full"], p.rules)
    full = st.exact_stiffness(p, time_limit=300)
    assert full["complete"] and r["lam"] <= full["lam"] * (1 + 1e-9) <= r["upper_bound"] * (1 + 2e-9)
    s = st.stiffness_search(p, time_limit=0.0, budget=1500)
    assert s["proven"] is False and s["status"] in ("best found", "no feasible design found")
