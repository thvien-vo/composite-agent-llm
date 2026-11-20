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
    # the GA optimises the 8 x 8 model: its value is lambda_8x8; "lambda" is the reported 16 x 16 value (step 2)
    assert r.stiffness["lambda_8x8"] == pytest.approx(ex["lam"], rel=1e-9) and r.hit_evals is not None
    assert r.stiffness["lambda"] == analyze_design(p, full_laminate(np.array([r.stiffness["half"]]), True)[0])["lambda"]
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


# ------------------------------------------------------------------------------------------------ step 2
def test_min_plies_fallback_for_the_default_ply_count(im7):
    """No QI laminate up to 64 plies meets the constraints for C8: N = fewest plies of the objective min_plies
    (count-level scan, proven: the load cannot buckle the plate); none for t = 0.125 mm -> NoFeasiblePlyCount."""
    p = _p(im7, (4e6, 2e6, 1e6), t=0.18288e-3)
    assert not st.default_ply_count(p)["found"]
    m = st.min_plies_count(p)
    assert m["found"] and m["n"] == 64 and m["proven"]
    assert m["scan"][-1]["max_strength_margin"] >= 1.5 > m["scan"][-2]["max_strength_margin"]
    q, info = st.resolve_ply_count(p, None)
    assert q.rules.n_min == q.rules.n_max == 64 and info["source"] == "min_plies fallback" and info["n_proven"]
    assert not st.min_plies_count(_p(im7, (4e6, 2e6, 1e6)))["found"]
    # the count-level scan gives the same ply count as the exact fewest-plies search of the API
    for loads, t in (((1e6, 5e5, 0), 0.18288e-3), ((1.5e6, 0, 0), 0.125e-3)):
        pp = _p(im7, loads, t=t)
        ref = optimize(None, problem=pp, objective="min_plies", method="exhaustive")
        assert st.min_plies_count(pp)["n"] == ref.n_plies and ref.status == "optimal"
    assert st.resolve_ply_count(_p(im7, (-1e4, 0, 0)), None)[1]["source"] == "QI_ks"


def test_arrangement_dp_matches_enumeration(im7, rng):
    """Exact DP over the arrangements of a count tuple = enumeration of every valid arrangement: largest weighted sum,
    number of arrangements, lexicographically smallest arrangement above a threshold."""
    from optimization.rules import check_rules
    p = _p(im7, (1e6, 0, 0), 16)
    for counts in [(4, 1, 1, 2), (5, 1, 1, 1), (0, 3, 3, 2), (2, 2, 2, 2), (3, 2, 2, 1)]:
        ms = np.repeat(np.arange(4), counts)
        valid = sorted({tuple(x) for x in itertools.permutations(ms)
                        if check_rules(full_laminate(np.array([x]), True)[0], p.rules)["contiguity"]})
        vals, w = rng.random(4), np.sort(rng.random(8))[::-1]
        dp = st.ArrangementDP(p, counts, vals, w)
        V = np.array([np.dot(w, vals[list(x)]) for x in valid])
        assert dp.n_arrangements == len(valid) and dp.vmax == pytest.approx(V.max(), rel=1e-13)
        for thr in (V.max(), np.median(V), V.min()):
            low = thr * (1 - 1e-12)
            best = min(x for x, v in zip(valid, V) if v >= low)
            assert tuple(dp.lex_smallest(low).tolist()) == best


@pytest.mark.parametrize("loads", [(1.5e6, 0, 0), (1e6, 5e5, 0), (4e6, 2e6, 1e6), (2e5, 1e6, 0)])
@pytest.mark.parametrize("n, t", [(12, 0.18288e-3), (16, 0.125e-3), (16, 0.18288e-3)])
def test_compliance_tie_break_equals_enumeration(im7, loads, n, t):
    """Compliance objective: c and the strength margin depend on the ply counts only (no difference at all within a
    count tuple over every valid design); ties broken by the bending stiffness of the main load direction (D11 for
    Nx, D22 for Ny, D66 for Nxy), then the sequence: the exact search equals the enumeration of every valid design."""
    p = _p(im7, loads, n, t=t)
    assert st.objective_kind(p) == "compliance"
    ex = st.exact_stiffness(p)
    bf = st.brute_force_merge(p, [st.brute_force(p, top_k=5)])
    assert bf["max_rel_spread_c_in_counts"] == 0.0 and bf["max_rel_spread_rs_in_counts"] == 0.0
    assert ex["half"] == bf["half"] and ex["status"] == bf["status"]
    d = int(np.argmax(np.abs(loads)))
    assert ex["direction"] == ("D11", "D22", "D66")[d]
    assert ex["ddir"] == pytest.approx(bf["ddir"], rel=1e-12)
    assert st.evaluate_half(p, ex["half"])["D"][d, d] == pytest.approx(ex["ddir"], rel=1e-12)


def test_order_with_bending_stiffness_tie_break():
    sf = 1.5
    # compliance: equal c and strength margin -> larger D of the main direction wins before the sequence
    rs, lam, c = [2.0, 2.0, 2.0], [np.inf] * 3, [10.0, 10.0 * (1 + 3e-10), 10.0]
    H = [[0, 1], [1, 0], [2, 2]]
    assert st.select(rs, lam, c, H, "compliance", sf, ddir=[5.0, 6.0, 6.0 * (1 - 4e-10)]) == 1
    assert st.select(rs, lam, c, H, "compliance", sf) == 0                 # without D: the sequence decides
    ka = st.make_key("compliance", sf, 2.0, np.inf, 10.0, [0, 1], 5.0)
    kb = st.make_key("compliance", sf, 2.0, np.inf, 10.0, [1, 0], 6.0)
    assert st.key_better(kb, ka) and not st.key_better(ka, kb)
    kl = st.make_key("lambda", sf, 2.0, 3.0, 10.0, [1, 0], 6.0)            # lambda objective: D is not used
    assert kl.tertiary == 0.0
    res = {"rf_strength": np.array(rs), "lam": np.array(lam), "c": np.array([10.0, 10.0, 10.0]),
           "ddir": np.array([5.0, 7.0, 6.0])}
    assert list(st.rank_order(res, sf, "compliance")) == [1, 2, 0]


def test_reported_lambda_is_the_same_everywhere(im7):
    """The buckling factor reported for a design (16 x 16 sine series) is the same number in analyze_design and in
    the result record (stiffness_record, i.e. the results file); the 8 x 8 value and the polynomial check are next
    to it. Objectives of the fewest-plies study keep their values (rf['buckling'], margin, max_margin 'lambda')."""
    p = _p(im7, (-1e4, 0, 5e3), 24)
    r = optimize(None, problem=p, objective="max_stiffness", method="exhaustive")
    s = r.stiffness
    full = full_laminate(np.array([s["half"]]), True)[0]
    det = analyze_design(p, full)
    assert s["lambda"] == det["lambda"] == det["lambda_16x16"] == r.rf["buckling"]
    assert s["lambda_16x16"] == s["lambda"] and s["lambda_8x8"] == pytest.approx(det["lambda_8x8"], rel=1e-12)
    assert s["lambda"] <= s["lambda_8x8"] * (1 + 1e-12)                    # nested bases: 16 x 16 <= 8 x 8
    assert abs(s["lambda_poly16"] / s["lambda"] - 1) < 0.01
    qfull = full_laminate(st.qi_half(24)[None], True)[0]
    assert s["qi"]["lambda"] == analyze_design(p, qfull)["lambda"]
    # the rule of the fewest-plies study is unchanged unless report_lambda: min(optimisation terms, converged)
    legacy = analyze_design(p, full, report_lambda=False)
    by = legacy["by_shear_sign"]
    ref = min(min(v["buckling_ritz_full"]["lambda_converged"], v["buckling_ritz_full"]["lambda_optimization_terms"])
              for v in by.values())
    assert legacy["rf"]["buckling"] == ref
    mm = optimize(None, problem=p, objective="max_margin", method="exhaustive")
    mfull = full_laminate(np.array([mm.stiffness["half"]]), True)[0]
    assert mm.objective == "max_margin" and mm.stiffness["lambda"] == mm.stiffness["lambda_8x8"]
    assert mm.rf["buckling"] == analyze_design(p, mfull)["rf"]["buckling"]


def test_top_k_rule_and_16x16_choice(im7):
    """k = 1 at the start: the gap of the list is 0 < the 8x8-16x16 difference, so k is doubled until the rule
    holds; the design returned is the best of ALL valid designs by the 16 x 16 value (enumeration, 12 plies)."""
    p = _p(im7, (-6e3, 0, 4e3), 12, t=0.125e-3)
    r = st.solve_max_stiffness(p, k_start=1)
    hist = r["k_history"]
    assert len(hist) >= 2 and hist[0]["k"] == 1 and r["k_rule_met"]
    assert hist[-1]["gap"] >= hist[-1]["e_max"] - 1e-9 or hist[-1]["all_designs_listed"]
    bf = st.brute_force(p, top_k=10 ** 6)
    ok = bf["rs"] >= p.safety_factor
    H = bf["halves"][ok]
    lam16 = st.report_lambdas(p, H)
    i = st.select(bf["rs"][ok], lam16, np.zeros(len(H)), H, "lambda", p.safety_factor)
    assert r["half"] == H[i].tolist() and r["lam16"] == pytest.approx(lam16[i], rel=1e-12)
    assert np.all(lam16 <= bf["lam"][ok] * (1 + 1e-12))
    assert r["proven_16x16"] and r["proven_in"] == st.PROVEN_IN


def test_incomplete_branch_is_reported_and_never_proven(im7):
    """Forced INCOMPLETE (node limit, tiny blocks): best design found, open bound, gap, status 'incomplete', never
    proven, the label says INCOMPLETE with the gap; the record and the label of a complete run differ."""
    p = _p(im7, (-1e4, 0, 5e3), 24)
    r = st.solve_max_stiffness(p, max_nodes=50)
    assert r["status"] == "incomplete" and not r["proven"] and not r["complete"] and r["proven_in"] is None
    assert r["upper_bound"] >= r["lam"] and r["gap"] > 0 and r["k_history"] == []
    ex = st.exact_stiffness(p, block=1, max_nodes=10)
    assert ex["status"] == "incomplete" and not ex["proven"]
    label = st.status_text(r["status"], r["proven"], r["gap"])
    assert label.startswith("INCOMPLETE") and "gap" in label and "proven optimum" not in label
    rec = st.stiffness_record(p, r["half"], status=r["status"], proven=r["proven"], runtime_s=0.0)
    assert rec["result"] == "incomplete (not proven)" and rec["proven_in"] is None
    full = st.solve_max_stiffness(p)
    assert full["complete"] and r["lam"] <= full["lam_8x8_best"] * (1 + 1e-9) <= r["upper_bound"] * (1 + 2e-9)
    assert st.status_text(full["status"], full["proven"]).startswith("proven optimum")
