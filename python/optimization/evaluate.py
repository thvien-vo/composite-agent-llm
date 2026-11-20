"""
evaluate.py - problem definition and batch evaluation of laminates.

Safety margin of a design = min over plies and over the active criteria of the reserve factor (RF):
    RF_max_stress (linear), RF_tsai_wu (quadratic, F12* given), RF_buckling (load factor), [RF_max_strain].
A design is feasible when margin >= safety_factor (default 1.5). An optional per-criterion divisor
(criterion_factors) reproduces formulations where a factor is built into one criterion only
(Le Riche & Haftka: lambda_cs = strain RF / 1.5, lambda_cb without factor).

Symmetric laminates use a fast exact path: with B = 0 and in-plane loads only, the laminate strain is uniform,
eps0 = A^-1 N, and A depends only on the ply counts, so ply stresses depend only on the ply angle;
D = sum of Qbar(angle) * position weight. Non-symmetric laminates use the general CLT path (A, B, D, curvature,
stresses at ply mid-height as in clt_engine.m).

compute_plies(..., details=True) also returns the quantities computed on the way (the values above are unchanged):
A, D (P, 3, 3) [N/m, N m], the strength margin rf_strength = min over the strength criteria (Max Stress, Tsai-Wu,
[Max Strain]), the buckling factor lam (inf when not computed: no buckling criterion, a load that cannot buckle the
plate, or lazy evaluation of a design below the safety factor in strength) and the in-plane compliance
c = N . eps0 = N^T A^-1 N [N/m] (largest value over the load cases +Nxy / -Nxy).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import buckling, clt, failure
from .material import DEFAULT_PLY_THICKNESS, Material
from .rules import ANGLES, EMPTY, ENCODINGS, Encoding, Rules, full_laminate

CRITERION_NAMES = ("max_stress", "tsai_wu", "max_strain", "buckling")
REPORT_TERMS = 16                   # sine-series size of the reported (converged) buckling factor, 16 x 16 terms
DIRECTION_NAMES = ("D11", "D22", "D66")


def main_load_direction(loads: Sequence[float]) -> int:
    """Main load direction: index of the largest |Nx|, |Ny|, |Nxy| (the first one on equal magnitudes); its bending
    stiffness D11, D22 or D66 (DIRECTION_NAMES) breaks the ties of the compliance objective (stiffness.py)."""
    return int(np.argmax(np.abs(np.asarray(loads, dtype=float)[:3])))


@dataclass(frozen=True)
class Plate:
    """Simply supported rectangular plate, a along x, b along y (m)."""
    a: float = 0.5
    b: float = 0.5


@dataclass
class Problem:
    """Everything that defines one optimization problem (all SI)."""
    loads: Tuple[float, float, float]                       # (Nx, Ny, Nxy) N/m, tension +
    material: Material
    rules: Rules = field(default_factory=Rules)
    safety_factor: float = 1.5
    plate: Plate = field(default_factory=Plate)
    ply_thickness: float = DEFAULT_PLY_THICKNESS
    encoding: str = "ply"                                   # "ply" or "stack" (0_2, +-45, 90_2)
    criteria: Tuple[str, ...] = ("max_stress", "tsai_wu", "buckling")
    f12_star: float = -0.5
    buckling_model: str = "closed_form"     # "closed_form" (Nx, Ny; D16 = D26 = 0), "ritz" (adds Nxy, D16 = D26 = 0),
                                            # "ritz_full" (Nx, Ny, Nxy and D16, D26: the project model)
    ritz_terms: int = 12                    # ritz_full: size k of the sine series, M x N = buckling.terms_for(k, a, b)
    buckling_modes: Optional[Tuple[int, int]] = None        # None = proven-complete mode scan
    strain_allowables: Optional[Tuple[float, float, float]] = None
    criterion_factors: Dict[str, float] = field(default_factory=dict)
    name: str = ""
    shear_both_signs: bool = False          # check strength and buckling for +Nxy AND -Nxy (shear may reverse)

    def __post_init__(self):
        self.loads = tuple(float(x) for x in self.loads)
        bad = [c for c in self.criteria if c not in CRITERION_NAMES]
        if bad:
            raise ValueError(f"unknown criteria {bad}")
        if "max_strain" in self.criteria and self.strain_allowables is None:
            raise ValueError("max_strain needs strain_allowables")
        if self.buckling_model not in ("closed_form", "ritz", "ritz_full"):
            raise ValueError("buckling_model must be 'closed_form', 'ritz' or 'ritz_full'")

    @property
    def enc(self) -> Encoding:
        return ENCODINGS[self.encoding]

    @property
    def has_compression(self) -> bool:
        Nx, Ny, Nxy = self.loads
        return Nx < 0 or Ny < 0 or (self.buckling_model in ("ritz", "ritz_full") and Nxy != 0)

    @property
    def both_signs(self) -> bool:
        """True when the option is on AND there is a shear load (otherwise both signs are the same load)."""
        return self.shear_both_signs and self.loads[2] != 0.0

    def load_cases(self) -> Tuple[Tuple[float, float, float], ...]:
        """(Nx, Ny, Nxy) and, with shear_both_signs, (Nx, Ny, -Nxy)."""
        Nx, Ny, Nxy = self.loads
        return ((Nx, Ny, Nxy), (Nx, Ny, -Nxy)) if self.both_signs else ((Nx, Ny, Nxy),)

    def describe(self) -> Dict:
        return {"Nx": self.loads[0], "Ny": self.loads[1], "Nxy": self.loads[2], "material": self.material.name,
                "safety_factor": self.safety_factor, "a": self.plate.a, "b": self.plate.b,
                "ply_thickness": self.ply_thickness, "encoding": self.encoding, "criteria": list(self.criteria),
                "f12_star": self.f12_star, "buckling_model": self.buckling_model, "ritz_terms": self.ritz_terms,
                "rules": {"symmetric": self.rules.symmetric, "balanced": self.rules.balanced,
                          "min_fraction": self.rules.min_fraction, "max_contiguous": self.rules.max_contiguous,
                          "n_min": self.rules.n_min, "n_max": self.rules.n_max},
                **({"shear_both_signs": True} if self.shear_both_signs else {})}


class Evaluator:
    """Batch evaluation with a cache. Counters: n_calls (designs requested), n_unique (designs analysed).

    ritz_full model: the basis integrals of the sine series are computed once (buckling.RitzModel), buckling
    factors are cached by D matrix, and with lazy = True (default for this model) the buckling factor is computed
    only for designs that already pass strength (strength RF >= safety factor). For the other designs the
    margin returned is the strength RF (an upper bound of the true margin; they are infeasible either way) and
    their buckling RF is reported as inf. NSGA-II sets lazy = False (it needs exact margins for its front)."""

    def __init__(self, problem: Problem, n_genes: Optional[int] = None, objective: Optional[str] = None):
        self.p = problem
        # objective "max_stiffness": evaluate() also returns rf_strength, lam, c and ddir (bending stiffness in the
        # main load direction, D11 / D22 / D66, see main_load_direction), cached with the design
        self.objective = objective
        self.extra: Dict[bytes, Tuple] = {}
        self.direction = main_load_direction(problem.loads)
        self.enc = problem.enc
        self.Q = clt.q_matrix(problem.material.E1, problem.material.E2, problem.material.G12, problem.material.nu12)
        self.Qb = clt.qbar(self.Q, ANGLES)                                   # (4, 3, 3)
        self.Qb_ext = np.concatenate([self.Qb, np.zeros((1, 3, 3))])         # index 4 = EMPTY -> 0
        self.N = np.array(problem.loads)
        unit = self.enc.plies_per_symbol * problem.rules.multiplier()
        self.n_genes = n_genes or problem.rules.n_max // unit
        self.cache: Dict[bytes, Tuple] = {}
        self.n_calls = 0
        self.n_unique = 0
        self.factors = np.array([problem.criterion_factors.get(c, 1.0) for c in problem.criteria])
        # shear_both_signs: every check is also made for -Nxy and the smaller RF is kept
        self.N_cases = [np.array(lc) for lc in problem.load_cases()]
        self.ritz_model = None
        self.ritz_models = []
        self.lazy = False
        if problem.buckling_model == "ritz_full" and "buckling" in problem.criteria:
            a, b = problem.plate.a, problem.plate.b
            mn = buckling.terms_for(problem.ritz_terms, a, b)
            self.ritz_models = [buckling.RitzModel(a, b, *lc, *mn) for lc in problem.load_cases()]
            self.ritz_model = self.ritz_models[0]
            self.lazy = True
        self.n_buckling = 0   # designs whose buckling factor was computed

    # ------------------------------------------------------------------ public
    def evaluate(self, genes: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate canonical design strings (P, n_genes). Returns arrays: n_plies, margin, rf (P, n_crit),
        crit (index of the governing criterion), feasible."""
        g = np.atleast_2d(np.asarray(genes, dtype=np.int8))
        self.n_calls += len(g)
        keys = [r.tobytes() for r in g]
        todo, seen = [], set()
        for i, k in enumerate(keys):
            if k not in self.cache and k not in seen:
                todo.append(i)
                seen.add(k)
        stiff = self.objective == "max_stiffness"
        if todo:
            res = self.compute_plies(self.enc.decode(g[todo]), details=stiff)
            for j, i in enumerate(todo):
                self.cache[keys[i]] = (res["n_plies"][j], res["margin"][j], res["rf"][j], res["crit"][j])
                if stiff:
                    self.extra[keys[i]] = (res["rf_strength"][j], res["lam"][j], res["c"][j],
                                           res["D"][j, self.direction, self.direction])
            self.n_unique += len(todo)
        n = np.array([self.cache[k][0] for k in keys])
        margin = np.array([self.cache[k][1] for k in keys])
        rf = np.array([self.cache[k][2] for k in keys])
        crit = np.array([self.cache[k][3] for k in keys])
        out = {"n_plies": n, "margin": margin, "rf": rf, "crit": crit,
               "feasible": margin >= self.p.safety_factor}
        if stiff:
            ex = np.array([self.extra[k] for k in keys], dtype=float).reshape(len(keys), 4)
            out.update(rf_strength=ex[:, 0], lam=ex[:, 1], c=ex[:, 2], ddir=ex[:, 3])
        return out

    def compute_plies(self, plies: np.ndarray, details: bool = False) -> Dict[str, np.ndarray]:
        """No cache, no counters. plies: (P, W) ply codes, EMPTY padding at the start (right-aligned).
        details=True adds A, D, rf_strength, lam and c (see the module doc)."""
        plies = np.atleast_2d(plies)
        if self.p.rules.symmetric:
            return self._symmetric(plies, details)
        full = full_laminate(plies, False)
        if full.dtype != object:
            return self._general(full, details)
        keys = ("n_plies", "margin", "rf", "crit") + (("A", "D", "rf_strength", "lam", "c") if details else ())
        out = {k: [None] * len(plies) for k in keys}
        lens = np.array([len(r) for r in full])
        for L in np.unique(lens):
            idx = np.where(lens == L)[0]
            r = self._general(np.array([full[i] for i in idx], dtype=int), details)
            for k in out:
                for j, i in enumerate(idx):
                    out[k][i] = r[k][j]
        return {k: np.array(v) for k, v in out.items()}

    # ------------------------------------------------------------------ paths
    def _finish(self, n_plies, rf_strength: Dict[str, np.ndarray], D: np.ndarray):
        cols = []
        need = None
        if self.lazy and rf_strength:
            fac = {c: self.factors[i] for i, c in enumerate(self.p.criteria)}
            rs = np.min(np.stack([v / fac[c] for c, v in rf_strength.items()], axis=1), axis=1)
            need = rs >= self.p.safety_factor
        for c in self.p.criteria:
            if c == "buckling":
                lam = np.full(len(D), np.inf)
                idx = np.arange(len(D)) if need is None else np.where(need)[0]
                if len(idx):
                    lam[idx] = self.buckling_rf(D[idx])
                cols.append(lam)
            else:
                cols.append(rf_strength[c])
        rf = np.stack(cols, axis=1) / self.factors[None, :]
        crit = np.argmin(rf, axis=1)
        return {"n_plies": n_plies, "margin": rf[np.arange(len(rf)), crit], "rf": rf, "crit": crit}

    def _strength_rf(self, stress12: np.ndarray, strain12: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        m = self.p.material
        if "max_stress" in self.p.criteria:
            out["max_stress"] = failure.rf_max_stress(stress12, m)
        if "tsai_wu" in self.p.criteria:
            out["tsai_wu"] = failure.rf_tsai_wu(stress12, m, self.p.f12_star)
        if "max_strain" in self.p.criteria:
            out["max_strain"] = failure.rf_max_strain(strain12, self.p.strain_allowables)
        return out

    @staticmethod
    def _min_cases(per_case):
        """Smallest RF over the load cases (+Nxy, -Nxy), criterion by criterion."""
        if len(per_case) == 1:
            return per_case[0]
        return {k: np.minimum.reduce([r[k] for r in per_case]) for k in per_case[0]}

    def _details(self, out: Dict[str, np.ndarray], A: np.ndarray, D: np.ndarray, comp) -> Dict[str, np.ndarray]:
        """A, D, strength margin, buckling factor and in-plane compliance of a batch (details=True)."""
        crit = list(self.p.criteria)
        strength = [i for i, c in enumerate(crit) if c != "buckling"]
        P = len(out["rf"])
        rs = out["rf"][:, strength].min(axis=1) if strength else np.full(P, np.inf)
        lam = out["rf"][:, crit.index("buckling")] if "buckling" in crit else np.full(P, np.inf)
        return {"A": A, "D": D, "rf_strength": rs, "lam": lam, "c": np.max(np.stack(comp), axis=0)}

    def _symmetric(self, half: np.ndarray, details: bool = False):
        P, W = half.shape
        t = self.p.ply_thickness
        counts = np.stack([(half == k).sum(axis=1) for k in range(4)], axis=1)          # (P, 4)
        n_plies = 2 * counts.sum(axis=1)
        A = 2.0 * t * np.einsum("pk,kij->pij", counts.astype(float), self.Qb)
        present = counts > 0
        per_case = []
        comp = []
        for N in self.N_cases:
            eps0 = clt.midplane_strain_symmetric(A, N)
            if details:
                comp.append(eps0 @ N)                              # c = N . eps0 = N^T A^-1 N  [N/m]
            strain12, stress12 = clt.ply_stress_from_strain(eps0, ANGLES, self.Q)       # (P, 4, 3)
            per_case.append({k: np.where(present, v, np.inf).min(axis=1)
                             for k, v in self._strength_rf(stress12, strain12).items()})
        rf_s = self._min_cases(per_case)
        j = np.arange(W)[::-1]                                     # distance index from the mid-plane
        w = 2.0 * t ** 3 * ((j + 1.0) ** 3 - j ** 3) / 3.0         # both mirrored plies
        Qp = self.Qb_ext[np.where(half >= 0, half, 4)]             # (P, W, 3, 3)
        D = np.einsum("w,pwij->pij", w, Qp)
        out = self._finish(n_plies, rf_s, D)
        if details:
            out.update(self._details(out, A, D, comp))
        return out

    def _general(self, full: np.ndarray, details: bool = False):
        P, n = full.shape
        t = np.full(full.shape, self.p.ply_thickness)
        ang = ANGLES[full]
        A, B, D = clt.abd_matrices(ang, t, self.Q)
        ABD = np.concatenate([np.concatenate([A, B], axis=2), np.concatenate([B, D], axis=2)], axis=1)
        z = clt.z_coordinates(t)
        zmid = (z[:, :-1] + z[:, 1:]) / 2.0
        per_case = []
        comp = []
        for N in self.N_cases:
            rhs = np.broadcast_to(np.concatenate([N, np.zeros(3)]), (P, 6))[..., None]
            sol = np.linalg.solve(ABD, rhs)[..., 0]
            eps0, kappa = sol[:, :3], sol[:, 3:]
            if details:
                comp.append(eps0 @ N)
            eps_xy = eps0[:, None, :] + zmid[..., None] * kappa[:, None, :]
            strain12 = clt.strain_to_material(eps_xy, ang)
            stress12 = strain12 @ self.Q.T
            per_case.append({k: v.min(axis=1) for k, v in self._strength_rf(stress12, strain12).items()})
        rf_s = self._min_cases(per_case)
        out = self._finish(np.full(P, n), rf_s, D)
        if details:
            out.update(self._details(out, A, D, comp))
        return out

    def lam_full(self, Dvec: np.ndarray) -> np.ndarray:
        """Ritz (D16, D26) buckling factors of D vectors (P, 6): the smallest over the load cases (+Nxy, -Nxy)."""
        lam = self.ritz_models[0].lam_cached(Dvec)
        for m in self.ritz_models[1:]:
            lam = np.minimum(lam, m.lam_cached(Dvec))
        return lam

    def buckling_rf(self, D: np.ndarray) -> np.ndarray:
        Nx, Ny, Nxy = self.p.loads
        a, b = self.p.plate.a, self.p.plate.b
        if self.p.buckling_model == "ritz_full":
            self.n_buckling += len(D)
            return self.lam_full(buckling.d_vector(D))
        if self.p.buckling_model == "ritz":
            lam = None
            for _, _, s in self.p.load_cases():
                lr = buckling.ritz(D[:, 0, 0], D[:, 0, 1], D[:, 1, 1], D[:, 2, 2], Nx, Ny, s, a, b,
                                   self.p.ritz_terms, self.p.ritz_terms)
                lam = lr if lam is None else np.minimum(lam, lr)
            return lam
        lam, _, _, _ = buckling.closed_form(D[:, 0, 0], D[:, 0, 1], D[:, 1, 1], D[:, 2, 2], Nx, Ny, a, b,
                                            self.p.buckling_modes)
        return lam


# ---------------------------------------------------------------------------------------------- details
def analyze_design(problem: Problem, full: Sequence[int], report_lambda: bool = False) -> Dict:
    """Detailed analysis of one design (full laminate ply codes, ply 1 first) with the general CLT path:
    per-ply RFs, critical ply / criterion, buckling mode, D16/D11, D26/D22, Ritz check (with Nxy).
    With shear_both_signs the design is analysed for +Nxy and -Nxy: every RF is the smaller of the two, the details
    (critical ply, mode...) are those of the governing direction, and "by_shear_sign" keeps both analyses.

    Buckling factor (ritz_full model): "lambda" is the REPORTED value, the converged Ritz double sine series with
    REPORT_TERMS x REPORT_TERMS = 16 x 16 terms (smaller value over +Nxy / -Nxy), also given as "lambda_16x16";
    "lambda_8x8" is the value with the optimisation terms (problem.ritz_terms). rf["buckling"] and the margin keep
    the rule of the fewest-plies study (min of the optimisation terms and buckling.converge) unless report_lambda is
    True (objective "max_stiffness"): then rf["buckling"] = lambda_16x16 and the margin, the governing criterion
    and the critical ply follow from it."""
    if not problem.both_signs:
        return _analyze_one(problem, full, report_lambda)
    Nx, Ny, Nxy = problem.loads
    one = replace(problem, shear_both_signs=False)
    det = {"+": _analyze_one(one, full, report_lambda),
           "-": _analyze_one(replace(one, loads=(Nx, Ny, -Nxy)), full, report_lambda)}
    sign = "+" if det["+"]["margin"] <= det["-"]["margin"] else "-"
    out = dict(det[sign])
    out["rf"] = {c: min(det["+"]["rf"][c], det["-"]["rf"][c]) for c in det["+"]["rf"]}
    out["margin"] = min(det["+"]["margin"], det["-"]["margin"])
    out["strength_margin"] = min(det["+"]["strength_margin"], det["-"]["strength_margin"])
    out["compliance_c"] = max(det["+"]["compliance_c"], det["-"]["compliance_c"])
    for k in ("lambda_16x16", "lambda_8x8"):
        out[k] = None if det["+"][k] is None else min(det["+"][k], det["-"][k])
    out["lambda"] = out["lambda_16x16"]
    out["governing_shear_sign"] = sign
    out["by_shear_sign"] = {s: {k: d[k] for k in ("rf", "margin", "governing", "critical_ply", "critical_angle",
                                                   "buckling_ritz_full", "lambda_16x16", "lambda_8x8")}
                            for s, d in det.items()}
    return out


def _analyze_one(problem: Problem, full: Sequence[int], report_lambda: bool = False) -> Dict:
    f = np.asarray(full, dtype=int)
    ang = ANGLES[f]
    m = problem.material
    Nx, Ny, Nxy = problem.loads
    res = clt.analyze(ang, problem.ply_thickness, m, Nx, Ny, Nxy)
    stress, strain = res["ply_stress"], res["ply_strain"]
    per_ply: Dict[str, np.ndarray] = {}
    if "max_stress" in problem.criteria:
        per_ply["max_stress"] = failure.rf_max_stress(stress, m)
    if "tsai_wu" in problem.criteria:
        per_ply["tsai_wu"] = failure.rf_tsai_wu(stress, m, problem.f12_star)
    if "max_strain" in problem.criteria:
        per_ply["max_strain"] = failure.rf_max_strain(strain, problem.strain_allowables)
    for c in per_ply:
        per_ply[c] = per_ply[c] / problem.criterion_factors.get(c, 1.0)
    D = res["D"]
    a, b = problem.plate.a, problem.plate.b
    lam_cf, mm, nn, capped = buckling.closed_form([D[0, 0]], [D[0, 1]], [D[1, 1]], [D[2, 2]], Nx, Ny, a, b,
                                                  problem.buckling_modes)
    lam_ritz = float(buckling.ritz([D[0, 0]], [D[0, 1]], [D[1, 1]], [D[2, 2]], Nx, Ny, Nxy, a, b, 20, 20)[0])
    full_model = None
    lam16 = None
    if problem.buckling_model == "ritz_full":
        conv = buckling.converge(D, Nx, Ny, Nxy, a, b, start=6, step=2, tol=0.005)
        mn = buckling.terms_for(problem.ritz_terms, a, b)
        lam_opt = float(buckling.ritz_full(D, Nx, Ny, Nxy, a, b, *mn)[0])
        mn16 = buckling.terms_for(REPORT_TERMS, a, b)
        lam16 = float(buckling.ritz_full(D, Nx, Ny, Nxy, a, b, *mn16)[0])
        full_model = {"lambda_optimization_terms": lam_opt, "optimization_terms": mn,
                      "lambda_converged": conv["lambda"], "converged_terms": (conv["M"], conv["N"]),
                      "converged": conv["converged"], "last_change": conv["last_change"], "sequence": conv["terms"],
                      "lambda_report_terms": lam16, "report_terms": mn16}
    rf = {c: float(np.min(v)) for c, v in per_ply.items()}
    crit_ply = {c: int(np.argmin(v)) + 1 for c, v in per_ply.items()}
    lam_8x8 = None
    if "buckling" in problem.criteria:
        lam_used = lam_ritz if problem.buckling_model == "ritz" else float(lam_cf[0])
        if full_model is not None:   # the lower of the two upper bounds (optimisation terms, converged series)
            lam_used = min(full_model["lambda_converged"], full_model["lambda_optimization_terms"])
            lam_8x8 = full_model["lambda_optimization_terms"] / problem.criterion_factors.get("buckling", 1.0)
            lam16 = lam16 / problem.criterion_factors.get("buckling", 1.0)
            if report_lambda:        # objective max_stiffness: the reported (16 x 16) value
                lam_used = full_model["lambda_report_terms"]
        rf["buckling"] = float(lam_used) / problem.criterion_factors.get("buckling", 1.0)
        if full_model is None:       # other buckling models: one value only
            lam_8x8 = lam16 = rf["buckling"]
    else:
        lam16 = None
    governing = min(rf, key=lambda c: rf[c])
    margin = rf[governing]
    strength_rf = [v for c, v in rf.items() if c != "buckling"]
    ms_ratios = failure.max_stress_ratios(stress, m)
    k = crit_ply.get(governing)
    mode = None
    if governing == "max_stress" or (governing == "tsai_wu" and k):
        mode = failure.MAX_STRESS_MODES[int(np.argmax(ms_ratios[k - 1]))]
    return {
        "n_plies": len(f), "angles": [float(x) for x in ang], "rf": rf, "margin": float(margin),
        "governing": governing, "critical_ply": k, "critical_angle": float(ang[k - 1]) if k else None,
        "max_stress_mode_in_critical_ply": mode, "critical_ply_per_criterion": crit_ply,
        "buckling_mode_mn": (int(mm[0]), int(nn[0])) if np.isfinite(lam_cf[0]) else None,
        "buckling_closed_form": float(lam_cf[0]), "buckling_ritz_with_shear": lam_ritz,
        "buckling_ritz_full": full_model,
        "mode_scan_capped": bool(capped),
        "D16_over_D11": float(abs(D[0, 2]) / D[0, 0]), "D26_over_D22": float(abs(D[1, 2]) / D[1, 1]),
        "ABD": res["ABD"], "max_failure_index": res["max_failure_index"], "critical_ply_matlab_style": res["critical_ply"],
        "thickness_mm": float(len(f) * problem.ply_thickness * 1e3),
        # added for the fixed-thickness objectives (values above unchanged): A [N/m], D [N m], strength margin =
        # min over the strength criteria, c = N . eps0 = N^T A^-1 N [N/m]; lambda = the reported buckling factor
        # (Ritz sine series 16 x 16, = lambda_16x16), lambda_8x8 = with the optimisation terms
        "A": res["A"], "D": D, "strength_margin": float(min(strength_rf)) if strength_rf else float("inf"),
        "lambda": lam16, "lambda_16x16": lam16, "lambda_8x8": lam_8x8,
        "compliance_c": float(res["eps0"] @ np.array([Nx, Ny, Nxy], dtype=float)),
    }


def lexi_rank(n_plies: np.ndarray, margin: np.ndarray, sf: float) -> np.ndarray:
    """Order (best first) for the 'fewest plies' problem: feasible before infeasible; feasible by fewer plies,
    then larger margin; infeasible by smaller violation (sf - margin) / sf, then fewer plies."""
    feas = margin >= sf
    viol = np.where(feas, 0.0, (sf - margin) / sf)
    return np.lexsort((-margin, np.where(feas, n_plies, 0), viol, ~feas))


def better(a: Tuple[int, float], b: Tuple[int, float], sf: float) -> bool:
    """True if design a = (n, margin) is strictly better than b for the fewest-plies problem."""
    fa, fb = a[1] >= sf, b[1] >= sf
    if fa != fb:
        return fa
    if fa:
        return a[0] < b[0] or (a[0] == b[0] and a[1] > b[1])
    return a[1] > b[1] or (a[1] == b[1] and a[0] < b[0])
