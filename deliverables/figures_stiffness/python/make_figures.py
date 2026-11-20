"""
Figures and table of the fixed-thickness stiffest-laminate study (objective max_stiffness, step 2). Nothing is
recomputed: every number comes from deliverables\\optimization\\stiffness_step2\\ (results.json, ga_history.json,
tradeoff_C6_N24_*.npz). Style: copy of the approved figures_v2 standard (style.py in this folder).

VS Code terminal at D:\\CompositeAgent:
    venv_fig\\Scripts\\activate
    python deliverables\\figures_stiffness\\python\\make_figures.py            all figures and the table
    python deliverables\\figures_stiffness\\python\\make_figures.py Fig3       only these

Output: pdf\\*.pdf (vector), png\\*.png (1000 dpi), tables\\stiffness_16_problems.tex / .csv, _report.json (text
collisions, colours, font sizes of every figure).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import style as st  # noqa: E402
from style import PAPER, Page  # noqa: E402

BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
DATA = os.path.join(ROOT, "deliverables", "optimization", "stiffness_step2")
T_KEYS = ("t0125", "t0183")
T_MM = {"t0125": 0.125, "t0183": 0.18288}
SF = 1.5
MINUS = "\u2212"
REPORT = []


# ------------------------------------------------------------------------------------------------ data
def read_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def tk_of(t_mm):
    return min(T_KEYS, key=lambda k: abs(T_MM[k] - float(t_mm)))


class Data:
    def __init__(self):
        res = read_json("results.json")
        self.rows = {(r["case"], tk_of(r["t (mm)"])): r for r in res["rows"]}
        self.probs = {(p["case"], p["thickness"]): p for p in res["problems"]}
        self.cases = list(dict.fromkeys(r["case"] for r in res["rows"]))
        self.ga = read_json("ga_history.json") if os.path.exists(os.path.join(DATA, "ga_history.json")) else []
        self.tradeoff = read_json("tradeoff.json") if os.path.exists(os.path.join(DATA, "tradeoff.json")) else {}

    def kind(self, case):
        return "lambda" if self.rows[(case, "t0125")]["objective"] == "max lambda" else "compliance"


NOT_PROVEN = r"* best known design, search incomplete (not proven)"


def not_proven(r):
    """The exact search did not finish: the value is that of the best known design (status INCOMPLETE)."""
    return (r.get("status") or "").startswith("INCOMPLETE")


def grid(W, H, left, right, bottom, top, gapx, gapy, rows, cols):
    w = (W - left - right - (cols - 1) * gapx) / cols
    h = (H - bottom - top - (rows - 1) * gapy) / rows
    return [[(left + c * (w + gapx), H - top - (r + 1) * h - r * gapy, w, h) for c in range(cols)] for r in range(rows)]


def nice_ticks(hi, n=5):
    raw = hi / n
    e = 10 ** math.floor(math.log10(raw))
    step = min((m * e for m in (1, 2, 2.5, 5, 10) if m * e >= raw), default=10 * e)
    return [step * i for i in range(int(hi / step) + 1)], step


def fmt_tick(v):
    return f"{v:g}" if v >= 0 else MINUS + f"{-v:g}"


def finish(p):
    REPORT.append(p.save(BASE))
    r = REPORT[-1]
    print(p.name, "collisions", len(r["collisions"]), "colours", len(r["colours"]), "min font", r["fonts"]["min_size"],
          flush=True)
    for x in r["collisions"] + r["colours"]:
        print("   ", x)


# ------------------------------------------------------------------------------------- grouped columns
def column_figure(d, s, name, cases, value, ylabel, empty_text):
    """Grouped columns: one group per case, t0125 left, t0183 right; value(row) -> number or None (then
    empty_text(row) is written at the column position)."""
    W, H = st.PAPER_W["single"], 64.0
    p = Page(name, W, H, s)
    vals = {(c, tk): value(d.rows[(c, tk)]) for c in cases for tk in T_KEYS}
    top = max(v for v in vals.values() if v is not None)
    yt, step = nice_ticks(top * 1.12, 5)
    ylim = (0, max(yt[-1], top * 1.18))
    box = (14.5, 15.5, W - 14.5 - 2.0, H - 15.5 - 11.0)
    ax = p.axes(box, (0.45, len(cases) + 0.55), ylim, xticks=range(1, len(cases) + 1), yticks=yt,
                ylabels=[f"{v:g}" for v in yt])
    ax.tick_params(axis="x", length=0)
    w, gap = 0.36, 0.02
    x, y, bw, bh = box
    y1 = y - s.ticklen - s.pad
    y2 = y1 - s.mm(s.tick) * 1.28
    flagged = False
    for i, c in enumerate(cases):
        xm = float(p.X(ax, i + 1))
        p.text(xm, y1, c, s.tick, va="top")
        ns = [d.rows[(c, tk)].get("N (-)") for tk in T_KEYS]
        p.text(xm, y2, " / ".join(str(n) if n else "--" for n in ns), s.barnum, va="top")
        for j, tk in enumerate(T_KEYS):
            xa = i + 1 - gap - w if j == 0 else i + 1 + gap
            xb = xa + w
            v = vals[(c, tk)]
            xc = float(p.X(ax, (xa + xb) / 2))
            if v is None:
                p.text(xc, float(p.Y(ax, 0)) + s.pad * 1.2, empty_text(d.rows[(c, tk)]), s.barnum, va="bottom",
                       rot=90, ha="left")
                continue
            p.bar(ax, xa, xb, v, tk, mark=st.THICK_MARK[tk])
            p.text(xc, float(p.Y(ax, v)) + s.pad * 0.9, f"{v:.1f}", s.barnum, va="bottom")
            if not_proven(d.rows[(c, tk)]):          # best known design only: star above the value
                flagged = True
                p.text(xc, float(p.Y(ax, v)) + s.pad * 0.9 + s.mm(s.barnum) * 1.05, "*", s.barnum, va="bottom")
    p.xlabel(ax, r"Load case and $N$ ($t = 0.125$ / $0.18288\,\mathrm{mm}$)", 1.0)
    p.ylabel(ax, ylabel, 2.2)
    if flagged:
        p.text(W - 2.0, H - 8.2, NOT_PROVEN, s.barnum, ha="right")
    p.key_bar(14.5, H - 3.6, "t0125", st.THICK_TEXT["t0125"], mark=st.THICK_MARK["t0125"])
    p.key_bar(48.0, H - 3.6, "t0183", st.THICK_TEXT["t0183"], mark=st.THICK_MARK["t0183"])
    finish(p)
    return vals


def fig1(d, s):
    cases = [c for c in d.cases if d.kind(c) == "lambda"]
    return column_figure(d, s, "Fig1_lambda_vs_qi", cases, lambda r: r.get("vs QI (%)"),
                         r"Increase of $\lambda$ over QI (\%)", lambda r: "no QI")


def fig2(d, s):
    cases = [c for c in d.cases if d.kind(c) == "compliance"]

    def empty(r):
        return "no feasible $N$" if r.get("N (-)") is None else "no QI with the same $N$"
    return column_figure(d, s, "Fig2_compliance_vs_qi", cases, lambda r: r.get("vs QI (%)"),
                         r"Decrease of $c$ from QI (\%)", empty)


# ------------------------------------------------------------------------------------------ ply strips
def parse_half(design):
    code = {"0": 0, "45": 1, "-45": 2, "90": 3}
    out = []
    for tok in design.strip()[1:-2].split("/"):
        ang, _, k = tok.partition("_")
        out += [code[ang]] * (int(k) if k else 1)
    return out


def strip(p, x, y, cw, sh, half):
    lw = 0.3
    for i, a in enumerate(half):
        p.ov.add_patch(st.Rectangle((x + i * cw, y), cw, sh, fc=st.ROLE[st.PLY_ROLE[a]], ec=st.BLACK, lw=lw,
                                    zorder=12))
    p.rects.append((x, y, x + len(half) * cw, y + sh))


def value_text(r, which):
    if r["objective"] == "max lambda":
        v = {"stiffest": r.get("lambda 16x16 (-)"), "strongest": r.get("max_margin lambda 16x16 (-)"),
             "qi": r.get("QI lambda 16x16 (-)")}[which]
        return rf"$\lambda = {v:.3f}$"
    v = {"stiffest": r.get("c (N/m)"), "strongest": r.get("max_margin c (N/m)"), "qi": r.get("QI c (N/m)")}[which]
    return rf"$c = {v:.0f}\,\mathrm{{N/m}}$"


def fig3(d, s):
    for tk in T_KEYS:
        rows = [d.rows[(c, tk)] for c in d.cases]
        nmax = max((r.get("N (-)") or 0) for r in rows) // 2
        W = st.PAPER_W["double"]
        sh, sgap, cgap = 2.7, 0.9, 3.0
        n_case = len(rows)
        top, bottom = 10.0, 7.5
        H = top + bottom + n_case * (3 * sh + 2 * sgap) + (n_case - 1) * cgap
        p = Page(f"Fig3_stacking_{tk}", W, H, s)
        x_case, x_lab, x0 = 2.0, 22.0, 44.0
        cw = min(4.2, (W - x0 - 27.0) / nmax)
        y = H - top
        flagged = False
        for r in rows:
            yc = y - (3 * sh + 2 * sgap) / 2
            p.text(x_case, yc + s.mm(s.tick) * 0.65, rf"\textbf{{{r['case']}}}", s.tick, ha="left")
            if r.get("N (-)") is None:
                p.text(x_case, yc - s.mm(s.tick) * 0.65, "no feasible $N$", s.barnum, ha="left")
                p.text(x_lab, yc, r"No laminate of $8$ to $64$ plies meets both constraints (no QI, no fewest-plies "
                                  r"design)", s.barnum, ha="left")
                y -= 3 * sh + 2 * sgap + cgap
                continue
            p.text(x_case, yc - s.mm(s.tick) * 0.65, rf"$N = {r['N (-)']}$", s.barnum, ha="left")
            qi_lab = "QI" if r.get("QI feasible") else r"QI (fails)"
            for which, lab, seq in (("stiffest", "Stiffest", r["max_stiffness sequence"]),
                                    ("strongest", "Strongest", r["max_margin sequence"]),
                                    ("qi", qi_lab, r["QI sequence"].replace("_", "_") if r.get("QI sequence") else None)):
                ys = y - sh
                if seq:
                    half = parse_half(seq) if not seq.startswith("[0/45/-45/90]_") else [0, 1, 2, 3] * (r["N (-)"] // 8)
                    strip(p, x0, ys, cw, sh, half)
                    star = "*" if (which == "stiffest" and not_proven(r)) else ""
                    flagged = flagged or bool(star)
                    p.text(x0 + len(half) * cw + 1.5, ys + sh / 2, value_text(r, which) + star, s.barnum, ha="left")
                p.text(x_lab, ys + sh / 2, lab, s.barnum, ha="left")
                y -= sh + sgap
            y += sgap
            y -= cgap
        # direction of the strips and colour key
        p.text(x0, bottom - 3.0, r"Outer ply $\rightarrow$ mid-plane (half laminate, symmetric)", s.barnum,
               ha="left")
        if flagged:
            p.text(W - 2.0, bottom - 3.0, NOT_PROVEN, s.barnum, ha="right")
        xk = x0
        for a in range(4):
            p.ov.add_patch(st.Rectangle((xk, H - 3.6 - 1.3), 3.2, 2.6, fc=st.ROLE[st.PLY_ROLE[a]], ec=st.BLACK,
                                        lw=0.3, zorder=12))
            p.rects.append((xk, H - 4.9, xk + 3.2, H - 2.3))
            p.text(xk + 4.2, H - 3.6, st.PLY_TEXT[a], s.legend, ha="left")
            xk += 19.0
        p.text(W - 2.0, H - 3.6, st.THICK_TEXT[tk], s.legend, ha="right")
        finish(p)


# --------------------------------------------------------------------------------------------- trade-off
def occupancy(x, y, xlim, ylim, nx, ny):
    """Grid cells that hold at least one design, merged row by row into rectangles (data units)."""
    ix = np.clip(((x - xlim[0]) / (xlim[1] - xlim[0]) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((y - ylim[0]) / (ylim[1] - ylim[0]) * ny).astype(int), 0, ny - 1)
    occ = np.zeros((ny, nx), bool)
    occ[iy, ix] = True
    dx, dy = (xlim[1] - xlim[0]) / nx, (ylim[1] - ylim[0]) / ny
    rects = []
    for j in range(ny):
        row = occ[j]
        i = 0
        while i < nx:
            if row[i]:
                k = i
                while k + 1 < nx and row[k + 1]:
                    k += 1
                rects.append((xlim[0] + i * dx, ylim[0] + j * dy, (k - i + 1) * dx, dy))
                i = k + 1
            else:
                i += 1
    return rects, int(occ.sum())


def fig4(d, s):
    from matplotlib.collections import PatchCollection
    W, H = st.PAPER_W["double"], 84.0
    p = Page("Fig4_tradeoff_C6_N24", W, H, s)
    g = grid(W, H, 15.0, 3.0, 12.0, 13.0, 12.0, 0, 1, 2)
    info = {}
    for i, tk in enumerate(T_KEYS):
        z = np.load(os.path.join(DATA, f"tradeoff_C6_N24_{tk}.npz"))
        lam, rs = z["lambda_8x8"].astype(float), z["strength_margin"].astype(float)
        t = d.tradeoff[tk]
        xlim = (math.floor(rs.min() * 10) / 10, math.ceil(rs.max() * 10) / 10)
        ylim = (math.floor(lam.min() * 10) / 10, math.ceil(lam.max() * 10) / 10)
        xt, _ = nice_ticks(xlim[1] - xlim[0], 5)
        xt = [v for v in np.round(np.arange(math.ceil(xlim[0] * 2) / 2, xlim[1] + 1e-9, 0.5), 6)]
        yt = [v for v in np.round(np.arange(math.ceil(ylim[0] * 2) / 2, ylim[1] + 1e-9, 0.5), 6)]
        if len(yt) > 7:
            yt = yt[::2]
        if len(xt) > 7:
            xt = xt[::2]
        ax = p.axes(g[0][i], xlim, ylim, xticks=xt, yticks=yt, xlabels=[f"{v:g}" for v in xt],
                    ylabels=[f"{v:g}" for v in yt])
        bx, by, bw, bh = g[0][i]
        nx, ny = int(bw / 0.45), int(bh / 0.45)
        rects, n_cells = occupancy(rs, lam, xlim, ylim, nx, ny)
        pc = PatchCollection([st.Rectangle((a, b), c, e) for a, b, c, e in rects], fc=st.ROLE["cloud"],
                             ec=st.ROLE["cloud"], lw=0.15, zorder=2)
        ax.add_collection(pc)
        # constraint lines (safety factor 1.5 on both)
        ax.plot([SF, SF], ylim, ls="--", lw=s.thin, color=st.BLACK, zorder=3)
        ax.plot(xlim, [SF, SF], ls="--", lw=s.thin, color=st.BLACK, zorder=3)
        p.segs.append(((float(p.X(ax, SF)), by), (float(p.X(ax, SF)), by + bh)))
        p.segs.append(((bx, float(p.Y(ax, SF))), (bx + bw, float(p.Y(ax, SF)))))
        # Pareto front (staircase)
        px, py = np.array(t["pareto_strength_margin"]), np.array(t["pareto_lambda_8x8"])
        o = np.argsort(px)
        px, py = px[o], py[o]
        sx = np.repeat(px, 2)[1:]
        sy = np.repeat(py, 2)[:-1]
        p.line(ax, sx, sy, "pareto", lw=s.lw * 0.8, label=f"Pareto {tk}", zorder=5)
        for role, xv, yv, mk in (("qi", t["qi_strength_margin"], t["qi_lambda_8x8"], "D"),
                                 ("stiffest", t["stiffest_strength_margin"], t["stiffest_lambda_8x8"], "o"),
                                 ("strongest", t["strongest_strength_margin"], t["strongest_lambda_8x8"], "s")):
            p.line(ax, [xv], [yv], role, ls="", marker=mk, ms=s.ms * 1.25, mec=st.WHITE, mew=0.6, zorder=8,
                   label=f"{role} {tk}")
        p.panel(ax, "ab"[i], st.THICK_TEXT[tk] + (r", default $N$ of C6 is $32$" if tk == "t0125" else
                                                    r", default $N$ of C6"))
        p.xlabel(ax, "Strength margin (--)", 1.0)
        info[tk] = {"cells": n_cells, "rects": len(rects), "n": int(len(rs))}
    p.ylabel(p.fig.axes[1], r"$\lambda$, Ritz $8 \times 8$ terms (--)", 2.6)
    xs = (15.0, 47.0, 77.0, 105.0, 145.0)
    y = H - 3.6
    p.key_bar(xs[0], y, "cloud", "Valid designs")
    p.key_line(xs[1], y, "stiffest", "Stiffest", ls="", marker="o", mec=st.WHITE, mew=0.6, ms=s.ms * 1.25)
    p.key_line(xs[2], y, "strongest", "Strongest", ls="", marker="s", mec=st.WHITE, mew=0.6, ms=s.ms * 1.25)
    p.key_line(xs[3], y, "qi", "QI", ls="", marker="D", mec=st.WHITE, mew=0.6, ms=s.ms * 1.25)
    p.key_line(xs[3] + 18.0, y, "pareto", "Pareto front", lw=s.lw * 0.8)
    p.key_line(xs[4] + 10.0, y, "exact", r"Limit $1.5$", ls="--", lw=s.thin)
    finish(p)
    return info


# ---------------------------------------------------------------------------------------- GA convergence
def best_so_far(h, grid_x):
    """lambda (16 x 16) of the run's best design so far at each number of evaluations (NaN before the first
    design meeting both constraints; held after the last change)."""
    ev = np.array(h["evals"], float)
    val = np.array([v if (v is not None and f) else np.nan for v, f in zip(h["lambda_16x16"], h["feasible"])],
                   float)
    idx = np.searchsorted(ev, grid_x, side="right") - 1
    out = np.where(idx >= 0, val[np.clip(idx, 0, None)], np.nan)
    return out


def fig5(d, s, picks):
    W, H = st.PAPER_W["double"], 66.0
    p = Page("Fig5_ga_convergence", W, H, s)
    g = grid(W, H, 15.0, 6.0, 12.0, 13.0, 9.0, 0, 1, len(picks))     # right 6 mm: the last tick label 10000
    x0, x1 = 50, 10000
    gx = np.unique(np.round(np.logspace(np.log10(x0), np.log10(x1), 400)).astype(int))
    xt = (100, 300, 1000, 3000, 10000)
    info = {}
    for i, (c, tk) in enumerate(picks):
        r = d.rows[(c, tk)]
        opt = r["lambda 16x16 (-)"]
        curves = {}
        lo_all, hi_all = [], []
        for algo, m in (("GA", "ga"), ("GA + LS", "ga_ls")):
            runs = [x for x in d.ga if (x["case"], x["thickness"], x["method"]) == (c, tk, m)]
            Y = np.array([best_so_far(x["history"], gx) for x in runs])
            ok = ~np.isnan(Y).any(axis=0)
            curves[algo] = (gx[ok], np.median(Y[:, ok], axis=0), Y[:, ok].min(axis=0), Y[:, ok].max(axis=0),
                            len(runs))
            lo_all.append(np.nanmin(Y))
            hi_all.append(np.nanmax(Y))
        lo, hi = min(lo_all + [opt]), max(hi_all + [opt])
        span = hi - lo
        ylim = (lo - 0.06 * span, hi + 0.08 * span)
        yt, step = nice_ticks(ylim[1] - ylim[0], 4)
        first = math.ceil(ylim[0] / step) * step
        yt = [round(first + k * step, 10) for k in range(10) if first + k * step <= ylim[1]]
        ax = p.axes(g[0][i], (x0, x1), ylim, xlog=True, xticks=xt, yticks=yt, xlabels=[f"{v:g}" for v in xt],
                    ylabels=[f"{v:.{max(0, -int(math.floor(math.log10(step))))}f}" for v in yt])
        p.hline(ax, opt, "exact", ls=":")
        for algo in ("GA", "GA + LS"):
            xx, med, mn, mx, n = curves[algo]
            p.line(ax, xx, med, algo, ls="-", label=f"{c} {tk} {algo} median")
            p.line(ax, xx, mn, algo, ls="--", lw=s.thin, label=f"{c} {tk} {algo} min")
            p.line(ax, xx, mx, algo, ls="--", lw=s.thin, label=f"{c} {tk} {algo} max")
        p.panel(ax, "abc"[i], rf"{c}, {st.THICK_TEXT[tk]}, $N = {r['N (-)']}$")
        p.xlabel(ax, "Distinct designs evaluated", 1.0)
        info[f"{c} {tk}"] = {"optimum_lambda_16x16": opt}
    p.ylabel(p.fig.axes[1], r"$\lambda$ of the best design so far (--)", 2.6)
    y = H - 3.6
    p.key_line(15.0, y, "GA", "GA, median", ls="-")
    p.key_line(48.0, y, "GA + LS", "GA + local search, median", ls="-")
    p.key_line(103.0, y, "exact", r"Min and max of $20$ seeds", ls="--", lw=s.thin)
    p.key_line(150.0, y, "exact", "Proven optimum", ls=":", lw=s.ref_lw)
    finish(p)
    return info


# -------------------------------------------------------------------------------------------------- table
def seq_tex(seq):
    if not seq:
        return "--"
    body = seq.strip()
    if body.startswith("[0/45/-45/90]_"):
        k = body.split("_")[1].rstrip("s")
        return rf"$[0/45/{MINUS}45/90]_{{{k}\mathrm{{s}}}}$".replace(MINUS, "-")
    body = re.sub(r"_(\d+)", r"_{\1}", body[1:-2])
    return rf"$[{body}]_{{\mathrm{{s}}}}$"


def status_short(r):
    s = r.get("status") or ""
    if s.startswith("proven optimum"):
        return "proven (count level)" if "count-level" in s else "proven ($8 \\times 8$ Ritz)"
    if s.startswith("INCOMPLETE"):
        m = re.search(r"gap ([0-9.]+) %", s)
        return rf"INCOMPLETE (gap ${m.group(1)}\,\%$)" if m else "INCOMPLETE"
    if s.startswith("INFEASIBLE") or "infeasible" in s.lower():
        return "infeasible"
    return s


def table(d):
    os.makedirs(os.path.join(BASE, "tables"), exist_ok=True)
    rows = []
    for c in d.cases:
        for tk in T_KEYS:
            r = d.rows[(c, tk)]
            stiff = "max_stiffness sequence" in r
            lam = r.get("objective") == "max lambda"
            rows.append({"case": c, "t (mm)": T_MM[tk], "N (-)": r.get("N (-)"), "N from": r.get("N from"),
                         "objective": "max lambda" if lam else "min c",
                         "stiffest sequence": r.get("max_stiffness sequence"),
                         "lambda 16x16 (-)": r.get("lambda 16x16 (-)") if (stiff and lam) else None,
                         "c (N/m)": r.get("c (N/m)") if (stiff and not lam) else None,
                         "strength margin (-)": r.get("strength margin (-)"), "vs QI (%)": r.get("vs QI (%)"),
                         "vs QI meaning": r.get("vs QI meaning"), "status": r.get("status"),
                         "time (s)": r.get("time max_stiffness (s)")})
    with open(os.path.join(BASE, "tables", "stiffness_16_problems.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = [r"% Fixed-thickness stiffest laminate (objective max_stiffness), 16 problems at the default ply count.",
         r"% Generated by deliverables/figures_stiffness/python/make_figures.py from",
         r"% deliverables/optimization/stiffness_step2/results.json. Needs \usepackage{booktabs}.",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Stiffest laminate of fixed thickness for the 16 problems (IM7/8552, $0.5 \times 0.5$~m simply "
         r"supported plate, safety factor $1.5$ on strength and buckling, shear of both signs). $\lambda$: Ritz sine "
         r"series $16 \times 16$ terms; optimality proven in the $8 \times 8$ Ritz model; when the load cannot buckle "
         r"the plate the in-plane compliance $c$ is minimised. vs QI: increase of $\lambda$ or decrease of $c$ "
         r"against $[0/45/-45/90]_{k\mathrm{s}}$ with the same $N$.}",
         r"\label{tab:stiffness16}",
         r"\begin{tabular}{llrlp{62mm}rrrl}", r"\toprule",
         r"Case & $t$ (mm) & $N$ & $N$ from & Stiffest stacking sequence & $\lambda$ or $c$ (N/m) & "
         r"Strength margin & vs QI (\%) & Status \\", r"\midrule"]
    for r in rows:
        if r["stiffest sequence"] is None:
            L.append(rf"{r['case']} & {r['t (mm)']:g} & -- & -- & \multicolumn{{5}}{{l}}{{infeasible: no QI and no "
                     rf"fewest-plies design with $8$ to $64$ plies}} \\")
            continue
        val = (f"{r['lambda 16x16 (-)']:.4f}" if r["lambda 16x16 (-)"] is not None else f"{r['c (N/m)']:.1f}")
        vs = f"{r['vs QI (%)']:+.2f}" if r["vs QI (%)"] is not None else "no QI with the same $N$"
        nfrom = "QI$_{ks}$" if r["N from"] == "QI_ks" else "min\\_plies"
        L.append(rf"{r['case']} & {r['t (mm)']:g} & {r['N (-)']} & {nfrom} & {seq_tex(r['stiffest sequence'])} & "
                 rf"{val} & {r['strength margin (-)']:.3f} & {vs} & {status_short(r)} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    with open(os.path.join(BASE, "tables", "stiffness_16_problems.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return rows


# --------------------------------------------------------------------------------------------------- main
def main():
    st.use_style()
    d = Data()
    want = set(sys.argv[1:])

    def on(name):
        return not want or name in want

    s = PAPER
    info = {}
    if on("Fig1"):
        info["Fig1"] = {f"{k[0]} {k[1]}": v for k, v in fig1(d, s).items()}
    if on("Fig2"):
        info["Fig2"] = {f"{k[0]} {k[1]}": v for k, v in fig2(d, s).items()}
    if on("Fig3"):
        fig3(d, s)
    if on("Fig4") and d.tradeoff:
        info["Fig4"] = fig4(d, s)
    if on("Fig5") and d.ga:
        picks = [tuple(x.split()) for x in os.environ.get("FIG5_PICKS", "C7 t0125,C6 t0125,C5 t0183").split(",")]
        info["Fig5"] = fig5(d, s, picks)
    if on("Table"):
        table(d)
    with open(os.path.join(BASE, "_report.json"), "w", encoding="utf-8") as f:
        json.dump({"figures": [{k: v for k, v in r.items() if k != "series"} for r in REPORT], "info": info}, f,
                  indent=1, default=str)
    bad = sum(len(r["collisions"]) + len(r["colours"]) + len(r["fonts"]["bad"]) for r in REPORT)
    print(f"{len(REPORT)} figure(s), {bad} problem(s) found by the checks")


if __name__ == "__main__":
    main()
