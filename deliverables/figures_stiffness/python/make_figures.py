"""
Figures and table of the fixed-thickness stiffest-laminate study (objective max_stiffness, step 2). Nothing is
recomputed: every number comes from deliverables\\optimization\\stiffness_step2\\ (results.csv, ga_history.json,
tradeoff_figure.json, tradeoff_C6_N24_*.npz). Style: copy of the approved figures_v2 standard (style.py in this folder).

VS Code terminal at D:\\CompositeAgent:
    venv_fig\\Scripts\\activate
    python deliverables\\figures_stiffness\\python\\make_figures.py            all figures and the table
    python deliverables\\figures_stiffness\\python\\make_figures.py Fig3       only these

Output: pdf\\*.pdf (vector), png\\*.png (1000 dpi), tables\\stiffness_16_problems.tex / .csv, _report.json (text
collisions, colours, font sizes of every figure).

Percentages against QI: column 'vs QI 1 decimal (%)' of results.csv (rounded once from the unrounded 'vs QI (%)');
figures and table print that column and never round again.
"""
from __future__ import annotations

import collections
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
from style import PAPER, PT_MM, Page  # noqa: E402

BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
DATA = os.path.join(ROOT, "deliverables", "optimization", "stiffness_step2")
T_KEYS = ("t0125", "t0183")
T_MM = {"t0125": 0.125, "t0183": 0.18288}
T_SHORT = {"t0125": "0.125", "t0183": "0.18288"}
SF = 1.5
VS_QI = "vs QI 1 decimal (%)"
REPORT = []


# ------------------------------------------------------------------------------------------------ data
def read_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def tk_of(t_mm):
    return min(T_KEYS, key=lambda k: abs(T_MM[k] - float(t_mm)))


def _cell(v):
    """A results.csv cell: '' -> None, 'True'/'False' -> bool, numbers -> float, other text unchanged."""
    if v is None or v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


class Data:
    def __init__(self):
        with open(os.path.join(DATA, "results.csv"), encoding="utf-8") as f:
            rows = [{k: _cell(v) for k, v in r.items()} for r in csv.DictReader(f)]
        for r in rows:
            if r.get("N (-)") is not None:
                r["N (-)"] = int(r["N (-)"])
        self.rows = {(r["case"], tk_of(r["t (mm)"])): r for r in rows}
        self.cases = list(dict.fromkeys(r["case"] for r in rows))
        self.ga = read_json("ga_history.json") if os.path.exists(os.path.join(DATA, "ga_history.json")) else []
        path = os.path.join(DATA, "tradeoff_figure.json")
        self.tradeoff = read_json("tradeoff_figure.json") if os.path.exists(path) else {}

    def kind(self, case):
        return "lambda" if self.rows[(case, "t0125")]["objective"] == "max lambda" else "compliance"


NOT_PROVEN = r"* best known design, search incomplete (not proven)"


def not_proven(r):
    """The exact search did not finish: the value is that of the best known design (status INCOMPLETE)."""
    return (r.get("status") or "").startswith("INCOMPLETE")


def thousands(n):
    """1864416 -> 1\\,864\\,416 (LaTeX thin spaces)."""
    return f"{int(n):,}".replace(",", r"\,")


def text_size_mm(p, txt, size):
    """Width and height (mm) of a text before it is placed."""
    t = p.ov.text(0, 0, txt, fontsize=size)
    e = t.get_window_extent(p.fig.canvas.get_renderer())
    t.remove()
    k = 25.4 / p.fig.dpi
    return (e.x1 - e.x0) * k, (e.y1 - e.y0) * k


def grid(W, H, left, right, bottom, top, gapx, gapy, rows, cols):
    w = (W - left - right - (cols - 1) * gapx) / cols
    h = (H - bottom - top - (rows - 1) * gapy) / rows
    return [[(left + c * (w + gapx), H - top - (r + 1) * h - r * gapy, w, h) for c in range(cols)] for r in range(rows)]


def nice_ticks(hi, n=5):
    raw = hi / n
    e = 10 ** math.floor(math.log10(raw))
    step = min((m * e for m in (1, 2, 2.5, 5, 10) if m * e >= raw), default=10 * e)
    return [step * i for i in range(int(hi / step) + 1)], step


def ticks_to(need, n=5):
    """Ticks from 0 whose last tick (the axis limit) is the first one >= need: the top of the axis has a number."""
    _, step = nice_ticks(need, n)
    k = math.ceil(need / step - 1e-9)
    return [step * i for i in range(k + 1)]


def finish(p):
    REPORT.append(p.save(BASE))
    r = REPORT[-1]
    print(p.name, "collisions", len(r["collisions"]), "colours", len(r["colours"]), "min font", r["fonts"]["min_size"],
          flush=True)
    for x in r["collisions"] + r["colours"]:
        print("   ", x)


def keys_row(p, x, y, items, gap=6.0):
    """Legend items placed one after the other from x: (kind, role, label, kwargs), kind 'line' or 'bar'."""
    s = p.s
    for kind, role, label, kw in items:
        if kind == "bar":
            p.key_bar(x, y, role, label, **kw)
            sample = s.sample * 0.8
        else:
            p.key_line(x, y, role, label, **kw)
            sample = s.sample if kw.get("ls", "-") not in ("", "None") else s.sample * 0.45
        x += sample + 1.2 * s.pad + text_size_mm(p, label, s.legend)[0] + gap
    return x


# ------------------------------------------------------------------------------------- grouped columns
def column_figure(d, s, name, cases, ylabel_lines, empty_text):
    """Grouped columns of the column 'vs QI 1 decimal (%)': one group per case, t0125 left, t0183 right; a missing
    value is replaced by empty_text(row, tk), written upwards at the column position."""
    W, H = st.PAPER_W["single"], 72.0
    p = Page(name, W, H, s)
    vals = {(c, tk): d.rows[(c, tk)].get(VS_QI) for c in cases for tk in T_KEYS}
    top = max(v for v in vals.values() if v is not None)
    yt = ticks_to(top * 1.12, 5)
    ylim = (0, yt[-1])
    left = 14.5 + 3.4 * (len(ylabel_lines) - 1)
    box = (left, 15.5, W - left - 2.0, H - 15.5 - 11.0)
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
                p.text(xc, float(p.Y(ax, 0)) + s.pad * 1.2, empty_text(d.rows[(c, tk)], tk), s.barnum, va="bottom",
                       rot=90, ha="left")
                continue
            p.bar(ax, xa, xb, v, tk, mark=st.THICK_MARK[tk])
            p.text(xc, float(p.Y(ax, v)) + s.pad * 0.9, f"{v:.1f}", s.barnum, va="bottom")
            if not_proven(d.rows[(c, tk)]):          # best known design only: star above the value
                flagged = True
                p.text(xc, float(p.Y(ax, v)) + s.pad * 0.9 + s.mm(s.barnum) * 1.05, "*", s.barnum, va="bottom")
    p.xlabel(ax, r"Load case and $N$ ($t = 0.125$ / $0.18288\,\mathrm{mm}$)", 1.0)
    for k, txt in enumerate(ylabel_lines):
        p.ylabel(ax, txt, 1.6 + 3.4 * k)
    p.key_bar(left, H - 3.6, "t0125", st.THICK_TEXT["t0125"], mark=st.THICK_MARK["t0125"])
    p.key_bar(left + 33.5, H - 3.6, "t0183", st.THICK_TEXT["t0183"], mark=st.THICK_MARK["t0183"])
    if flagged:
        p.text(W - 2.0, H - 8.2, NOT_PROVEN, s.barnum, ha="right")
    finish(p)
    return vals


def fig1(d, s):
    cases = [c for c in d.cases if d.kind(c) == "lambda"]
    return column_figure(d, s, "Fig1_lambda_vs_qi", cases,
                         [r"Increase of $\lambda$ (Ritz $16 \times 16$)", r"over QI (\%)"], lambda r, tk: "no QI")


def fig2(d, s):
    cases = [c for c in d.cases if d.kind(c) == "compliance"]

    def empty(r, tk):
        why = r"no feasible $N$" if r.get("N (-)") is None else r"no QI with the same $N$"
        return rf"$t = {T_SHORT[tk]}$: {why}"
    return column_figure(d, s, "Fig2_compliance_vs_qi", cases, [r"Decrease of $c$ from QI (\%)"], empty)


# ------------------------------------------------------------------------------------------ ply strips
def parse_half(design):
    code = {"0": 0, "45": 1, "-45": 2, "90": 3}
    out = []
    for tok in design.strip()[1:-2].split("/"):
        ang, _, k = tok.partition("_")
        out += [code[ang]] * (int(k) if k else 1)
    return out


def half_of(seq, n):
    return parse_half(seq) if not seq.startswith("[0/45/-45/90]_") else [0, 1, 2, 3] * (n // 8)


def strip(p, x, y, cw, sh, half):
    lw = 0.3
    for i, a in enumerate(half):
        p.ov.add_patch(st.Rectangle((x + i * cw, y), cw, sh, fc=st.ROLE[st.PLY_ROLE[a]], ec=st.BLACK, lw=lw,
                                    zorder=12))
    p.rects.append((x, y, x + len(half) * cw, y + sh))


VALUE_COLS = {"lambda": {"stiffest": "lambda 16x16 (-)", "strongest": "max_margin lambda 16x16 (-)",
                         "qi": "QI lambda 16x16 (-)"},
              "c": {"stiffest": "c (N/m)", "strongest": "max_margin c (N/m)", "qi": "QI c (N/m)"},
              "margin": {"stiffest": "strength margin (-)", "strongest": "max_margin strength margin (-)",
                         "qi": "QI strength margin (-)"}}


def value_text(r, which, star=""):
    """lambda (16 x 16) or c, and the strength margin, of one row of figure 3 (results.csv)."""
    m = r.get(VALUE_COLS["margin"][which])
    if r["objective"] == "max lambda":
        v = r.get(VALUE_COLS["lambda"][which])
        head = rf"$\lambda = {v:.3f}${star}"
    else:
        v = r.get(VALUE_COLS["c"][which])
        head = rf"$c = {v:.0f}\,\mathrm{{N/m}}$"
    return head + rf", margin $= {m:.3f}$"


def fig3_notes(rows):
    """Notes on stiffest vs strongest, checked on the data of this figure (not assumed)."""
    mirror, same_counts, other = [], [], []
    for r in rows:
        if r.get("N (-)") is None:
            continue
        a, b = half_of(r["max_stiffness sequence"], r["N (-)"]), half_of(r["max_margin sequence"], r["N (-)"])
        vals_equal = all(r.get(VALUE_COLS[k]["stiffest"]) == r.get(VALUE_COLS[k]["strongest"])
                         for k in ("lambda", "c", "margin"))
        if [{1: 2, 2: 1}.get(x, x) for x in b] == a and vals_equal:
            mirror.append(r["case"])
        elif r["objective"] == "min c":
            if collections.Counter(a) == collections.Counter(b) and vals_equal:
                same_counts.append(r["case"])
            else:
                other.append(r)
    out = []
    if mirror:
        out.append(f"{', '.join(mirror)}: stiffest and strongest are mirror images ($+45$ and $-45$ swapped) with "
                   f"identical values.")
    if same_counts:
        out.append(f"{', '.join(same_counts)}: $c$ and margin depend only on ply counts; stiffest and strongest have "
                   f"the same counts, only the ply order differs.")
    for r in other:
        out.append(rf"{r['case']}: stiffest and strongest have different ply counts (margin "
                   rf"${r[VALUE_COLS['margin']['stiffest']]:.3f}$ vs ${r[VALUE_COLS['margin']['strongest']]:.3f}$, "
                   rf"$c = {r[VALUE_COLS['c']['stiffest']]:.0f}$ vs ${r[VALUE_COLS['c']['strongest']]:.0f}\,"
                   rf"\mathrm{{N/m}}$).")
    return out


def fig3(d, s):
    for tk in T_KEYS:
        rows = [d.rows[(c, tk)] for c in d.cases]
        notes = fig3_notes(rows)
        nmax = max((r.get("N (-)") or 0) for r in rows) // 2
        W = st.PAPER_W["double"]
        sh, sgap, cgap = 2.7, 0.9, 3.0
        n_case = len(rows)
        lead = s.mm(s.barnum) * 1.55
        top, bottom = 10.0, 9.0 + lead * (len(notes) + 1)
        H = top + bottom + n_case * (3 * sh + 2 * sgap) + (n_case - 1) * cgap
        p = Page(f"Fig3_stacking_{tk}", W, H, s)
        x_case, x_lab, x0 = 2.0, 22.0, 44.0
        widest = max(text_size_mm(p, value_text(r, w, "*"), s.barnum)[0] for r in rows if r.get("N (-)")
                     for w in ("stiffest", "strongest", "qi"))
        cw = min(4.2, (W - x0 - 1.5 - widest - 1.0) / nmax)
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
                                    ("qi", qi_lab, r.get("QI sequence"))):
                ys = y - sh
                if seq:
                    half = half_of(seq, r["N (-)"])
                    strip(p, x0, ys, cw, sh, half)
                    star = "*" if (which == "stiffest" and not_proven(r)) else ""
                    flagged = flagged or bool(star)
                    p.text(x0 + len(half) * cw + 1.5, ys + sh / 2, value_text(r, which, star), s.barnum, ha="left")
                p.text(x_lab, ys + sh / 2, lab, s.barnum, ha="left")
                y -= sh + sgap
            y += sgap
            y -= cgap
        # direction of the strips, notes, colour key
        yb = 2.5
        for txt in reversed(notes):
            p.text(x_case, yb, txt, s.barnum, ha="left", va="bottom")
            yb += lead
        yb += lead * 0.35
        p.text(x0, yb, r"Outer ply $\rightarrow$ mid-plane (half laminate, symmetric)", s.barnum, ha="left",
               va="bottom")
        if flagged:
            p.text(W - 2.0, yb, NOT_PROVEN, s.barnum, ha="right", va="bottom")
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
def free_window(blocked, lo, hi, need):
    """Lowest interval [y, y + need] inside [lo, hi] that meets none of the blocked (y0, y1) intervals (mm)."""
    y = lo
    for a, b in sorted(blocked):
        if b <= y:
            continue
        if a >= y + need:
            break
        y = max(y, b)
    return y if y + need <= hi else None


def fig4(d, s):
    W, H = st.PAPER_W["double"], 100.0
    p = Page("Fig4_tradeoff_C6_N24", W, H, s)
    lead = s.mm(s.barnum) * 1.55
    bottom = 13.5 + 3 * lead
    g = grid(W, H, 15.0, 3.0, bottom, 13.0, 12.0, 0, 1, 2)
    info = {}
    lines = ["infeasible", r"(margin $< 1.5$)"]
    tw = max(text_size_mm(p, x, s.barnum)[0] for x in lines)
    for i, tk in enumerate(T_KEYS):
        t = d.tradeoff[tk]
        z = np.load(os.path.join(DATA, f"tradeoff_C6_N24_{tk}.npz"))
        lam, rs = z["lambda_8x8"].astype(float), z["strength_margin"].astype(float)
        xlim = (math.floor(rs.min() * 10) / 10, math.ceil(rs.max() * 10) / 10)
        bw = g[0][i][2]
        need_left = tw + 2.2              # room for the 'infeasible' label between the axis and the 1.5 line
        if (SF - xlim[0]) / (xlim[1] - xlim[0]) * bw < need_left:
            x0 = (SF * bw - need_left * xlim[1]) / (bw - need_left)
            xlim = (math.floor(x0 * 20) / 20, xlim[1])
        ylim = (math.floor(lam.min() * 10) / 10, math.ceil(lam.max() * 10) / 10)
        xt = [v for v in np.round(np.arange(math.ceil(xlim[0] * 2) / 2, xlim[1] + 1e-9, 0.5), 6)]
        yt = [v for v in np.round(np.arange(math.ceil(ylim[0] * 2) / 2, ylim[1] + 1e-9, 0.5), 6)]
        if len(yt) > 7:
            yt = yt[::2]
        if len(xt) > 7:
            xt = xt[::2]
        ax = p.axes(g[0][i], xlim, ylim, xticks=xt, yticks=yt, xlabels=[f"{v:g}" for v in xt],
                    ylabels=[f"{v:g}" for v in yt])
        bx, by, bw, bh = g[0][i]
        # valid designs: one solid line per strength-margin value (= per ply-count tuple), lambda 8x8 min to max
        lw_cloud = s.lw * 0.8
        for gr in t["groups"]:
            p.line(ax, [gr["strength_margin"]] * 2, [gr["lambda_8x8_min"], gr["lambda_8x8_max"]], "cloud",
                   lw=lw_cloud, zorder=2, label=f"designs {tk} margin {gr['strength_margin']:.6f}")
        # constraint lines (safety factor 1.5 on both), drawn and registered only where they are inside the axes
        h_in = ylim[0] < SF < ylim[1]
        if xlim[0] < SF < xlim[1]:
            ax.plot([SF, SF], ylim, ls="--", lw=s.thin, color=st.BLACK, zorder=3)
            p.segs.append(((float(p.X(ax, SF)), by), (float(p.X(ax, SF)), by + bh)))
        if h_in:
            ax.plot(xlim, [SF, SF], ls="--", lw=s.thin, color=st.BLACK, zorder=3)
            p.segs.append(((bx, float(p.Y(ax, SF))), (bx + bw, float(p.Y(ax, SF)))))
        # Pareto front, both objectives maximised: lambda 8x8 reachable with a strength margin >= x. From each
        # Pareto point the horizontal step runs LEFT to the previous point (smaller margin, larger lambda).
        px = np.array([a for a, _ in t["pareto"]])
        py = np.array([b for _, b in t["pareto"]])
        o = np.argsort(px)
        px, py = px[o], py[o]
        sx, sy = np.repeat(px, 2)[:-1], np.repeat(py, 2)[1:]
        p.line(ax, sx, sy, "pareto", lw=s.lw * 0.8, label=f"Pareto staircase {tk}", zorder=5)
        p.line(ax, px, py, "pareto", ls="", marker="o", ms=s.ms_small, mec=st.ROLE["pareto"], mew=0, zorder=6,
               label=f"Pareto points {tk}")
        for role, xv, yv, mk in (("qi", t["qi_strength_margin"], t["qi_lambda_8x8"], "D"),
                                 ("stiffest", t["stiffest_strength_margin"], t["stiffest_lambda_8x8"], "o"),
                                 ("strongest", t["strongest_strength_margin"], t["strongest_lambda_8x8"], "s")):
            p.line(ax, [xv], [yv], role, ls="", marker=mk, ms=s.ms * 1.25, mec=st.WHITE, mew=0.6, zorder=8,
                   label=f"{role} {tk}")
        # 'infeasible' label just left of the vertical limit, in the lowest free place
        lh = s.mm(s.barnum) * 1.15
        need = lh * len(lines) + 0.4
        xr = float(p.X(ax, SF)) - 1.0
        xl = xr - tw
        half_lw = lw_cloud * PT_MM / 2 + 0.4
        blocked = [(p.Y(ax, gr["lambda_8x8_min"]) - 0.5, p.Y(ax, gr["lambda_8x8_max"]) + 0.5) for gr in t["groups"]
                   if xl - half_lw <= p.X(ax, gr["strength_margin"]) <= xr + half_lw]
        if h_in:
            blocked.append((p.Y(ax, SF) - 0.8, p.Y(ax, SF) + 0.8))
        for j in range(len(sx) - 1):
            xa, xb = sorted((float(p.X(ax, sx[j])), float(p.X(ax, sx[j + 1]))))
            if xb >= xl - 0.5 and xa <= xr + 0.5:
                ya, yb2 = sorted((float(p.Y(ax, sy[j])), float(p.Y(ax, sy[j + 1]))))
                blocked.append((ya - 0.8, yb2 + 0.8))
        for xm, ym, r in p.marks:
            if xl - r - 0.5 <= xm <= xr + r + 0.5:
                blocked.append((ym - r - 0.5, ym + r + 0.5))
        y0 = free_window([(float(a), float(b)) for a, b in blocked], by + 0.5, by + bh - 0.5, need)
        if y0 is None:
            raise RuntimeError(f"no free place for the 'infeasible' label ({tk})")
        for k, txt in enumerate(lines):
            p.text(xr, y0 + need - 0.2 - (k + 0.5) * lh, txt, s.barnum, ha="right")
        dn = t["default_N"]
        p.panel(ax, "ab"[i], st.THICK_TEXT[tk] + rf", default $N$ of C6 is ${dn}$")
        p.xlabel(ax, "Strength margin (--)", bottom - 8.6)
        info[tk] = {"groups": len(t["groups"]), "n": int(len(rs)), "pareto": t["pareto"],
                    "strongest": [t["strongest_strength_margin"], t["strongest_lambda_8x8"]],
                    "n_tied": t["n_tied_for_strongest"], "infeasible_label_y_mm": y0}
    p.ylabel(p.fig.axes[1], r"$\lambda$, Ritz $8 \times 8$ terms (--)", 2.6)
    # notes outside the plots
    yb = 1.5
    p.text(15.0, yb, "Designs with the same ply counts share the same strength margin, so they form vertical lines.",
           s.barnum, ha="left", va="bottom")
    for i, tk in reversed(list(enumerate(T_KEYS))):
        t = d.tradeoff[tk]
        yb += lead
        p.text(15.0, yb, rf"({'ab'[i]}) {thousands(t['n_tied_for_strongest'])} designs tie for the strongest; their "
                         rf"$\lambda$ ranges from ${t['tied_lambda_8x8_min']:.3f}$ to ${t['tied_lambda_8x8_max']:.3f}$.",
               s.barnum, ha="left", va="bottom")
    n_valid = {d.tradeoff[tk]["n_valid"] for tk in T_KEYS}
    assert len(n_valid) == 1
    keys_row(p, 15.0, H - 3.6, [
        ("line", "cloud", rf"Valid designs (${thousands(n_valid.pop())}$)", {"lw": s.lw * 0.8}),
        ("line", "stiffest", "Stiffest", dict(ls="", marker="o", mec=st.WHITE, mew=0.6, ms=s.ms * 1.25)),
        ("line", "strongest", "Strongest", dict(ls="", marker="s", mec=st.WHITE, mew=0.6, ms=s.ms * 1.25)),
        ("line", "qi", "QI", dict(ls="", marker="D", mec=st.WHITE, mew=0.6, ms=s.ms * 1.25)),
        ("line", "pareto", "Pareto front", dict(lw=s.lw * 0.8, marker="o", ms=s.ms_small, mew=0)),
        ("line", "exact", r"Limit $1.5$", dict(ls="--", lw=s.thin))], gap=5.0)
    finish(p)
    return info


# ---------------------------------------------------------------------------------------- GA convergence
def best_so_far(h, grid_x):
    """lambda 8x8 (the GA's objective) of the run's best design so far at each number of evaluations (NaN before
    the first design meeting both constraints; held after the last change)."""
    ev = np.array(h["evals"], float)
    val = np.array([v if (v is not None and f) else np.nan for v, f in zip(h["objective_8x8"], h["feasible"])],
                   float)
    idx = np.searchsorted(ev, grid_x, side="right") - 1
    out = np.where(idx >= 0, val[np.clip(idx, 0, None)], np.nan)
    return out


def fig5(d, s, picks):
    W, H = st.PAPER_W["double"], 74.0
    p = Page("Fig5_ga_convergence", W, H, s)
    g = grid(W, H, 18.5, 6.0, 15.0, 17.0, 9.0, 0, 1, len(picks))     # right 6 mm: the last tick label 10000
    x0, x1 = 50, 10000
    gx = np.unique(np.round(np.logspace(np.log10(x0), np.log10(x1), 400)).astype(int))
    xt = (100, 300, 1000, 3000, 10000)
    info = {}
    for i, (c, tk) in enumerate(picks):
        r = d.rows[(c, tk)]
        opt = r["lambda 8x8 of the 8x8 best (-)"]
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
        p.xlabel(ax, "Distinct designs evaluated", 5.2)
        info[f"{c} {tk}"] = {"optimum_lambda_8x8": opt,
                             "ga_ls_first_point": float(curves["GA + LS"][0][0]) if len(curves["GA + LS"][0]) else None}
    ax0 = p.fig.axes[1]
    p.ylabel(ax0, r"$\lambda$ (Ritz $8 \times 8$)", 1.6)
    p.ylabel(ax0, r"of the best design so far (--)", 5.0)
    keys_row(p, 18.5, H - 3.6, [("line", "GA", "GA, median", {"ls": "-"}),
                                ("line", "GA + LS", "GA + local search, median", {"ls": "-"}),
                                ("line", "exact", "Proven optimum", {"ls": ":", "lw": s.ref_lw})])
    keys_row(p, 18.5, H - 8.4, [("line", "GA", "GA, min/max", {"ls": "--", "lw": s.thin}),
                                ("line", "GA + LS", "GA + local search, min/max", {"ls": "--", "lw": s.thin})])
    p.text(18.5, 1.0, "GA + local search history starts after its first local search.", s.barnum, ha="left",
           va="bottom")
    finish(p)
    return info


# -------------------------------------------------------------------------------------------------- table
def seq_tex(seq):
    if not seq:
        return "--"
    body = seq.strip()
    if body.startswith("[0/45/-45/90]_"):
        k = body.split("_")[1].rstrip("s")
        return rf"$[0/45/-45/90]_{{{k}\mathrm{{s}}}}$"
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
            stiff = r.get("max_stiffness sequence") is not None
            lam = r.get("objective") == "max lambda"
            rows.append({"case": c, "t (mm)": T_MM[tk], "N (-)": r.get("N (-)"), "N from": r.get("N from"),
                         "objective": "max lambda" if lam else "min c",
                         "stiffest sequence": r.get("max_stiffness sequence"),
                         "lambda 16x16 (-)": r.get("lambda 16x16 (-)") if (stiff and lam) else None,
                         "c (N/m)": r.get("c (N/m)") if (stiff and not lam) else None,
                         "strength margin (-)": r.get("strength margin (-)"), "vs QI (%)": r.get("vs QI (%)"),
                         VS_QI: r.get(VS_QI), "vs QI meaning": r.get("vs QI meaning"), "status": r.get("status"),
                         "time (s)": r.get("time max_stiffness (s)")})
    with open(os.path.join(BASE, "tables", "stiffness_16_problems.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = [r"% Fixed-thickness stiffest laminate (objective max_stiffness), 16 problems at the default ply count.",
         r"% Generated by deliverables/figures_stiffness/python/make_figures.py from",
         r"% deliverables/optimization/stiffness_step2/results.csv. Needs \usepackage{booktabs}.",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Stiffest laminate of fixed thickness for the 16 problems (IM7/8552, $0.5 \times 0.5$~m simply "
         r"supported plate, safety factor $1.5$ on strength and buckling, shear of both signs). $\lambda$: Ritz sine "
         r"series $16 \times 16$ terms; optimality proven in the $8 \times 8$ Ritz model; when the load cannot buckle "
         r"the plate the in-plane compliance $c$ is minimised. vs QI: increase of $\lambda$ or decrease of $c$ "
         r"against $[0/45/-45/90]_{k\mathrm{s}}$ with the same $N$, computed from unrounded values and rounded once "
         r"to one decimal.}",
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
        vs = f"{r[VS_QI]:+.1f}" if r[VS_QI] is not None else "no QI with the same $N$"
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
