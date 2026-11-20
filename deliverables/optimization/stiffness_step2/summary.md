# Fixed thickness, stiffest laminate (objective max_stiffness): step 2

IM7/8552, simply supported 0.5 m x 0.5 m plate, safety factor 1.5 on the strength margin min(RF Max Stress, RF Tsai-Wu) and on the buckling factor lambda, shear checked for +Nxy and -Nxy, symmetric balanced laminates of 0/+45/-45/90 plies with the design rules of the project. Objective: largest lambda; when the load cannot buckle the plate (C1, C2, C8), smallest in-plane compliance c = N^T A^-1 N [N/m].

- **Search model and proofs**: Ritz double sine series 8 x 8 terms. 'proven' means proven in the 8x8 Ritz model (the compliance problems involve no Ritz model: count-level search, exact).
- **Reported lambda** (tables, figures except figure 4, analyze_design): Ritz double sine series 16 x 16 terms; the polynomial Ritz solution p = 16 and the 8 x 8 value are given next to it. The k best designs of the 8 x 8 model (k = 20 first) are re-evaluated with 16 x 16 terms; k is doubled while (lambda8_1 - lambda8_k) / lambda8_1 < the largest relative 8x8-16x16 difference in the list (up to 500), and the design reported is the best of the list by the 16 x 16 value. The 16 x 16 basis contains the 8 x 8 one, so lambda16 <= lambda8 for every design and no design outside the list can be better by the 16 x 16 value.
- **Ties of the compliance objective**: (1) smallest c, (2) largest strength margin, (3) largest bending stiffness in the main load direction (largest of |Nx|, |Ny|, |Nxy|: D11, D22 or D66), (4) lexicographically smallest sequence (0 < +45 < -45 < 90 from the outer ply). Within 1e-9 (relative) two values are equal.
- **Default N**: smallest [0/45/-45/90]_ks (N <= 64) meeting both constraints ('QI_ks'); if none, the fewest plies of the objective min_plies ('min_plies fallback'); if neither exists the problem is infeasible.

## Key points

- 16 problems; 15 have a default N and were solved, 1 infeasible; 14 proven optima, 1 INCOMPLETE (not proven).
- **C8, t = 0.125 mm: INFEASIBLE.** No [0/45/-45/90]_ks and no design of 8-64 plies meets both constraints (count-level scan of the largest strength margin, exact). Old fewest-plies study: results/cases_summary.csv 'infeasible' (best margin 1.0427 < 1.5 with 64 plies), results/shear_both_signs.csv 'infeasible': the new result agrees with the old one. Details below.
- **C5, t = 0.125 mm, N = 56: INCOMPLETE (not proven, gap 0.455 % in the 8x8 model).** The exact search reached its time limit of 3600 s (total 3617 s). Best design found: [45_3/-45/45/-45_4/45/-45_4/45_2/-45/45/-45/45_3/0/90_2/0/90/0]s, lambda 8x8 = 2.905409 (16x16 = 2.903949); upper bound of the 8x8 model 2.918628, gap 0.455 % in the 8x8 model. Not proven: the optimum lies between these two values of lambda 8x8. GA and GA + local search (40 runs) compared with this best known design: largest lambda 8x8 found 2.905409, no run found a better design.
- **C6, t = 0.18288 mm, N = 24: the best design by the 16 x 16 value is not the 8 x 8 optimum.** Best by the 8 x 8 model: [90_4/45/90/-45/90/-45/45/0_2]s (lambda 8x8 = 13.516848, 16x16 = 13.439954, strength margin 1.6166). Best by the 16 x 16 value (reported): [90_4/45/-45/90_2/-45/45/0_2]s (lambda 8x8 = 13.515667, 16x16 = 13.440752, strength margin 1.6166). The enumeration of all 1,864,416 valid designs (figure 4) gives the same largest lambda 8x8: yes.

## Default ply count

| case | t (mm) | objective | N | N from | detail | old fewest plies (cases_summary.csv) |
|---|---|---|---|---|---|---|
| C1 | 0.125 | c | 48 | QI_ks | [0/45/-45/90]_6s | 24 (optimal, proven) |
| C1 | 0.18288 | c | 32 | QI_ks | [0/45/-45/90]_4s | 16 (optimal, proven) |
| C2 | 0.125 | c | 32 | QI_ks | [0/45/-45/90]_4s | 20 (optimal, proven) |
| C2 | 0.18288 | c | 24 | QI_ks | [0/45/-45/90]_3s | 14 (optimal, proven) |
| C3 | 0.125 | lambda | 24 | QI_ks | [0/45/-45/90]_3s | 20 (optimal, proven) |
| C3 | 0.18288 | lambda | 16 | QI_ks | [0/45/-45/90]_2s | 14 (optimal, proven) |
| C4 | 0.125 | lambda | 24 | QI_ks | [0/45/-45/90]_3s | 22 (optimal, proven) |
| C4 | 0.18288 | lambda | 24 | QI_ks | [0/45/-45/90]_3s | 16 (optimal, proven) |
| C5 | 0.125 | lambda | 56 | QI_ks | [0/45/-45/90]_7s | 42 (unknown (time limit)) |
| C5 | 0.18288 | lambda | 40 | QI_ks | [0/45/-45/90]_5s | 28 (optimal, proven) |
| C6 | 0.125 | lambda | 32 | QI_ks | [0/45/-45/90]_4s | 24 (optimal, proven) |
| C6 | 0.18288 | lambda | 24 | QI_ks | [0/45/-45/90]_3s | 16 (optimal, proven) |
| C7 | 0.125 | lambda | 24 | QI_ks | [0/45/-45/90]_3s | 20 (optimal, proven) |
| C7 | 0.18288 | lambda | 16 | QI_ks | [0/45/-45/90]_2s | 14 (optimal, proven) |
| C8 | 0.125 | c | - | - | no design of 8-64 plies meets the constraints (source: cases_summary.csv) | 64 (infeasible) |
| C8 | 0.18288 | c | 64 | min_plies fallback | min_plies 64 plies (proven; source: cases_summary.csv) | 64 (optimal, proven) |

cases_summary.csv is the fewest-plies study with shear of one sign. For C8 (no buckling, balanced laminates) the strength margin is the same for +Nxy and -Nxy, so its ply count holds for both signs; it is re-checked here by the count-level scan of the largest strength margin over every N (exact) and by results/shear_both_signs.csv.

- **C8, t = 0.125 mm: infeasible.** No [0/45/-45/90]_ks up to 64 plies meets the constraints (QI strength margins: 8: rs=0.1201 lam8=inf; 16: rs=0.2403 lam8=inf; 24: rs=0.3604 lam8=inf; 32: rs=0.4806 lam8=inf; 40: rs=0.6007 lam8=inf; 48: rs=0.7209 lam8=inf; 56: rs=0.841 lam8=inf; 64: rs=0.9612 lam8=inf). Largest strength margin of any valid design (count-level scan, exact): 1.0427 at 64 plies < 1.5. Old fewest-plies study (cases_summary.csv): status 'infeasible', feasible False, best design [90/-45/45/0/90/-45/45/0/90/-45/45/0/90/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/-45/45]s (64 plies) with margin 1.0427; re-evaluated in this model (both shear signs): strength margin 1.0427. shear_both_signs.csv: infeasible. The old and the new results agree (no feasible design of at most 64 plies).

## Compliance tie-break: exact search vs enumeration of every valid design

c and the strength margin were compared design by design with the first design of the same ply counts: largest relative difference over every valid design below (0 = identical numbers).

| case | t (mm) | N | valid designs | designs tied on c and margin | status | direction | exact design | enumeration design | same | D (N.m) exact / enumeration | spread of c / margin within counts | match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.125 | 16 | 10,056 | - (infeasible) | infeasible | D11 | [0_4/45/0/-45/90]s | [0_4/45/0/-45/90]s | yes | 101.9413506 / 101.9413506 | 0.0e+00 / 0.0e+00 | yes |
| C1 | 0.18288 | 16 | 10,056 | 258 | optimal | D11 | [0_4/45/0/-45/90]s | [0_4/45/0/-45/90]s | yes | 319.2412359 / 319.2412359 | 0.0e+00 / 0.0e+00 | yes |
| C2 | 0.125 | 16 | 10,056 | - (infeasible) | infeasible | D11 | [0_4/45/-45/90_2]s | [0_4/45/-45/90_2]s | yes | 98.75836225 / 98.75836225 | 0.0e+00 / 0.0e+00 | yes |
| C2 | 0.18288 | 16 | 10,056 | 258 | optimal | D11 | [0_4/45/0/-45/90]s | [0_4/45/0/-45/90]s | yes | 319.2412359 / 319.2412359 | 0.0e+00 / 0.0e+00 | yes |
| C8 | 0.125 | 16 | 10,056 | - (infeasible) | infeasible | D11 | [0_3/45_2/-45_2/90]s | [0_3/45_2/-45_2/90]s | yes | 90.07845613 / 90.07845613 | 0.0e+00 / 0.0e+00 | yes |
| C8 | 0.18288 | 16 | 10,056 | - (infeasible) | infeasible | D11 | [0_3/45_2/-45_2/90]s | [0_3/45_2/-45_2/90]s | yes | 282.0911974 / 282.0911974 | 0.0e+00 / 0.0e+00 | yes |
| C1 | 0.125 | 24 | 1,864,416 | 2796 | optimal | D11 | [0_4/45/0_4/-45/90_2]s | [0_4/45/0_4/-45/90_2]s | yes | 337.2674059 / 337.2674059 | 0.0e+00 / 0.0e+00 | yes |
| C1 | 0.18288 | 24 | 1,864,416 | 2796 | optimal | D11 | [0_4/45/0_4/-45/90_2]s | [0_4/45/0_4/-45/90_2]s | yes | 1056.19224 / 1056.19224 | 0.0e+00 / 0.0e+00 | yes |
| C2 | 0.125 | 24 | 1,864,416 | 11184 | optimal | D11 | [0_4/45/0_3/90/-45/90_2]s | [0_4/45/0_3/90/-45/90_2]s | yes | 329.8486194 / 329.8486194 | 0.0e+00 / 0.0e+00 | yes |
| C2 | 0.18288 | 24 | 1,864,416 | 11184 | optimal | D11 | [0_4/45/0_3/90/-45/90_2]s | [0_4/45/0_3/90/-45/90_2]s | yes | 1032.959444 / 1032.959444 | 0.0e+00 / 0.0e+00 | yes |
| C8 | 0.125 | 24 | 1,864,416 | - (infeasible) | infeasible | D11 | [0_4/45_3/-45_3/90_2]s | [0_4/45_3/-45_3/90_2]s | yes | 290.4089726 / 290.4089726 | 0.0e+00 / 0.0e+00 | yes |
| C8 | 0.18288 | 24 | 1,864,416 | - (infeasible) | infeasible | D11 | [0_4/45_3/-45_3/90_2]s | [0_4/45_3/-45_3/90_2]s | yes | 909.4495881 / 909.4495881 | 0.0e+00 / 0.0e+00 | yes |

## The 16 problems (max_stiffness, exact search, default N)

lambda: 16 x 16 value (reported); vs QI: change against [0/45/-45/90]_ks with the same N (lambda: increase of the 16 x 16 value, c: decrease). Strength margin = min(RF Max Stress, RF Tsai-Wu).

vs QI is computed once from the unrounded values (results.csv column 'vs QI (%)') and rounded once to one decimal (column 'vs QI 1 decimal (%)'); this summary, the table and the figures of deliverables/figures_stiffness/ all print that column.

| case | t (mm) | N | N from | stiffest sequence | lambda 16x16 | lambda poly p=16 | lambda 8x8 | c (N/m) | strength margin | vs QI (%) | status | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.125 | 48 | QI_ks | [0_4/45/0_4/45/0_4/-45/0_4/90/0/-45/90_2]s | inf | inf | inf | 3062.36 | 3.434 | +50.3 (c) | proven optimum (proven, exact count-level search) | 2.2 |
| C1 | 0.18288 | 32 | QI_ks | [0_4/45/0_4/-45/0_4/90_2]s | inf | inf | inf | 3007.37 | 3.511 | +52.4 (c) | proven optimum (proven, exact count-level search) | 0.5 |
| C2 | 0.125 | 32 | QI_ks | [0_4/45/0_4/90/0_2/90/-45/90_2]s | inf | inf | inf | 3165.69 | 2.556 | +17.6 (c) | proven optimum (proven, exact count-level search) | 0.5 |
| C2 | 0.18288 | 24 | QI_ks | [0_4/45/0_3/90/-45/90_2]s | inf | inf | inf | 2892.75 | 2.771 | +17.4 (c) | proven optimum (proven, exact count-level search) | 0.3 |
| C3 | 0.125 | 24 | QI_ks | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 3.0073 | 3.0073 | 3.0073 | 0.678171 | 138.7 | +26.2 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 1.4 |
| C3 | 0.18288 | 16 | QI_ks | [45/-45_2/45/-45/45/0/90]s | 2.8163 | 2.8163 | 2.8163 | 0.806444 | 108.6 | +28.7 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 0.4 |
| C4 | 0.125 | 24 | QI_ks | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 2.0048 | 2.0048 | 2.0048 | 0.544699 | 201.4 | +26.2 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 1.1 |
| C4 | 0.18288 | 24 | QI_ks | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 6.2784 | 6.2784 | 6.2784 | 0.372306 | 294.7 | +26.2 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 1.1 |
| C5 | 0.125 | 56 | QI_ks | [45_3/-45/45/-45_4/45/-45_4/45_2/-45/45/-45/45_3/0/90_2/0/90/0]s | 2.9039 | 2.9039 | 2.9054 | 381.811 | 9.312 | +25.5 (lambda) | INCOMPLETE (not proven, gap 0.455 % in the 8x8 model) | 3617.3 |
| C5 | 0.18288 | 40 | QI_ks | [45_2/-45_2/45/-45_2/45/-45/45/-45_3/45_3/0/90_2/0]s | 3.317 | 3.3169 | 3.3186 | 359.73 | 9.884 | +26.9 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 77.6 |
| C6 | 0.125 | 32 | QI_ks | [90_4/45/90_3/-45_3/45/0/45/0_2]s | 10.039 | 10.034 | 10.095 | 4422.44 | 1.603 | +88.8 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 17.8 |
| C6 | 0.18288 | 24 | QI_ks | [90_4/45/-45/90_2/-45/45/0_2]s | 13.441 | 13.434 | 13.516 | 4395.37 | 1.617 | +102.9 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 4.0 |
| C7 | 0.125 | 24 | QI_ks | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 2.8798 | 2.8798 | 2.8798 | 0.962754 | 126.8 | +28.3 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 2.5 |
| C7 | 0.18288 | 16 | QI_ks | [45/-45_2/45/-45/45/0/90]s | 2.69 | 2.69 | 2.6902 | 1.0705 | 103.2 | +31.5 (lambda) | proven optimum (proven in the 8x8 Ritz model) | 0.8 |
| C8 | 0.125 | - | - | - | - | - | - | - | - | - | **INFEASIBLE: no QI laminate up to 64 plies and no min_plies design up to 64 plies** | - |
| C8 | 0.18288 | 64 | min_plies fallback | [0_4/45/0_4/45/0_2/45_4/-45/45_2/-45_3/45/-45_4/90_2/-45/90_2]s | inf | inf | inf | 21808.9 | 1.512 | no QI with the same N | proven optimum (proven, exact count-level search) | 13.3 |

QI with the same N and bending stiffness:

| case | t (mm) | N | QI | QI lambda 16x16 | QI c (N/m) | QI strength margin | QI feasible | D11 / D22 / D66 (N.m) stiffest | main direction |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.125 | 48 | [0/45/-45/90]_6s | inf | 6162.66 | 1.692 | yes | 2614 / 283.7 / 183.5 | D11 |
| C1 | 0.18288 | 32 | [0/45/-45/90]_4s | inf | 6318.35 | 1.65 | yes | 2474 / 241.2 / 157.2 | D11 |
| C2 | 0.125 | 32 | [0/45/-45/90]_4s | inf | 3840.79 | 1.922 | yes | 773.2 / 104.3 / 44.99 | D11 |
| C2 | 0.18288 | 24 | [0/45/-45/90]_3s | inf | 3500.28 | 2.109 | yes | 1033 / 122.5 / 61.33 | D11 |
| C3 | 0.125 | 24 | [0/45/-45/90]_3s | 2.382 | 0.547792 | 194.9 | yes | 118.4 / 108.8 / 90.48 | D11 |
| C3 | 0.18288 | 16 | [0/45/-45/90]_2s | 2.1888 | 0.561631 | 190.1 | yes | 105.6 / 101.9 / 85.6 | D11 |
| C4 | 0.125 | 24 | [0/45/-45/90]_3s | 1.588 | 0.512105 | 229.2 | yes | 118.4 / 108.8 / 90.48 | D11 |
| C4 | 0.18288 | 24 | [0/45/-45/90]_3s | 4.9731 | 0.350028 | 335.4 | yes | 370.8 / 340.6 / 283.3 | D11 |
| C5 | 0.125 | 56 | [0/45/-45/90]_7s | 2.3138 | 555.758 | 6.397 | yes | 1415 / 1414 / 1178 | D66 |
| C5 | 0.18288 | 40 | [0/45/-45/90]_5s | 2.6128 | 531.812 | 6.685 | yes | 1616 / 1608 / 1347 | D66 |
| C6 | 0.125 | 32 | [0/45/-45/90]_4s | 5.316 | 3601.98 | 1.938 | yes | 102.2 / 732.2 / 66.51 | D11 |
| C6 | 0.18288 | 24 | [0/45/-45/90]_3s | 6.6232 | 3282.64 | 2.127 | yes | 126.1 / 980 / 85.99 | D11 |
| C7 | 0.125 | 24 | [0/45/-45/90]_3s | 2.2454 | 0.908006 | 155.3 | yes | 118.4 / 108.8 / 90.48 | D11 |
| C7 | 0.18288 | 16 | [0/45/-45/90]_2s | 2.0459 | 0.930944 | 151.4 | yes | 105.6 / 101.9 / 85.6 | D11 |
| C8 | 0.18288 | 64 | [0/45/-45/90]_8s | inf | 24694.8 | 1.406 | no | 1.629e+04 / 3181 / 2384 | D11 |

## Top-k rule (8x8 -> 16x16)

| case | t (mm) | N | k final | gap of the list | largest 8x8-16x16 difference | rule met | best 8x8 = best 16x16 | 8x8 rank of the reported design |
|---|---|---|---|---|---|---|---|---|
| C5 | 0.125 | 56 | - | - | - | no list: exact search INCOMPLETE, the best known design is reported | - | - |
| C5 | 0.18288 | 40 | 320 | 1.027e-03 | 5.100e-04 | yes | yes | 1 |
| C3 | 0.125 | 24 | 20 | 6.235e-05 | 4.957e-06 | yes | yes | 1 |
| C3 | 0.18288 | 16 | 20 | 9.848e-03 | 8.122e-04 | yes | yes | 1 |
| C4 | 0.125 | 24 | 20 | 6.229e-05 | 4.959e-06 | yes | yes | 1 |
| C4 | 0.18288 | 24 | 20 | 6.229e-05 | 4.959e-06 | yes | yes | 1 |
| C6 | 0.125 | 32 | 160 | 7.455e-03 | 5.890e-03 | yes | yes | 1 |
| C6 | 0.18288 | 24 | 40 | 1.008e-02 | 5.925e-03 | yes | no | 7 |
| C7 | 0.125 | 24 | 20 | 3.043e-03 | 4.865e-05 | yes | yes | 1 |
| C7 | 0.18288 | 16 | 20 | 2.324e-02 | 4.721e-04 | yes | yes | 1 |

- C6 t = 0.18288 mm, N = 24: the best design by the 16 x 16 value is not the 8 x 8 optimum; the 16 x 16 one is reported.

  | design | sequence | lambda 8x8 | lambda 16x16 | lambda poly p=16 | strength margin | 8x8 rank | 16x16 rank |
  |---|---|---|---|---|---|---|---|
  | reported (best by 16x16) | [90_4/45/-45/90_2/-45/45/0_2]s | 13.515667 | 13.440752 | 13.433657 | 1.6166 | 7 | 1 |
  | 8 x 8 optimum | [90_4/45/90/-45/90/-45/45/0_2]s | 13.516848 | 13.439954 | 13.432229 | 1.6166 | 1 | 5 |


## Stiffest (max_stiffness) vs strongest (max_margin) at the same N

max_margin maximises margin = min(strength margin, lambda 8x8) (objective of the fewest-plies study, unchanged); its lambda is also given with 16 x 16 terms. Once lambda 8x8 reaches the strength margin of the ply counts a larger lambda does not count, so many designs can tie; the exact search keeps the first one it finds (see figure 4 below for the number of ties). With shear of both signs a design and its mirror (+45 and -45 swapped) have the same values; the two objectives break such ties differently, so 'same up to swap' is given.

| case | t (mm) | N | stiffest | lambda 16x16 | c (N/m) | strength margin | strongest | lambda 16x16 | c (N/m) | strength margin | strongest status | same (up to +45/-45 swap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.125 | 48 | [0_4/45/0_4/45/0_4/-45/0_4/90/0/-45/90_2]s | inf | 3062.36 | 3.434 | [90/-45/0/45/0_2/90/0_4/-45/0_4/45/0_4/90/0_2]s | inf | 3062.36 | 3.434 | optimal | no |
| C1 | 0.18288 | 32 | [0_4/45/0_4/-45/0_4/90_2]s | inf | 3007.37 | 3.511 | [0/90/0/-45/0_4/45/0_4/90/0_2]s | inf | 3007.37 | 3.511 | optimal | no |
| C2 | 0.125 | 32 | [0_4/45/0_4/90/0_2/90/-45/90_2]s | inf | 3165.69 | 2.556 | [90/-45/0/45/0/90/0_2/90/0_4/90/0_2]s | inf | 3165.69 | 2.556 | optimal | no |
| C2 | 0.18288 | 24 | [0_4/45/0_3/90/-45/90_2]s | inf | 2892.75 | 2.771 | [90/-45/0/45/0/90/0_3/90/0_2]s | inf | 2892.75 | 2.771 | optimal | no |
| C3 | 0.125 | 24 | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 3.0073 | 0.678171 | 138.7 | [-45/45_2/-45/45/-45_2/45/0_2/90_2]s | 3.0073 | 0.678171 | 138.7 | optimal | yes |
| C3 | 0.18288 | 16 | [45/-45_2/45/-45/45/0/90]s | 2.8163 | 0.806444 | 108.6 | [-45/45_2/-45/45/-45/0/90]s | 2.8163 | 0.806444 | 108.6 | optimal | yes |
| C4 | 0.125 | 24 | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 2.0048 | 0.544699 | 201.4 | [-45/45_2/-45/45/-45_2/45/0_2/90_2]s | 2.0048 | 0.544699 | 201.4 | optimal | yes |
| C4 | 0.18288 | 24 | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 6.2784 | 0.372306 | 294.7 | [-45/45_2/-45/45/-45_2/45/0_2/90_2]s | 6.2784 | 0.372306 | 294.7 | optimal | yes |
| C5 | 0.125 | 56 | [45_3/-45/45/-45_4/45/-45_4/45_2/-45/45/-45/45_3/0/90_2/0/90/0]s | 2.9039 | 381.811 | 9.312 | [90/-45/45/0/90/-45/45/0/90/-45/45/0/-45/45/-45/45/-45/45/-45/45/-45/45/-45/45/-45/45/-45/45]s | 2.4035 | 381.811 | 9.312 | not proven (arrangement cap of exact_full (a ply-count combination has > 3 000 000 arrangements)) | no |
| C5 | 0.18288 | 40 | [45_2/-45_2/45/-45_2/45/-45/45/-45_3/45_3/0/90_2/0]s | 3.317 | 359.73 | 9.884 | [90/-45/45/0/90/-45/45/0/-45/45/-45/45/-45/45/-45/45/-45/45/-45/45]s | 2.7253 | 359.73 | 9.884 | not proven (arrangement cap of exact_full (a ply-count combination has > 3 000 000 arrangements)) | no |
| C6 | 0.125 | 32 | [90_4/45/90_3/-45_3/45/0/45/0_2]s | 10.039 | 4422.44 | 1.603 | [90/-45/45/0/90/-45/45/0/-45/45/0/-45/0/45/0_2]s | 6.3526 | 3039.08 | 2.321 | optimal | no |
| C6 | 0.18288 | 24 | [90_4/45/-45/90_2/-45/45/0_2]s | 13.441 | 4395.37 | 1.617 | [90/-45/45/0/90/-45/45/0/-45/0/45/0]s | 8.8623 | 2908.49 | 2.424 | optimal | no |
| C7 | 0.125 | 24 | [45/-45_2/45/-45/45_2/-45/0_2/90_2]s | 2.8798 | 0.962754 | 126.8 | [-45/45_2/-45/45/-45_2/45/0_2/90_2]s | 2.8798 | 0.962754 | 126.8 | optimal | yes |
| C7 | 0.18288 | 16 | [45/-45_2/45/-45/45/0/90]s | 2.69 | 1.0705 | 103.2 | [-45/45_2/-45/45/-45/0/90]s | 2.69 | 1.0705 | 103.2 | optimal | yes |
| C8 | 0.18288 | 64 | [0_4/45/0_4/45/0_2/45_4/-45/45_2/-45_3/45/-45_4/90_2/-45/90_2]s | inf | 21808.9 | 1.512 | [90/-45/45/0/90/-45/45/0/90/-45/45/0/90/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/-45/45]s | inf | 22550.6 | 1.526 | optimal | no |

## Old fewest-plies design of each case (reference)

From results/cases_summary.csv (shear of one sign), re-evaluated here with both shear signs at its own N.

| case | t (mm) | N old | sequence | lambda 16x16 | c (N/m) | strength margin | fewest plies with both signs (shear_both_signs.csv) |
|---|---|---|---|---|---|---|---|
| C1 | 0.125 | 24 | [90/0/-45/0/45/0_4/90/0_2]s | inf | 6443.43 | 1.64 | - |
| C1 | 0.18288 | 16 | [90/0/-45/0_2/45/0_2]s | inf | 6897.92 | 1.512 | - |
| C2 | 0.125 | 20 | [90/-45/0/45/0_3/90/0_2]s | inf | 5065.99 | 1.573 | - |
| C2 | 0.18288 | 14 | [90/0/-45/0/45/0_2]s | inf | 4958.06 | 1.546 | - |
| C3 | 0.125 | 20 | [45/-45/45/-45_3/45_2/0/90]s | 1.7623 | 1.05375 | 79.45 | - |
| C3 | 0.18288 | 14 | [45/-45_2/45/0_2/90]s | 1.8346 | 0.598045 | 157.7 | - |
| C4 | 0.125 | 22 | [-45/45_2/-45/45/-45/0/90/0/90_2]s | 1.5058 | 0.624423 | 178.7 | - |
| C4 | 0.18288 | 16 | [-45/45_2/-45/45/-45/0/90]s | 1.8775 | 0.586245 | 178.4 | - |
| C5 | 0.125 | 42 | [-45_4/90/-45_3/0/90/0_2/45/0/45_4/90/45_2]s | 0.55313 | 585.429 | 6.073 | 46 |
| C5 | 0.18288 | 28 | [-45_4/0/-45/90_2/45_3/0/45_2]s | 0.51091 | 566.25 | 6.279 | 32 |
| C6 | 0.125 | 24 | [90/-45/45/0/90/-45/45/0/-45/0/45/0]s | 2.8299 | 4255.23 | 1.657 | 24 |
| C6 | 0.18288 | 16 | [90/-45/45/0/-45/0/45/0]s | 2.5646 | 4154.47 | 1.698 | 16 |
| C7 | 0.125 | 20 | [-45_2/45/-45/45/-45/45_2/0/90]s | 1.4867 | 1.34614 | 76.76 | 20 |
| C7 | 0.18288 | 14 | [-45_2/45_2/0_2/90]s | 1.5505 | 0.976961 | 131.3 | 14 |
| C8 | 0.125 | 64 | [90/-45/45/0/90/-45/45/0/90/-45/45/0/90/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/-45/45]s | - | - | - | 64 |
| C8 | 0.18288 | 64 | [90/-45/45/0/90/-45/45/0/90/-45/45/0/90/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/0/-45/45/-45/45]s | inf | 22550.6 | 1.526 | 64 |

## GA and GA + local search (20 seeds, budget 10 000 designs, 8x8 model)

Hit = the run's best design reaches the objective value of the exact 8 x 8 optimum (lambda8 or c within a relative 1e-9) with the same feasibility and tier; 'same values' also requires the same strength margin and (compliance) the same bending stiffness.

| case | t (mm) | N | method | hit (full budget) | hit <= 1 000 | hit <= 2 000 | same values | same sequence | median evaluations to hit | target |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.125 | 48 | ga | 20/20 | 20/20 | 20/20 | 7/20 | 2/20 | 320 | optimum of the 8x8 model |
| C1 | 0.125 | 48 | ga_ls | 20/20 | 0/20 | 0/20 | 16/20 | 16/20 | 2623 | optimum of the 8x8 model |
| C1 | 0.18288 | 32 | ga | 20/20 | 20/20 | 20/20 | 20/20 | 18/20 | 264.5 | optimum of the 8x8 model |
| C1 | 0.18288 | 32 | ga_ls | 20/20 | 19/20 | 20/20 | 20/20 | 20/20 | 829.5 | optimum of the 8x8 model |
| C2 | 0.125 | 32 | ga | 20/20 | 20/20 | 20/20 | 9/20 | 9/20 | 196 | optimum of the 8x8 model |
| C2 | 0.125 | 32 | ga_ls | 20/20 | 13/20 | 20/20 | 19/20 | 19/20 | 938 | optimum of the 8x8 model |
| C2 | 0.18288 | 24 | ga | 20/20 | 20/20 | 20/20 | 20/20 | 18/20 | 86 | optimum of the 8x8 model |
| C2 | 0.18288 | 24 | ga_ls | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 424 | optimum of the 8x8 model |
| C3 | 0.125 | 24 | ga | 16/20 | 2/20 | 5/20 | 16/20 | 4/20 | 3316 | optimum of the 8x8 model |
| C3 | 0.125 | 24 | ga_ls | 19/20 | 12/20 | 14/20 | 19/20 | 7/20 | 435 | optimum of the 8x8 model |
| C3 | 0.18288 | 16 | ga | 20/20 | 16/20 | 20/20 | 20/20 | 20/20 | 668 | optimum of the 8x8 model |
| C3 | 0.18288 | 16 | ga_ls | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 125.5 | optimum of the 8x8 model |
| C4 | 0.125 | 24 | ga | 18/20 | 2/20 | 5/20 | 18/20 | 2/20 | 4025 | optimum of the 8x8 model |
| C4 | 0.125 | 24 | ga_ls | 20/20 | 12/20 | 14/20 | 20/20 | 11/20 | 459 | optimum of the 8x8 model |
| C4 | 0.18288 | 24 | ga | 16/20 | 2/20 | 5/20 | 16/20 | 3/20 | 3652 | optimum of the 8x8 model |
| C4 | 0.18288 | 24 | ga_ls | 20/20 | 13/20 | 14/20 | 20/20 | 11/20 | 433.5 | optimum of the 8x8 model |
| C5 | 0.125 | 56 | ga | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | - | best known (not proven) |
| C5 | 0.125 | 56 | ga_ls | 5/20 | 0/20 | 0/20 | 5/20 | 0/20 | 8376 | best known (not proven) |
| C5 | 0.18288 | 40 | ga | 2/20 | 0/20 | 0/20 | 2/20 | 0/20 | 5002 | optimum of the 8x8 model |
| C5 | 0.18288 | 40 | ga_ls | 3/20 | 0/20 | 0/20 | 3/20 | 0/20 | 7782 | optimum of the 8x8 model |
| C6 | 0.125 | 32 | ga | 4/20 | 0/20 | 0/20 | 4/20 | 1/20 | 7121 | optimum of the 8x8 model |
| C6 | 0.125 | 32 | ga_ls | 18/20 | 7/20 | 14/20 | 18/20 | 10/20 | 1072 | optimum of the 8x8 model |
| C6 | 0.18288 | 24 | ga | 17/20 | 2/20 | 2/20 | 17/20 | 6/20 | 2994 | optimum of the 8x8 model |
| C6 | 0.18288 | 24 | ga_ls | 20/20 | 18/20 | 20/20 | 20/20 | 15/20 | 483 | optimum of the 8x8 model |
| C7 | 0.125 | 24 | ga | 11/20 | 0/20 | 0/20 | 11/20 | 6/20 | 6175 | optimum of the 8x8 model |
| C7 | 0.125 | 24 | ga_ls | 17/20 | 8/20 | 9/20 | 17/20 | 10/20 | 1723 | optimum of the 8x8 model |
| C7 | 0.18288 | 16 | ga | 20/20 | 16/20 | 20/20 | 20/20 | 20/20 | 515.5 | optimum of the 8x8 model |
| C7 | 0.18288 | 16 | ga_ls | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 123.5 | optimum of the 8x8 model |
| C8 | 0.18288 | 64 | ga | 20/20 | 20/20 | 20/20 | 0/20 | 0/20 | 58 | optimum of the 8x8 model |
| C8 | 0.18288 | 64 | ga_ls | 20/20 | 0/20 | 0/20 | 17/20 | 13/20 | 7600 | optimum of the 8x8 model |

## Stiffness-strength trade-off, C6 with 24 plies (figure 4)

'Strongest' is the max_margin design (objective and code unchanged). max_margin maximises margin = min(strength margin, lambda 8x8). When lambda 8x8 is at least the strength margin of the ply counts, the margin equals that strength margin and a larger lambda adds nothing, so every arrangement of the best ply counts with lambda 8x8 >= its strength margin ties. The exact search keeps the first design it finds with the largest margin (no further tie-break). Figure 4 marks that design: the max_margin sequence of results.csv (the one of figure 3) when N = 24 is the default N, else the max_margin result at N = 24 (tradeoff.json). Tie counts and lambda ranges are counted in the enumeration files (float32 values; exact equality and a relative 1e-6 give the same counts).

- t = 0.125 mm (NOT the default N of C6 for this thickness (default 32)): 1,864,416 valid designs, 418,000 meet both constraints; stiffest lambda8 = 3.59313 (strength margin 1.517); QI lambda8 = 2.128, strength margin 1.454; enumeration = exact search: yes, = max_margin (margin value): yes; CPU 634 s. Strongest (max_margin, tradeoff.json (max_margin at N = 24, the same objective and tie-break)): [90/-45/45/0/90/-45/45/0/-45/0/45/0]s, strength margin 1.6565, lambda8 2.8472 (16x16 2.8299), margin 1.656522. 208,902 designs tie for the strongest; their lambda8 ranges from 1.6565 to 3.3878. Pareto points (strength margin, lambda8): (1.6565, 3.3878), (1.5172, 3.5931), (1.4537, 3.8028), (1.4231, 4.0059), (1.3031, 4.2278), (1.1049, 4.3163), (0.8515, 4.3604).
- t = 0.18288 mm (default N of C6 for this thickness): 1,864,416 valid designs, 1,826,904 meet both constraints; stiffest lambda8 = 13.5168 (strength margin 1.617); QI lambda8 = 6.663, strength margin 2.127; enumeration = exact search: yes, = max_margin (margin value): yes; CPU 639 s. Strongest (max_margin, results.csv (max_margin at the default N)): [90/-45/45/0/90/-45/45/0/-45/0/45/0]s, strength margin 2.4236, lambda8 8.9164 (16x16 8.8623), margin 2.423559. 269,640 designs tie for the strongest; their lambda8 ranges from 2.7493 to 10.6091. Pareto points (strength margin, lambda8): (2.4236, 10.6091), (2.2198, 11.2523), (2.1268, 11.9088), (2.0821, 12.5451), (1.9064, 13.2400), (1.6166, 13.5168), (1.2457, 13.6552).

## Checks

- MATLAB clt_engine.m (strength recomputed, compared with Python) of the 30 designs (max_stiffness and max_margin): 30/30 pass.
- lambda: 16 x 16 sine series vs polynomial Ritz p = 16, largest relative difference over the buckling problems: 5.28e-04.
- analyze_design gives the same lambda as the results file for every design (column 'lambda analyze_design (-)' of results.csv equals 'lambda 16x16 (-)'): yes.
- Demo regression (fewest plies, default objective): "Design the lightest IM7/8552 laminate for a shear load Nxy = 40 kN/m." -> [-45/45_2/-45/45/-45_2/45/90/0_2/90]s, 24 plies, margin 1.6988 -> PASS.

## Changes from step 1

- Compliance (tension-only) problems: ties on c and the strength margin are now broken by the bending stiffness of the main load direction before the sequence, so their stacking sequences can change (c and the strength margin cannot: they depend on the ply counts only). Same (case, t, N) as step 1:
  - C1 t = 0.125 mm, N = 48: step 1 [0_4/45/0_4/45/0_4/-45/0_4/-45/90/0/90_2]s -> step 2 [0_4/45/0_4/45/0_4/-45/0_4/90/0/-45/90_2]s
  - C1 t = 0.18288 mm, N = 32: step 1 [0_4/45/0_4/-45/0_4/90_2]s -> step 2 [0_4/45/0_4/-45/0_4/90_2]s (unchanged)
  - C1 t = 0.125 mm, N = 16 (verification): step 1 [0_4/45/0/-45/90]s -> step 2 [0_4/45/0/-45/90]s (unchanged)
  - C1 t = 0.18288 mm, N = 16 (verification): step 1 [0_4/45/0/-45/90]s -> step 2 [0_4/45/0/-45/90]s (unchanged)
  - C2 t = 0.125 mm, N = 16 (verification): step 1 [0_4/45/-45/90_2]s -> step 2 [0_4/45/-45/90_2]s (unchanged)
  - C2 t = 0.18288 mm, N = 16 (verification): step 1 [0_4/45/0/-45/90]s -> step 2 [0_4/45/0/-45/90]s (unchanged)
  - C8 t = 0.125 mm, N = 16 (verification): step 1 [0_3/45_2/-45_2/90]s -> step 2 [0_3/45_2/-45_2/90]s (unchanged)
  - C8 t = 0.18288 mm, N = 16 (verification): step 1 [0_3/45_2/-45_2/90]s -> step 2 [0_3/45_2/-45_2/90]s (unchanged)
  - C1 t = 0.125 mm, N = 24 (verification): step 1 [0_4/45/0_4/-45/90_2]s -> step 2 [0_4/45/0_4/-45/90_2]s (unchanged)
  - C1 t = 0.18288 mm, N = 24 (verification): step 1 [0_4/45/0_4/-45/90_2]s -> step 2 [0_4/45/0_4/-45/90_2]s (unchanged)
- lambda is now reported with 16 x 16 terms (step 1: 8 x 8, the optimisation model) and the reported design of a buckling problem is the best of the k best 8 x 8 designs by the 16 x 16 value.
- Default N: when no QI laminate up to 64 plies meets the constraints (C8), N is the fewest plies of the objective min_plies (step 1: infeasible, no N).
- deliverables/optimization/stiffness_step1/ is kept unchanged as the record of step 1.

## Reproducibility

- Code: git commit bedc356 (the step-2 code itself is in the commit that adds this folder); run from python/: `..\venv\Scripts\python.exe -m optimization.experiments.stiffness_step2` (add --reuse to continue a stopped run).
- Python 3.12.10, numpy 2.5.4, scipy 1.18.1; Windows 11.
- GA seeds 0-19, budget 10 000 distinct designs per run; at most 2 worker processes.
- tradeoff_C6_N24_*.npz (lambda 8x8 and strength margin of every valid design, float32) are written by the 'tradeoff' stage of the same script.
- Time of the last invocation (stages reused from earlier, stopped runs not counted; see the stage times in the tables and the run log): 4 s.

## Run log (run_log.jsonl)

Free physical memory at the start of each stage (a stage starts only with at least 4 GB free), worker processes of the stage and the tasks already saved by an earlier, stopped run (reused with --reuse).

| time | stage | event | free memory (GB) | workers | tasks | already saved |
|---|---|---|---|---|---|---|
| 2026-11-20 15:26:54 | problems | start | 5.54 | 1 | 15 | 14 |
| 2026-11-20 16:07:08 | problems | start | 7.79 | 1 | 15 | 14 |
| 2026-11-20 17:07:47 | ga_runs | start | 6.36 | 2 | 600 | 0 |
| 2026-11-20 17:30:42 | ga_runs | stopped by the user (machine shutdown); finished runs saved | - | - | - | 272 |
| 2026-11-20 20:07:59 | ga_runs | start | 5.02 | 2 | 600 | 272 |
| 2026-11-20 20:32:55 | tradeoff | start | 9.08 | 2 | 16 | 0 |
| 2026-11-20 20:44:43 | matlab_check | start | 9.05 | 0 | 30 | 0 |
| 2026-11-20 20:45:29 | regression_demo | start | 8.06 | 0 | 1 | 0 |
