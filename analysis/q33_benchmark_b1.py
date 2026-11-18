"""Question 3.3: Benchmark B1 (20 frozen initial states, lander/pointmass.py) with our LCvx planner.

Guidance: minimum fuel (solve_min_fuel); when the pad is unreachable (no feasible t_f), the two-stage minimum
landing error guidance of Question 3.2 (solve_min_landing_error). As long as Question 3.2 is not implemented,
the unreachable cases 17-19 are reported as "waiting for Q3.2"; nothing else has to change once it is.
The benchmark is run twice: certify=False (lossy solutions accepted) and certify=True (only lossless ones).

Usage (from the repository root):  python -m analysis.q33_benchmark_b1
  -> tables printed and written to analysis/results/q33_b1_certify_{false,true}.md
"""
import os
import numpy as np

from lander.lcvx import LanderSOCP, solve_min_fuel, solve_min_landing_error
from lander.params import P as PARAMS
from lander.pointmass import run_b1, B1_CASES, B1_UNREACHABLE

RES_DIR = os.path.join(os.path.dirname(__file__), "results")


def q32_available(p=PARAMS):
    """True once solve_min_landing_error (Question 3.2) is implemented."""
    case = B1_CASES[B1_UNREACHABLE[0]]
    try:
        solve_min_landing_error(p, case[:4], p.m_dry + case[4], n_grid=3)
    except NotImplementedError:
        return False
    return True


def make_guidance(certify=True, p=PARAMS, N=40, use_q32=True):
    """(guidance(x0, m0) -> LanderSolution | None, d_star_fn or None) as expected by run_b1.
    The minimum-fuel SOCP is compiled once and reused for every case (DPP)."""
    prob = LanderSOCP(p, N=N)
    stage = {}                                  # (x0, m0) -> (sol, d_star) of the two-stage guidance

    def two_stage(x0, m0):
        key = (tuple(np.round(x0, 9)), round(float(m0), 9))
        if key not in stage:
            sol, d_star, _ = solve_min_landing_error(p, x0, m0, N=N, certify=certify)
            stage[key] = (sol, d_star)
        return stage[key]

    def guidance(x0, m0):
        sol, _ = solve_min_fuel(prob, x0, m0, certify=certify)
        if sol is None and use_q32:
            sol = two_stage(x0, m0)[0]
        return sol

    return guidance, ((lambda x0, m0: two_stage(x0, m0)[1]) if use_q32 else None)


def summarize(rows, q32):
    """Summary dict over all cases (landed counts, fuel, worst violations, solve times)."""
    reach = [r for r in rows if r["reachable"]]
    unreach = [r for r in rows if not r["reachable"]]
    flown = [r for r in rows if r["ok"]]
    return dict(
        landed_reachable=sum(r["landed"] for r in reach), n_reachable=len(reach),
        landed_unreachable=sum(r["landed"] for r in unreach) if q32 else None, n_unreachable=len(unreach),
        fuel_mean_g=1e3 * np.mean([r["fuel"] for r in flown]) if flown else np.nan,
        viol_T_max=max((r["viol_T"] for r in flown), default=np.nan),
        viol_gs_max=max((r["viol_gs"] for r in flown), default=np.nan),
        gap_max=max((r["gap"] for r in flown), default=np.nan),
        t_solve_mean=np.mean([r["t_solve"] for r in rows]), t_solve_max=max(r["t_solve"] for r in rows))


def table_markdown(rows, certify, q32):
    s = summarize(rows, q32)
    unreach = (f"{s['landed_unreachable']}/{s['n_unreachable']}" if q32 else "waiting for Q3.2")
    lines = [f"## Benchmark B1, certify={certify}", "",
             f"- landed, reachable cases 0-16: **{s['landed_reachable']}/{s['n_reachable']}**",
             f"- landed within d* + 5 cm, unreachable cases 17-19: **{unreach}**",
             f"- mean fuel (flown cases): {s['fuel_mean_g']:.1f} g",
             f"- worst thrust violation: {s['viol_T_max']:.2e} N, worst glide-slope violation: {s['viol_gs_max']:.2e} m",
             f"- worst nodal relaxation gap max_k |sigma_k - ||u_k|||: {s['gap_max']:.2e} m/s^2",
             f"- solve time: mean {s['t_solve_mean']:.2f} s, max {s['t_solve_max']:.2f} s", "",
             "| case | x0 | y0 | vx0 | vy0 | fuel0 (kg) | reachable | landed | d_land (m) | d* (m) | fuel (g) "
             "| viol_T (N) | viol_gs (m) | gap (m/s^2) | t_solve (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = B1_CASES[r["case"]]
        head = f"| {r['case']} | {c[0]:+.2f} | {c[1]:.2f} | {c[2]:+.2f} | {c[3]:+.2f} | {c[4]:.3f} | " \
               f"{'yes' if r['reachable'] else 'no'} | "
        if not r["ok"]:
            why = "waiting for Q3.2" if (not r["reachable"] and not q32) else "no solution"
            lines.append(head + f"**no** ({why}) | – | – | – | – | – | – | {r['t_solve']:.2f} |")
            continue
        ds = f"{r['d_star']:.3f}" if r["d_star"] is not None else "–"
        lines.append(head + f"{'yes' if r['landed'] else '**no**'} | {r['d_land']:.3f} | {ds} | "
                     f"{1e3 * r['fuel']:.1f} | {r['viol_T']:.2e} | {r['viol_gs']:.2e} | {r['gap']:.2e} | "
                     f"{r['t_solve']:.2f} |")
    return "\n".join(lines)


def run(res_dir=RES_DIR):
    os.makedirs(res_dir, exist_ok=True)
    q32 = q32_available()
    out = {}
    for certify in (False, True):
        guidance, d_star_fn = make_guidance(certify=certify, use_q32=q32)
        rows = run_b1(guidance, d_star_fn)
        md = table_markdown(rows, certify, q32)
        with open(os.path.join(res_dir, f"q33_b1_certify_{str(certify).lower()}.md"), "w") as f:
            f.write("# Question 3.3 — Benchmark B1\n\nGenerated by `python -m analysis.q33_benchmark_b1`.\n\n" + md + "\n")
        out[certify] = (rows, md)
    return out, q32


if __name__ == "__main__":
    out, q32 = run()
    if not q32:
        print("Question 3.2 (solve_min_landing_error) is not implemented yet: cases 17-19 are reported as pending.\n")
    for certify, (rows, md) in out.items():
        print(md + "\n")
    print(f"results in {RES_DIR}")
