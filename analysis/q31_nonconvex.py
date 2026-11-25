"""Question 3.1, fourth item: our SOCP vs the direct nonconvex baseline (SLSQP on the unrelaxed problem (P)).

Both methods use N = 20 and the same t_f: the optimum found by solve_min_fuel with N = 20 (Case A). The baseline
(lander/nonconvex.py, provided) is run from 12 random initial guesses; the SOCP has no initial guess (convex: the
solver returns the global optimum), it is solved 12 times to measure its time.

Usage (from the repository root):  python -m analysis.q31_nonconvex
  -> table printed and written to analysis/results/q31_nonconvex.md, figure in analysis/figures/q31_nonconvex.{png,pdf}
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lander.lcvx import LanderSOCP, solve_min_fuel, is_lossless
from lander.nonconvex import nonconvex_baseline
from lander.params import P as PARAMS
from lander.style import C, apply
from analysis.q31_plots import X0_CASE_A, FIG_DIR

RES_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------- data
def socp_runs(x0=X0_CASE_A, m0=None, N=20, n_runs=12, p=PARAMS):
    """t_f* from solve_min_fuel with N intervals, then n_runs solves at t_f*.
    Returns (tf, compile_time, list of dicts: converged, feasible, certified, fuel kg, time s)."""
    m0 = p.m_wet if m0 is None else m0
    t0 = time.perf_counter()
    prob = LanderSOCP(p, N=N)
    sol = prob.solve(x0, 3.0, m0)                    # first solve: CVXPY compiles the problem (DPP) once
    compile_time = time.perf_counter() - t0
    best, _ = solve_min_fuel(prob, x0, m0)
    tf = best.tgo
    runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sol = prob.solve(x0, tf, m0)
        dt = time.perf_counter() - t0
        runs.append(dict(converged=sol is not None, feasible=sol is not None, certified=bool(sol and is_lossless(sol)),
                         fuel=sol.fuel if sol else np.nan, time=dt, solver_time=sol.solve_time if sol else np.nan))
    return tf, compile_time, runs


def socp_in_baseline_model(x0, tf, m0=None, N=20, p=PARAMS, g=-10.0):
    """Fly the SOCP thrusts T_k = m_k u_k in the baseline's own discrete model (force held on each interval, mass of
    the start of the interval): returns (fuel kg, |r_N| m, |v_N| m/s), for a like-for-like objective comparison."""
    m0 = p.m_wet if m0 is None else m0
    sol = LanderSOCP(p, N=N).solve(x0, tf, m0)
    dt, gv = tf / N, np.array([0.0, g])
    r, v, m = np.array(x0[:2], float), np.array(x0[2:], float), m0
    for T in sol.thrust:
        a = T / m + gv
        r, v, m = r + v * dt + 0.5 * a * dt ** 2, v + a * dt, m - p.alpha * np.linalg.norm(T) * dt
    return m0 - m, float(np.linalg.norm(r)), float(np.linalg.norm(v))


def summary(runs, ref_fuel):
    fuel = np.array([r["fuel"] for r in runs if r["feasible"]])
    return dict(n=len(runs), converged=sum(r["converged"] for r in runs), feasible=len(fuel),
                at_opt=int(np.sum(np.abs(fuel - ref_fuel) <= 1e-3 * ref_fuel)),
                best=fuel.min() if len(fuel) else np.nan, median=np.median(fuel) if len(fuel) else np.nan,
                worst=fuel.max() if len(fuel) else np.nan,
                t_mean=np.mean([r["time"] for r in runs]), t_max=max(r["time"] for r in runs))


def tables_markdown(tf, compile_time, socp, slsqp, N=20, in_baseline=None):
    ref = socp[0]["fuel"]
    s, b = summary(socp, ref), summary(slsqp, ref)
    g = lambda v: f"{1e3 * v:.2f}" if np.isfinite(v) else "–"
    lines = [f"Case A, N = {N}, t_f = {tf:.3f} s (optimum of solve_min_fuel with N = {N}); "
             f"SOCP fuel = {1e3 * ref:.2f} g, certified = {socp[0]['certified']}", "",
             "| | SOCP (LCvx, Clarabel) | SLSQP on (P), 12 random starts |",
             "|---|---|---|",
             f"| solver reports success | {s['converged']}/{s['n']} | {b['converged']}/{b['n']} |",
             f"| feasible (constraints of (P) to 1e-4) | {s['feasible']}/{s['n']} | {b['feasible']}/{b['n']} |",
             f"| reaches the SOCP optimum (±0.1 %) | {s['at_opt']}/{s['n']} | {b['at_opt']}/{b['n']} |",
             f"| fuel, best / median / worst feasible (g) | {g(s['best'])} / {g(s['median'])} / {g(s['worst'])} "
             f"| {g(b['best'])} / {g(b['median'])} / {g(b['worst'])} |",
             f"| time per solve, mean / max (s) | {s['t_mean']:.3f} / {s['t_max']:.3f} | {b['t_mean']:.2f} / {b['t_max']:.2f} |",
             f"| one-time compilation (s) | {compile_time:.2f} (DPP, once for all t_f) | – |",
             *([f"| SOCP plan flown in the baseline's discrete model | fuel {1e3 * in_baseline[0]:.2f} g, "
                f"|r_N| = {in_baseline[1]:.1e} m, |v_N| = {in_baseline[2]:.1e} m/s | – |"] if in_baseline else []),
             "", "SLSQP run by run:", "",
             "| start | converged | feasible | fuel (g) | fuel − SOCP (%) | time (s) |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(slsqp):
        d = f"{100 * (r['fuel'] / ref - 1):+.2f}" if r["feasible"] else "–"
        lines.append(f"| {i} | {'yes' if r['converged'] else 'no'} | {'yes' if r['feasible'] else '**no**'} | "
                     f"{1e3 * r['fuel']:.2f} | {d} | {r['time']:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- plot
def plot_comparison(ax_fuel, ax_time, socp, slsqp):
    ref = socp[0]["fuel"]
    k = np.arange(len(slsqp))
    feas = np.array([r["feasible"] for r in slsqp])
    fuel = 1e3 * np.array([r["fuel"] for r in slsqp])
    ax_fuel.axhline(1e3 * ref, color=C["blue"], lw=1.4, label=f"SOCP optimum ({1e3 * ref:.1f} g)")
    ax_fuel.plot(k[feas], fuel[feas], "o", color=C["orange"], label="SLSQP, feasible")
    if (~feas).any():
        ax_fuel.plot(k[~feas], fuel[~feas], "x", color=C["red"], ms=7, label="SLSQP, infeasible")
    ax_fuel.set_xticks(k); ax_fuel.set_xlabel("SLSQP initial guess"); ax_fuel.set_ylabel("fuel (g)")
    ax_fuel.set_title("Objective reached")
    ax_fuel.legend(loc="best")
    ax_time.bar(k, [r["time"] for r in slsqp], color=[C["orange"] if f else C["red"] for f in feas], label="SLSQP")
    ax_time.axhline(np.mean([r["time"] for r in socp]), color=C["blue"], lw=1.4, label="SOCP (mean)")
    ax_time.set_yscale("log")
    ax_time.set_xticks(k); ax_time.set_xlabel("SLSQP initial guess"); ax_time.set_ylabel("time (s)")
    ax_time.set_title("Computation time per solve")
    ax_time.legend(loc="center right")


def run(fig_dir=FIG_DIR, res_dir=RES_DIR, N=20, n_starts=12):
    apply()
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    tf, compile_time, socp = socp_runs(N=N, n_runs=n_starts)
    slsqp = nonconvex_baseline(X0_CASE_A, tf, N=N, n_starts=n_starts)
    md = tables_markdown(tf, compile_time, socp, slsqp, N, socp_in_baseline_model(X0_CASE_A, tf, N=N))
    with open(os.path.join(res_dir, "q31_nonconvex.md"), "w") as f:
        f.write("# Question 3.1 — SOCP vs nonconvex SLSQP baseline\n\n"
                "Generated by `python -m analysis.q31_nonconvex`.\n\n" + md + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    plot_comparison(ax[0], ax[1], socp, slsqp)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "q31_nonconvex.png"))
    fig.savefig(os.path.join(fig_dir, "q31_nonconvex.pdf"))
    plt.close(fig)
    return tf, socp, slsqp, md


if __name__ == "__main__":
    *_, md = run()
    print(md)
    print(f"\nresults in {RES_DIR}, figure in {FIG_DIR}")
