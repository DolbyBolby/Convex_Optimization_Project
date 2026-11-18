"""Question 3.2: Case B (fuel-starved), two-stage minimum-landing-error guidance (Blackmore et al., 2010).

  1. check that the pad center cannot be reached: solve_min_fuel finds no feasible t_f;
  2. stage 1: d* = min over t_f of |x_N| (mode="error"), stage 2: minimum fuel with |x_N| <= d* + 1 mm at t_f*
     (mode="fuel_with_error_bound"), both through solve_min_landing_error;
  3. figures: landing error versus t_f (stage 1), and the stage-2 trajectory, thrust and relaxation gap;
     the stage-2 plan is also flown in the point-mass simulator.

Usage (from the repository root):  python -m analysis.q32_case_b
  -> report printed and written to analysis/results/q32_case_b.md, figures analysis/figures/q32_case_b_*.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lander.lcvx import LanderSOCP, solve_min_fuel, solve_min_landing_error, tgo_bounds, is_lossless
from lander.params import P as PARAMS
from lander.pointmass import simulate
from lander.style import C, apply
from analysis.q31_plots import plot_trajectory, plot_thrust, plot_gap, FIG_DIR

X0_CASE_B = np.array([-6.0, 9.0, -6.0, 0.0])     # r0 = (-6, 9) m, v0 = (-6, 0) m/s
FUEL_CASE_B = 0.17                               # kg of fuel
RES_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------- data
def error_curve(tf, x0=X0_CASE_B, m0=None, p=PARAMS, N=40):
    """Stage-1 landing error |x_N| at each t_f of the array tf. Returns (err_certified, err_any): np.inf where
    infeasible; err_certified is also np.inf where the solution fails is_lossless."""
    m0 = p.m_dry + FUEL_CASE_B if m0 is None else m0
    prob = LanderSOCP(p, N=N, mode="error")
    cert, anyv = np.full(len(tf), np.inf), np.full(len(tf), np.inf)
    for i, t in enumerate(tf):
        sol = prob.solve(x0, t, m0)
        if sol is not None:
            anyv[i] = sol.landing_error
            if is_lossless(sol):
                cert[i] = sol.landing_error
    return cert, anyv


def case_b(p=PARAMS, N=40):
    """Everything reported for Case B, as a dict."""
    m0 = p.m_dry + FUEL_CASE_B
    fuel_sol, fuel_evals = solve_min_fuel(LanderSOCP(p, N=N), X0_CASE_B, m0)
    out = dict(m0=m0, bounds=tgo_bounds(p, X0_CASE_B, m0), reachable=fuel_sol is not None,
               n_fuel_evals=len(fuel_evals), n_fuel_feasible=sum(np.isfinite(f) for _, f in fuel_evals))
    for certify in (True, False):
        sol, d_star, evals = solve_min_landing_error(p, X0_CASE_B, m0, N=N, certify=certify)
        out[certify] = dict(sol=sol, d_star=d_star, evals=evals)
    return out


def report_markdown(res, sims, p=PARAMS):
    b = res[True]; sol = b["sol"]
    lines = [f"Case B: r0 = (-6, 9) m, v0 = (-6, 0) m/s, fuel = {FUEL_CASE_B} kg (m0 = {res['m0']:.2f} kg), N = 40",
             "",
             f"**Reachability.** solve_min_fuel: {res['n_fuel_feasible']}/{res['n_fuel_evals']} feasible t_f on the "
             f"grid over [t_min, t_max] = [{res['bounds'][0]:.2f}, {res['bounds'][1]:.2f}] s -> "
             + ("pad center REACHABLE" if res["reachable"] else "**no feasible t_f: the pad center cannot be reached**"),
             "",
             "| | certify=True (reported) | certify=False |",
             "|---|---|---|"]
    rows = [("minimum landing error d* (m)", lambda r: f"{r['d_star']:.3f}"),
            ("flight time t_f* (s)", lambda r: f"{r['sol'].tgo:.3f}"),
            ("stage-2 fuel consumed (g)", lambda r: f"{1e3 * r['sol'].fuel:.2f}"),
            ("stage-2 landing error \\|x_N\\| (m)", lambda r: f"{r['sol'].landing_error:.3f}"),
            ("fuel left at touchdown (g)", lambda r: f"{1e3 * (np.exp(r['sol'].z[-1]) - p.m_dry):.3f}"),
            ("stage-2 solution certified (is_lossless)", lambda r: str(is_lossless(r["sol"]))),
            ("stage-1 t_f with a feasible (certified) solution",
             lambda r: f"{sum(np.isfinite(v) for _, v in r['evals'])}/{len(r['evals'])}")]
    for name, fmt in rows:
        lines.append(f"| {name} | {fmt(res[True])} | {fmt(res[False])} |")
    lines += ["", "Stage-2 plan flown in the point-mass simulator (errors relative to the planned touchdown point):", "",
              "| mode | final position error (m) | final velocity error (m/s) | fuel (g) | x_final (m) |",
              "|---|---|---|---|---|"]
    for mode, sim in sims.items():
        lines.append(f"| `{mode}` | {sim['final_pos_err']:.2e} | {sim['final_vel_err']:.2e} | "
                     f"{1e3 * sim['fuel']:.2f} | {sim['x_final'][0]:+.3f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- plots
def plot_error_vs_tf(ax, tf, cert, anyv, sol=None, d_star=None):
    """Stage 1: minimum landing error |x_N| versus t_f (certified, feasible but lossy), with t_f* and d*."""
    ok, lossy = np.isfinite(cert), np.isfinite(anyv) & ~np.isfinite(cert)
    ax.plot(tf, np.where(ok, cert, np.nan), "o-", color=C["blue"], ms=3, label="stage 1, certified")  # gaps = none
    if lossy.any():
        ax.plot(tf[lossy], anyv[lossy], "x", color=C["orange"], ms=5, label="feasible, not certified")
    if sol is not None:
        ax.plot(sol.tgo, d_star, "*", color=C["red"], ms=11, zorder=5,
                label=f"$d^*$ = {d_star:.3f} m at $t_f^*$ = {sol.tgo:.2f} s")
    feas = np.isfinite(anyv)
    if feas.any():
        pad = 0.15 * (tf[feas][-1] - tf[feas][0] + 0.1)
        ax.set_xlim(tf[feas][0] - pad, tf[feas][-1] + pad)
    ax.set_xlabel(r"time of flight $t_f$ (s)"); ax.set_ylabel(r"landing error $|x_N|$ (m)")
    ax.set_title("Stage 1: minimum landing error versus $t_f$")
    ax.legend(loc="best")


# ---------------------------------------------------------------------- run
def run(fig_dir=FIG_DIR, res_dir=RES_DIR):
    apply()
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    res = case_b()
    b = res[True]
    sol, d_star = b["sol"], b["d_star"]
    assert sol is not None, "Case B: no stage-2 solution"
    sims = {mode: simulate(sol, mode=mode) for mode in ("accel", "thrust")}
    md = report_markdown(res, sims)
    with open(os.path.join(res_dir, "q32_case_b.md"), "w") as f:
        f.write("# Question 3.2 — Case B, two-stage minimum-landing-error guidance\n\n"
                "Generated by `python -m analysis.q32_case_b`.\n\n" + md + "\n")

    feas = [t for t, v in res[False]["evals"] if np.isfinite(v)]
    tf = np.unique(np.r_[np.linspace(min(feas) - 0.3, max(feas) + 0.3, 90), sol.tgo])
    cert, anyv = error_curve(tf)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    plot_error_vs_tf(ax, tf, cert, anyv, sol, d_star)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "q32_case_b_error_vs_tf.png"))
    fig.savefig(os.path.join(fig_dir, "q32_case_b_error_vs_tf.pdf"))

    fig2, ax2 = plt.subplots(1, 3, figsize=(13, 3.8))
    plot_trajectory(ax2[0], sol)
    ax2[0].set_title(f"Case B, stage 2: $t_f^*$ = {sol.tgo:.2f} s, fuel = {1e3 * sol.fuel:.1f} g")
    plot_thrust(ax2[1], sol)
    plot_gap(ax2[2], sol)
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, "q32_case_b_solution.png"))
    fig2.savefig(os.path.join(fig_dir, "q32_case_b_solution.pdf"))
    plt.close("all")
    return res, md


if __name__ == "__main__":
    res, md = run()
    print(md)
    print(f"\nresults in {RES_DIR}, figures in {FIG_DIR}")
