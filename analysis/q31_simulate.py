"""Question 3.1, third item: fly the Case A plan in the point-mass simulator under both actuation modes.

  "accel"   the engine tracks the planned acceleration: T(t) = m(t) u_k  (thrust decreases as the mass drops)
  "thrust"  the engine holds the force of the start of the interval: T(t) = m_k u_k on [t_k, t_k+1)

Usage (from the repository root):  python -m analysis.q31_simulate
  -> table printed and written to analysis/results/q31_simulate.md, figure in analysis/figures/q31_simulate.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lander.params import P as PARAMS
from lander.pointmass import simulate
from lander.style import C, apply
from analysis.q31_plots import solve_case_a, FIG_DIR

RES_DIR = os.path.join(os.path.dirname(__file__), "results")
MODES = ("accel", "thrust")
COLORS = dict(accel=C["blue"], thrust=C["orange"])


# ---------------------------------------------------------------------- data
def simulate_modes(sol, p=PARAMS, g=-10.0, modes=MODES):
    """{mode: simulate(...) result} for each actuation mode."""
    return {mode: simulate(sol, p, g, mode=mode) for mode in modes}


def fine_time(sol, sim):
    """Time of each simulator sub-step (the simulator uses n_sub RK4 steps per interval)."""
    return np.arange(len(sim["thrust"])) * sol.tgo / len(sim["thrust"])


def actual_accel(sim):
    """Thrust acceleration ||T(t)|| / m(t) actually produced at each sub-step."""
    return sim["thrust"] / sim["traj"][:-1, 4]


def error_table(sol, sims):
    """Rows (mode, final position error m, final velocity error m/s, fuel g, landing (x, y), thrust violation N)."""
    rows = []
    for mode, sim in sims.items():
        xf = sim["x_final"]
        rows.append((mode, sim["final_pos_err"], sim["final_vel_err"], 1e3 * sim["fuel"], xf[0], xf[1],
                     sim["thrust_violation"]))
    return rows


def table_markdown(sol, rows):
    lines = [f"Case A plan: t_f = {sol.tgo:.3f} s, N = {len(sol.u)}, planned fuel = {1e3 * sol.fuel:.2f} g",
             "",
             "| mode | final position error (m) | final velocity error (m/s) | fuel (g) | x_final (m) | y_final (m) "
             "| thrust violation (N) |",
             "|---|---|---|---|---|---|---|"]
    for mode, ep, ev, fu, xf, yf, vt in rows:
        lines.append(f"| `{mode}` | {ep:.2e} | {ev:.2e} | {fu:.2f} | {xf:+.2e} | {yf:+.2e} | {vt:.2e} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- plots
def plot_trajectories(ax, sol, sims, zoom=None):
    """Planned nodes and the simulated trajectories; zoom = half-width (m) of a window around the pad."""
    ax.plot(sol.p[:, 0], sol.p[:, 1], "o", color=C["ink2"], ms=3, mfc="none", label="plan (nodes)")
    for mode, sim in sims.items():
        tr = sim["traj"]
        ax.plot(tr[:, 0], tr[:, 1], color=COLORS[mode], lw=1.4, ls="-" if mode == "accel" else "--",
                label=f"simulated, {mode}")
        if zoom is not None:
            ax.plot(*tr[-1, :2], "o", color=COLORS[mode], ms=5)
    ax.plot(0, 0, "*", color=C["red"], ms=10, label="pad")
    if zoom is not None:
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
        ax.set_title(f"Zoom on touchdown (±{100 * zoom:.0f} cm)")
    else:
        ax.set_title("Planned and simulated trajectories")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")


def plot_accel_deviation(ax, sol, sims):
    """Relative deviation (%) of the produced thrust acceleration ||T||/m from the planned ||u_k||."""
    for mode, sim in sims.items():
        a = actual_accel(sim)
        plan = np.repeat(np.linalg.norm(sol.u, axis=1), len(a) // len(sol.u))
        ax.plot(fine_time(sol, sim), 100 * (a / plan - 1), color=COLORS[mode], lw=1.2,
                ls="-" if mode == "accel" else "--", label=mode)
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xlabel("t (s)"); ax.set_ylabel(r"$(\|T\|/m) / \|u_k\| - 1$ (%)")
    ax.set_title("Produced vs planned acceleration")
    ax.legend(loc="upper left")


def plot_thrust_near_rho1(ax, sol, sims, p=PARAMS):
    """Thrust force ||T(t)|| during the minimum-thrust arc, zoomed on rho1."""
    for mode, sim in sims.items():
        ax.plot(fine_time(sol, sim), sim["thrust"], color=COLORS[mode], lw=1.2,
                ls="-" if mode == "accel" else "--", label=mode)
    ax.axhline(p.rho1, color=C["red"], lw=0.9, ls=":")
    ax.text(0.02, p.rho1, r" $\rho_1$", va="bottom", color=C["red"], fontsize=8)
    on_rho1 = np.linalg.norm(sol.thrust, axis=1) < p.rho1 + 0.1
    t_end = sol.t[1:][on_rho1].max() if on_rho1.any() else sol.tgo
    ax.set_xlim(0, t_end); ax.set_ylim(p.rho1 - 0.08, p.rho1 + 0.08)
    ax.set_xlabel("t (s)"); ax.set_ylabel(r"$\|T(t)\|$ (N)")
    ax.set_title(r"Thrust force on the minimum-thrust arc")
    ax.legend(loc="upper right")


# ---------------------------------------------------------------------- run
def run(fig_dir=FIG_DIR, res_dir=RES_DIR):
    apply()
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    _, sol, _ = solve_case_a()
    sims = simulate_modes(sol)
    rows = error_table(sol, sims)
    md = table_markdown(sol, rows)
    with open(os.path.join(res_dir, "q31_simulate.md"), "w") as f:
        f.write("# Question 3.1 — Case A plan flown in both actuation modes\n\n"
                "Generated by `python -m analysis.q31_simulate`.\n\n" + md + "\n")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    plot_trajectories(ax[0, 0], sol, sims)
    plot_trajectories(ax[0, 1], sol, sims, zoom=0.05)
    plot_accel_deviation(ax[1, 0], sol, sims)
    plot_thrust_near_rho1(ax[1, 1], sol, sims)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "q31_simulate.png"))
    fig.savefig(os.path.join(fig_dir, "q31_simulate.pdf"))
    plt.close(fig)
    return sol, sims, md


if __name__ == "__main__":
    sol, sims, md = run()
    print(md)
    th = sims["thrust"]
    print(f"\nthrust mode, last interval: planned ||u|| = {np.linalg.norm(sol.u[-1]):.3f} m/s^2, "
          f"produced at its end = {actual_accel(th)[-1]:.3f} m/s^2")
    print(f"results in {RES_DIR}, figure in {FIG_DIR}")
