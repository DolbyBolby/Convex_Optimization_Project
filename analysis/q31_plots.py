"""Figures of Question 3.1 for Case A (point-mass lander, minimum fuel, touchdown at the pad center).

  (a) the optimal trajectory with the glide-slope cone, the thrust magnitude ||T_k|| with its limits rho1, rho2,
      and the nodal relaxation gap sigma_k - ||u_k|| (with the certificate of is_lossless);
  (b) the optimal fuel as a function of the time of flight t_f over the feasible range.

Usage (from the repository root):  python -m analysis.q31_plots      -> figures in analysis/figures/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lander.lcvx import LanderSOCP, solve_min_fuel, tgo_bounds, is_lossless
from lander.params import P as PARAMS
from lander.style import C, apply

X0_CASE_A = np.array([-6.0, 10.0, -2.0, -1.0])   # r0 = (-6, 10) m, v0 = (-2, -1) m/s, full tank
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


# ---------------------------------------------------------------------- data
def solve_case_a(p=PARAMS, N=40, gamma_deg=45.0):
    """Fuel-optimal solution of Case A; returns (prob, sol, evals) as in solve_min_fuel."""
    prob = LanderSOCP(p, N=N, gamma_deg=gamma_deg)
    sol, evals = solve_min_fuel(prob, X0_CASE_A, p.m_wet)
    return prob, sol, evals


def fuel_curve(prob, tf, x0=X0_CASE_A, m0=None):
    """Optimal fuel (kg) at each t_f of the array tf. Returns (fuel_certified, fuel_any): np.inf where the SOCP is
    infeasible; fuel_certified is also np.inf where the solution fails is_lossless (lossy relaxation)."""
    m0 = prob.p.m_wet if m0 is None else m0
    cert, anyf = np.full(len(tf), np.inf), np.full(len(tf), np.inf)
    for i, t in enumerate(tf):
        sol = prob.solve(x0, t, m0)
        if sol is not None:
            anyf[i] = sol.fuel
            if is_lossless(sol):
                cert[i] = sol.fuel
    return cert, anyf


# ---------------------------------------------------------------------- plots
def plot_trajectory(ax, sol, gamma_deg=45.0, n_arrows=10):
    """Trajectory (x, y) with the glide-slope cone y >= |x - x_N| tan(gamma) and the thrust direction."""
    x, y = sol.p[:, 0], sol.p[:, 1]
    xN = x[-1]
    tg = np.tan(np.radians(gamma_deg))
    h = 1.1 * max(y.max(), 1.0)
    xs = np.array([xN - h / tg, xN, xN + h / tg])
    ax.fill_between(xs, np.abs(xs - xN) * tg, h, color=C["aqua"], alpha=0.10, lw=0, label="glide-slope cone")
    ax.plot(xs, np.abs(xs - xN) * tg, color=C["aqua"], lw=1.0)
    ax.plot(x, y, color=C["blue"], marker="o", ms=2.5, label="optimal trajectory")
    T = sol.thrust
    k = np.linspace(0, len(T) - 1, n_arrows).astype(int)
    scale = 0.15 * h / PARAMS.rho2                           # arrow of 0.15 h for full thrust
    ax.quiver(x[k], y[k], T[k, 0] * scale, T[k, 1] * scale, angles="xy", scale_units="xy", scale=1,
              color=C["orange"], width=0.004, label="thrust direction")
    ax.plot(x[0], y[0], "s", color=C["ink"], ms=4, label="start")
    ax.plot(xN, y[-1], "*", color=C["red"], ms=9, label="pad")
    ax.set_xlim(min(x.min(), xN - 1.0) - 1.0, max(x.max(), xN + 1.0) + 1.0)
    ax.set_ylim(-0.3, h)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Optimal trajectory, $t_f$ = {sol.tgo:.2f} s")
    ax.legend(loc="upper right")


def plot_thrust(ax, sol, p=PARAMS):
    """Thrust magnitude ||T_k|| = m_k ||u_k|| (piecewise constant) with the limits rho1 and rho2."""
    t, Tn = sol.t, np.linalg.norm(sol.thrust, axis=1)
    ax.stairs(Tn, t, baseline=None, color=C["blue"], lw=1.6, label=r"$\|T_k\|$")
    ax.stairs(np.exp(sol.z[:-1]) * sol.sigma, t, baseline=None, color=C["orange"], lw=1.0, ls="--", label=r"$\Gamma_k = m_k\sigma_k$")
    for v, name in ((p.rho1, r"$\rho_1$"), (p.rho2, r"$\rho_2$")):
        ax.axhline(v, color=C["ink2"], lw=0.9, ls=":")
        ax.text(t[-1], v, f" {name} = {v:.1f} N", va="center", ha="left", fontsize=8, color=C["ink2"])
    ax.set_ylim(0, 1.15 * p.rho2)
    ax.set_xlabel("t (s)"); ax.set_ylabel("thrust (N)")
    ax.set_title("Thrust magnitude")
    ax.legend(loc="lower left")


def plot_gap(ax, sol):
    """Nodal relaxation gap sigma_k - ||u_k|| (zero = the relaxation is exact at node k)."""
    gap = sol.relaxation_gap
    ok = is_lossless(sol)
    ax.stem(sol.t[:-1], gap, linefmt=C["blue"], markerfmt="o", basefmt=" ")
    ax.axhline(0, color=C["ink2"], lw=0.8)
    rel = np.max(gap / np.maximum(sol.sigma, 1e-9))
    ax.set_xlabel("t (s)"); ax.set_ylabel(r"$\sigma_k - \|u_k\|$ (m/s$^2$)")
    ax.set_title(f"Relaxation gap: max relative {rel:.1e} -> " + ("certified" if ok else "NOT certified"))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def plot_fuel_vs_tf(ax, tf, fuel_cert, fuel_any=None, sol=None, t_bounds=None, xlim=None):
    """Optimal fuel (g) versus t_f: certified solutions, feasible but lossy ones, infeasible range, optimum."""
    ok = np.isfinite(fuel_cert)
    ax.plot(tf[ok], 1e3 * fuel_cert[ok], color=C["blue"], marker="o", ms=2.5, label="optimal fuel (certified)")
    if fuel_any is not None:
        lossy = np.isfinite(fuel_any) & ~ok
        if lossy.any():
            ax.plot(tf[lossy], 1e3 * fuel_any[lossy], "x", color=C["orange"], ms=5, label="feasible, not certified")
    first = tf[np.isfinite(fuel_any if fuel_any is not None else fuel_cert)][0]
    ax.axvspan(tf[0], first, color=C["red"], alpha=0.08, lw=0, label="SOCP infeasible")
    if sol is not None:
        ax.plot(sol.tgo, 1e3 * sol.fuel, "*", color=C["red"], ms=11, zorder=5,
                label=f"optimum: {sol.tgo:.2f} s, {1e3 * sol.fuel:.1f} g")
    if xlim is not None:
        ax.set_xlim(*xlim)
        sel = (tf >= xlim[0]) & (tf <= xlim[1]) & ok
        ax.set_ylim(0.95e3 * fuel_cert[sel].min(), 1.05e3 * fuel_cert[sel].max())
    if t_bounds is not None:
        for v, name in zip(t_bounds, (r"$t_{min}$ (Q2.2)", r"$t_{max}$ (Q2.2)")):
            if xlim is None or xlim[0] <= v <= xlim[1]:
                ax.axvline(v, color=C["ink2"], lw=0.9, ls=":")
                ax.text(v, ax.get_ylim()[1], " " + name, rotation=90, va="top", ha="left", fontsize=8,
                        color=C["ink2"])
    ax.set_xlabel(r"time of flight $t_f$ (s)"); ax.set_ylabel("fuel (g)")
    ax.legend(loc="lower right" if xlim is None else "upper left")


# ---------------------------------------------------------------------- figures
def make_figures(out_dir=FIG_DIR):
    apply()
    os.makedirs(out_dir, exist_ok=True)
    prob, sol, _ = solve_case_a()
    assert sol is not None, "Case A has no certified solution"

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    plot_trajectory(ax[0], sol)
    plot_thrust(ax[1], sol)
    plot_gap(ax[2], sol)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q31_case_a_solution.png"))
    fig.savefig(os.path.join(out_dir, "q31_case_a_solution.pdf"))

    tb = tgo_bounds(prob.p, X0_CASE_A, prob.p.m_wet)
    tf = np.unique(np.r_[np.linspace(*tb, 150), np.linspace(sol.tgo - 0.15, sol.tgo + 0.3, 46)])
    cert, anyf = fuel_curve(prob, tf)
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 3.8))
    plot_fuel_vs_tf(ax2[0], tf, cert, anyf, sol, tb)
    ax2[0].set_title(r"Optimal fuel versus $t_f$ over $[t_{min}, t_{max}]$")
    plot_fuel_vs_tf(ax2[1], tf, cert, anyf, sol, tb, xlim=(sol.tgo - 0.15, sol.tgo + 1.0))
    ax2[1].set_title("Zoom near the optimum")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "q31_fuel_vs_tf.png"))
    fig2.savefig(os.path.join(out_dir, "q31_fuel_vs_tf.pdf"))
    plt.close("all")
    return sol, tf, cert, anyf


if __name__ == "__main__":
    sol, tf, cert, anyf = make_figures()
    Tn = np.linalg.norm(sol.thrust, axis=1)
    ok, feas = np.isfinite(cert), np.isfinite(anyf)
    print(f"Case A: t_f* = {sol.tgo:.3f} s, fuel = {1e3 * sol.fuel:.1f} g, certified = {is_lossless(sol)}")
    print(f"  max relaxation gap = {sol.relaxation_gap.max():.2e}, landing error = {sol.landing_error:.1e} m")
    print(f"  nodes at rho1: {np.sum(Tn < PARAMS.rho1 + 0.1)}, at rho2: {np.sum(Tn > PARAMS.rho2 - 0.1)}, "
          f"in between: {np.sum((Tn >= PARAMS.rho1 + 0.1) & (Tn <= PARAMS.rho2 - 0.1))} (of {len(Tn)})")
    print(f"  SOCP feasible from t_f = {tf[feas][0]:.3f} s, certified from {tf[ok][0]:.3f} s to {tf[ok][-1]:.2f} s")
    print(f"  Q2.2 bounds [{tf[0]:.2f}, {tf[-1]:.2f}] s; fuel at the longest certified t_f = {1e3 * cert[ok][-1]:.1f} g")
    i = ok & (tf > 6)
    print(f"  slope of fuel(t_f) for t_f > 6 s: {1e3 * np.polyfit(tf[i], cert[i], 1)[0]:.1f} g/s "
          f"(alpha rho1 = {1e3 * PARAMS.alpha * PARAMS.rho1:.1f} g/s, alpha m_wet |g| = {1e3 * PARAMS.alpha * PARAMS.m_wet * 10:.1f} g/s)")
    print(f"figures written to {FIG_DIR}")
