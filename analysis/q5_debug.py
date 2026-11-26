"""Debug view of one episode of the Windy Hazard Landing Challenge (Question 5).

One figure per (policy, scenario, seed):
  left   the map in world coordinates: terrain, pad, hazards, hazards inflated by the planning box R_X x R_Y
         (the set the centre of mass must avoid), optional convex regions of the policy (GCS / corridors),
         every plan computed during the flight (light = early, dark = late), the flown trajectory coloured by
         outcome, the planning box every second and at the moment of minimum hazard clearance;
  right  time series: position vs. active plan, tracking error (with the flight phase), signed clearance of
         the planning box to the hazards and to the terrain (and the minimum clearance of each plan), wind
         (true / known part / known + GustEstimator), tilt, engine commands.

What the tool reads from the policy (all optional, only for this offline analysis):
  * plans: either an explicit list `policy.debug_plans` of dicts
        {"t": time the plan starts (s), "p": (N+1, 2) planned positions (pad frame),
         "dt": node spacing (s) or "t_nodes": (N+1,) node times relative to "t",
         "margin": clearance margin used by the planner (m), ...}       (other keys are ignored)
    or, if absent, the attribute `plan` (a dict with "p" and "dt", as in CascadeController) of the policy or of
    one of its attributes; a new plan is detected when the object changes.
  * `policy.debug_regions`: list of convex polygons (pad frame) drawn on the map (e.g. the GCS regions).
  * `phase` (str), on the policy or on the same nested controller (e.g. "POWERED", "TERMINAL").
The wind estimate is recomputed here with a separate GustEstimator, so it works for any policy.

Usage (from the repository root):
  python -m analysis.q5_debug --policy lander.controllers:make_cascade --scenarios S04 --seeds 3
  python -m analysis.q5_debug --policy submissions/Q5/gcs.py --scenarios all --seeds 0-9 --failures-only
  python -m analysis.q5_debug --policy ... --scenarios S05 --seeds 1 --show       # interactive window
Figures: analysis/figures/debug/<policy>_<scenario>_seed<seed>.png (change with --out-dir).
"""
import argparse, os, time, warnings
import numpy as np
import matplotlib

from lander.env import FuelLander, score_episode
from lander.evaluate import ALL, get_factory
from lander.geometry import closest_point_convex, obstacle_polygons
from lander.params import R_X, R_Y, LEG_DOWN
from lander.scenarios import SCENARIOS
from lander.wind import GustEstimator, engine_powers, known_wind

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "debug")
OUTCOME_COLOR = dict(success="aqua", off_pad="yellow", hazard="red")      # keys of lander.style.C, else orange


# ---------------------------------------------------------------------- geometry
def inflate(poly, rx=R_X, ry=R_Y):
    """Minkowski sum of a convex polygon and the box [-rx, rx] x [-ry, ry] (CCW vertices).
    The box intersects the polygon iff its centre lies in the inflated polygon."""
    from scipy.spatial import ConvexHull
    P = np.asarray(poly, float)
    pts = np.concatenate([P + np.array([sx * rx, sy * ry]) for sx in (-1, 1) for sy in (-1, 1)])
    return pts[ConvexHull(pts).vertices]


def obstacles(spec):
    """Inflated hazards (rocks and zones) and inflated terrain pieces, pad frame.
    The terrain pieces under the pad are left out: the box always touches them at touchdown."""
    polys, _ = obstacle_polygons(spec)
    n_h = len(spec["hazards"])
    terrain, hazards = polys[:len(polys) - n_h], polys[len(polys) - n_h:]
    c = 0.5 * (spec["pad"][0] + spec["pad"][1])
    x1, x2 = spec["pad"][0] - c, spec["pad"][1] - c
    keep = [T for T in terrain
            if not (x1 - 1e-6 <= T[3, 0] and T[2, 0] <= x2 + 1e-6 and abs(T[2, 1] + LEG_DOWN) < 1e-6
                    and abs(T[3, 1] + LEG_DOWN) < 1e-6)]
    return [inflate(h) for h in hazards], [inflate(T) for T in keep]


def clearance(points, inflated):
    """Signed distance (m, < 0 = collision) of the planning box centred at each point to a set of inflated
    polygons (minimum over the polygons). Returns +inf if there is no polygon."""
    points = np.atleast_2d(points)
    out = np.full(len(points), np.inf)
    for Q in inflated:
        out = np.minimum(out, [closest_point_convex(Q, p)[1] for p in points])
    return out


# ---------------------------------------------------------------------- recording
def _find_planner(policy):
    """The object that holds `plan` / `phase`: the policy itself or one of its attributes."""
    if hasattr(policy, "plan") or hasattr(policy, "phase"):
        return policy
    for v in vars(policy).values():
        if hasattr(v, "plan") or hasattr(v, "phase"):
            return v
    return None


class PlanRecorder:
    """Collects the plans of a policy after every call of act()."""

    def __init__(self, policy):
        self.policy, self.inner = policy, _find_planner(policy)
        self.explicit = hasattr(policy, "debug_plans")
        self.plans, self._last, self._n = [], None, 0

    def after_act(self, t):
        if self.explicit:
            new = self.policy.debug_plans[self._n:]
            self._n = len(self.policy.debug_plans)
            self.plans += [dict(P) for P in new]
            return
        if self.inner is None:
            return
        P = getattr(self.inner, "plan", None)
        if P is self._last:
            return
        self._last = P
        if P is None:
            self.plans.append(dict(t=t, p=None))
        else:
            self.plans.append(dict(t=t, p=np.array(P["p"]), dt=P.get("dt"), tgo=P.get("tgo"),
                                   margin=P.get("margin")))

    def phase(self):
        return getattr(self.inner, "phase", None) if self.inner is not None else None


def run(source, sid, seed):
    """Fly one episode and record everything the figure needs. Returns a dict."""
    sc = ALL[sid]
    spec = sc.public_dict()
    env = FuelLander(sc)
    policy = get_factory(source)(spec)
    obs, info = env.reset(seed=seed)
    policy.reset()
    rec, est = PlanRecorder(policy), GustEstimator(spec)
    est.reset()
    T, S, A, Wd, XI, PH = [], [], [], [], [], []
    t_pol = 0.0
    est.update(info["state"], np.zeros(2))
    for k in range(sc.max_steps + 5):
        t0 = time.perf_counter()
        a = np.asarray(policy.act(obs, info), float).reshape(2)
        t_pol += time.perf_counter() - t0
        rec.after_act(info["t"])
        T.append(info["t"]); S.append(np.array(info["state"])); A.append(a); PH.append(rec.phase())
        obs, _, term, trunc, info = env.step(a)
        Wd.append(env.wind_now)                    # wind applied during this step
        XI.append(est.update(info["state"], a))    # its estimate from the next state
        if term or trunc:
            break
    env.close()
    outcome = info["outcome"] or "timeout"
    return dict(sid=sid, seed=seed, spec=spec, sc=sc, outcome=outcome, score=score_episode(outcome, env.fuel),
                fuel=env.fuel, t=np.array(T), s=np.array(S), a=np.array(A), wind=np.array(Wd), xi=np.array(XI),
                phase=PH, plans=rec.plans, regions=getattr(policy, "debug_regions", None),
                policy_ms=1000 * t_pol / len(T))


# ---------------------------------------------------------------------- reference of the active plan
def _node_times(P):
    if P.get("t_nodes") is not None:
        return np.asarray(P["t_nodes"], float)
    n = len(P["p"])
    dt = P.get("dt") or ((P["tgo"] / (n - 1)) if P.get("tgo") else None)
    return None if dt is None else dt * np.arange(n)


def reference(ep):
    """Planned position of the active plan at every step (NaN without a plan, after its end, or outside the
    POWERED phase when the policy reports a phase)."""
    ref = np.full((len(ep["t"]), 2), np.nan)
    plans = [P for P in ep["plans"]]
    j = -1
    for k, t in enumerate(ep["t"]):
        while j + 1 < len(plans) and plans[j + 1]["t"] <= t + 1e-9:
            j += 1
        if j < 0 or plans[j]["p"] is None or ep["phase"][k] in ("TERMINAL", "TOUCHDOWN"):
            continue
        tn = _node_times(plans[j])
        if tn is None:
            continue
        tau = t - plans[j]["t"]
        if tau <= tn[-1]:
            ref[k] = [np.interp(tau, tn, plans[j]["p"][:, i]) for i in (0, 1)]
    return ref


# ---------------------------------------------------------------------- figure
def plot(ep, name="policy"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.gridspec import GridSpec
    from lander.plot import draw_map
    from lander.style import C, apply
    apply()

    sc, spec = ep["sc"], ep["spec"]
    off = np.array([sc.pad_center, sc.pad[2] + LEG_DOWN])          # pad frame -> world
    t, s = ep["t"], ep["s"]
    hz, terr = obstacles(spec)
    cl_h, cl_t = clearance(s[:, :2], hz), clearance(s[:, :2], terr)
    ref = reference(ep)
    plans = [P for P in ep["plans"] if P["p"] is not None]
    t_plans = [P["t"] for P in plans]
    col = C.get(OUTCOME_COLOR.get(ep["outcome"], "orange"))

    fig = plt.figure(figsize=(16, 9.5))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[1.35, 1], hspace=0.35, wspace=0.12)

    # ---- map
    ax = fig.add_subplot(gs[:, 0])
    draw_map(ax, sc, label=False)
    if ep["regions"]:
        for R in ep["regions"]:
            ax.add_patch(Polygon(np.asarray(R) + off, closed=True, fc=C["blue"], ec=C["blue"], alpha=0.08, lw=0.8))
    for Q in hz:
        ax.add_patch(Polygon(Q + off, closed=True, fill=False, ec=C["red"], ls="--", lw=0.9))
    cmap = plt.get_cmap("Blues")
    for i, P in enumerate(plans):
        c = cmap(0.3 + 0.65 * i / max(len(plans) - 1, 1))
        xy = P["p"] + off
        ax.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=0.8, alpha=0.9, zorder=3)
        ax.plot(xy[0, 0], xy[0, 1], "o", color=c, ms=2.5, zorder=3)
    xy = s[:, :2] + off
    ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.2, zorder=4, label=f"vol ({ep['outcome']})")
    ax.plot(*xy[0], "o", color=col, ms=5, zorder=4)
    for k in range(0, len(t), 50):                 # planning box every second
        ax.add_patch(Rectangle(xy[k] - [R_X, R_Y], 2 * R_X, 2 * R_Y, fill=False, ec=C["ink2"], lw=0.5,
                               alpha=0.45, zorder=2))
    k_min = int(np.argmin(cl_h)) if np.isfinite(cl_h).any() else None
    if k_min is not None:
        ax.add_patch(Rectangle(xy[k_min] - [R_X, R_Y], 2 * R_X, 2 * R_Y, fill=False, ec=C["red"], lw=1.4,
                               zorder=5, label=f"boîte à la clearance min ({cl_h[k_min]:+.2f} m, t={t[k_min]:.2f} s)"))
    ax.plot([], [], "--", color=C["red"], lw=0.9, label="dangers gonflés (R_X, R_Y)")
    ax.plot([], [], color=cmap(0.8), lw=0.8, label=f"plans ({len(plans)})")
    top = max([xy[:, 1].max()] + [P["p"][:, 1].max() + off[1] for P in plans]) + R_Y + 0.3
    ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], top))      # the lander may climb above the map
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("x monde (m)"); ax.set_ylabel("y monde (m)")

    # ---- time series
    axs = [fig.add_subplot(gs[i, 1]) for i in range(6)]
    for a_ in axs[1:]:
        a_.sharex(axs[0])
    for a_ in axs[:-1]:
        a_.tick_params(labelbottom=False)
    for a_ in axs:
        for tp in t_plans:
            a_.axvline(tp, color=C["grid"], lw=0.7, zorder=0)
        if k_min is not None:
            a_.axvline(t[k_min], color=C["red"], lw=0.8, ls=":", zorder=1)

    a_ = axs[0]
    a_.plot(t, s[:, 0], color=C["blue"], label="x"); a_.plot(t, ref[:, 0], "--", color=C["blue"], lw=1)
    a_.plot(t, s[:, 1], color=C["orange"], label="y"); a_.plot(t, ref[:, 1], "--", color=C["orange"], lw=1)
    a_.set_ylabel("position (m)"); _legend(a_)
    a_.text(0.01, 0.92, "trait plein : vol, tirets : plan actif", transform=a_.transAxes, fontsize=7,
            color=C["ink2"], va="top")

    a_ = axs[1]
    a_.plot(t, np.linalg.norm(s[:, :2] - ref, axis=1), color=C["violet"])
    a_.set_ylabel("erreur suivi (m)")
    _shade_phases(a_, t, ep["phase"], C)

    a_ = axs[2]
    finite = [cl_h[np.isfinite(cl_h)], cl_t[np.isfinite(cl_t)]]
    lo = max(-3.0, min(-0.5, *(f.min() for f in finite if f.size)) - 0.2)
    hi = min(8.0, max(3.0, *(f.max() for f in finite if f.size)) + 0.3)
    if np.isfinite(cl_h).any():
        a_.plot(t, cl_h, color=C["red"], label="dangers")
        pm = np.array([clearance(P["p"], hz).min() for P in plans])
        if len(pm):           # minimum clearance along each plan; below the axis: triangle on the bottom edge
            inside = pm >= lo
            a_.plot(np.array(t_plans)[inside], pm[inside], "o", color=C["red"], ms=3, mfc="none",
                    label="min du plan")
            a_.plot(np.array(t_plans)[~inside], np.full((~inside).sum(), lo), "v", color=C["red"], ms=4,
                    label=f"min du plan < {lo:.1f} m" if (~inside).any() else None)
    if np.isfinite(cl_t).any():
        a_.plot(t, cl_t, color=C["ink2"], lw=1, label="terrain (hors pad)")
    margins = [(P["t"], P["margin"]) for P in plans if P.get("margin") is not None]
    if margins:
        a_.step(*zip(*margins), where="post", color=C["green"], lw=1, ls="--", label="marge du plan")
    a_.axhline(0, color=C["red"], lw=0.8)
    a_.set_ylim(lo, hi)
    a_.set_ylabel("clearance (m)"); _legend(a_, ncol=2)

    a_ = axs[3]
    wk = known_wind(spec["wind"], s[:, 1])
    a_.plot(t, ep["wind"], color=C["ink"], lw=0.8, label="vrai")
    a_.plot(t, wk, "--", color=C["ink2"], lw=1, label="connu (moy. + cisaillement)")
    a_.plot(t, wk + ep["xi"], color=C["orange"], lw=1.2, label="connu + rafale estimée")
    w_all = np.r_[ep["wind"], wk]                  # the estimate spikes at ground / rock contact: not in the limits
    a_.set_ylim(w_all.min() - 0.6, w_all.max() + 0.9)
    a_.set_ylabel("vent (m/s²)"); _legend(a_, ncol=3)

    a_ = axs[4]
    a_.plot(t, np.degrees(s[:, 4]), color=C["blue"])
    a_.set_ylabel("θ (°)")

    a_ = axs[5]
    pc = np.array([engine_powers(a) for a in ep["a"]])
    a_.plot(t, pc[:, 0], color=C["orange"], lw=1, label="moteur p")
    a_.plot(t, pc[:, 1], color=C["aqua"], lw=0.8, label="latéral c")
    a_.set_ylim(-1.1, 1.1); a_.set_ylabel("commandes"); a_.set_xlabel("t (s)")
    _legend(a_)

    fig.suptitle(f"{name} — {sc.sid} {sc.name} — seed {ep['seed']} — {ep['outcome']}, score {ep['score']:.1f}, "
                 f"carburant {ep['fuel']:.2f} s, durée {t[-1]:.1f} s, {ep['policy_ms']:.1f} ms/pas",
                 fontsize=11)
    return fig


def _legend(ax, ncol=2):
    ax.legend(loc="upper right", ncol=ncol, fontsize=7, frameon=True, framealpha=0.85, edgecolor="none")


def _shade_phases(ax, t, phases, C):
    colors = dict(TERMINAL=C["yellow"], TOUCHDOWN=C["aqua"])
    k = 0
    while k < len(t):
        ph, j = phases[k], k
        while j < len(t) and phases[j] == ph:
            j += 1
        if ph in colors:
            ax.axvspan(t[k], t[min(j, len(t) - 1)], color=colors[ph], alpha=0.15, lw=0, label=ph)
        k = j
    h, l = ax.get_legend_handles_labels()
    if h:
        d = dict(zip(l, h))
        ax.legend(d.values(), d.keys(), loc="upper right", fontsize=7, ncol=2)


# ---------------------------------------------------------------------- CLI
def _seeds(txt):
    out = []
    for part in txt.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="file.py (with make_policy) or module:function")
    ap.add_argument("--scenarios", default="S01", help="S01,S04 or 'all'")
    ap.add_argument("--seeds", default="0", help="e.g. 3 or 0-9 or 1,4,7")
    ap.add_argument("--failures-only", action="store_true", help="save only the episodes that are not a success")
    ap.add_argument("--out-dir", default=FIG_DIR)
    ap.add_argument("--show", action="store_true", help="open an interactive window instead of saving")
    a = ap.parse_args()
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore")
    sids = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
    name = os.path.splitext(os.path.basename(a.policy.split(":")[-1]))[0]
    os.makedirs(a.out_dir, exist_ok=True)
    for sid in sids:
        for seed in _seeds(a.seeds):
            ep = run(a.policy, sid, seed)
            line = f"{sid} seed {seed:3d}: {ep['outcome']:8s} score {ep['score']:6.1f}  plans {len(ep['plans']):3d}"
            if a.failures_only and ep["outcome"] == "success":
                print(line); continue
            fig = plot(ep, name)
            if a.show:
                plt.show()
            else:
                path = os.path.join(a.out_dir, f"{name}_{sid}_seed{seed}.png")
                fig.savefig(path)
                line += f"  ->  {path}"
            plt.close(fig)
            print(line)


if __name__ == "__main__":
    main()
