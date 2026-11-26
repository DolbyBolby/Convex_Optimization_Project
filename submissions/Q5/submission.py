from controllers import CascadeController
from lander.geometry import to_pad
from lander.wind import known_wind
import numpy as np

from gcs import decompose, find_paths, load_map
from planner import PathPlanner


class GCSPolicy(CascadeController):
    def __init__(self, spec):
        super().__init__(gravity=spec["gravity"], spec=spec)
        self.spec = spec
        start = to_pad(spec, spec["start"][:2])
        gate = self.gate = np.array([0.0, self.y_gate])
        carte = load_map(spec)
        self.regions = decompose(carte, seeds=[start, gate])
        self.debug_regions = [r.points for r in self.regions]
        self.debug_obstacles = [o.points for o in carte.obstacles]
        self.paths = find_paths(self.regions, start, gate, keep=5)
        self.debug_paths = self.paths

        # trajectoires optimisées au départ (moyen) sur les 3 chemins les plus courts (pour les figures)
        planner = self.planner = PathPlanner(self, self.regions, spec["wind"])
        x0 = np.r_[start, spec["start"][2:]]
        shortest = sorted(self.paths, key=lambda p: p[0])[:3]
        self.path_plans = {tuple(path): planner.optimize(path, points, x0, self.p.m_wet)
                           for _, path, points in shortest}
        self.debug_path_plans = self.path_plans

    def reset(self):
        super().reset()
        self.route = None           # (indices des régions, nœuds de changement) du plan courant

    # ------------------------------------------------------------------ planification en vol
    def _search(self, s, refine=6):
        """Nouveau plan depuis l'état mesuré : le plus économe parmi les 2 meilleurs chemins."""
        # si la position mesurée n'est dans aucune région, on reprend les chemins du départ moyen
        paths = find_paths(self.regions, s[:2], self.gate, keep=2) or self.paths[:2]
        found = [(plan, path) for _, path, points in paths
                 if (plan := self.planner.optimize(path, points, s[:4], s[6])) is not None]
        if not found:
            return None
        plan, path = min(found, key=lambda f: f[0].fuel)
        # marge de temps pour les rotations (comme dans la cascade)
        sol = self.planner.solve(path, plan.switches, self.tgo_margin * plan.tgo, s[:4], s[6])
        return self._adopt(path, plan.switches, plan.solution if sol is None else sol)

    def _replan(self, s):
        """Horizon décroissant le long du même chemin ; nouvelle recherche seulement en dernier recours."""
        if self.plan is None or self.route is None:
            return self._search(s)
        elapsed = self.k_since / 50
        remaining = self.plan["tgo"] - elapsed
        if remaining <= 0.6:
            return self._search(s)
        path, switches = self.route
        times = [k * self.plan["dt"] - elapsed for k in switches]    # temps restant avant chaque changement
        while times and times[0] <= 0 and self.regions[path[1]].contains(s[:2]):
            path, times = path[1:], times[1:]                        # région quittée

        # d'abord les contraintes strictes, puis souples (le vent peut avoir poussé hors des régions)
        for max_slack, factors, offsets in ((1e-6, (1.0, 1.15, 1.3), (0, -2, 2, -4, 4)),
                                            (np.inf, (1.0, 1.3), (0,))):
            for factor in factors:
                for offset in offsets:
                    sw = [int(np.clip(round(t / remaining * self.N) + offset, 1, self.N - 1)) for t in times]
                    sol = self.planner.solve(path, sw, factor * remaining, s[:4], s[6], certify=False,
                                             max_slack=max_slack)
                    if sol is not None:
                        return self._adopt(path, sw, sol)
                    if not sw:                                       # une seule région : pas de décalage
                        break
        return self._search(s)

    def _terminal(self, s):
            v_ref = np.array([np.clip(-0.3 * s[0], -0.3, 0.3), -self.v_gate])
            a_cmd = np.array([0.0, -self.g]) + np.array([2.0, 3.0]) * (v_ref - s[2:4])
            a_cmd[0] -= known_wind(self.spec["wind"], s[1])    # seul ajout : annule le vent connu
            engine = self.a_min(s[6]) < -self.g or s[3] < -self.v_gate
            return self._to_action(s, a_cmd, engine=engine)

    def _adopt(self, path, switches, sol):
        self.route = (path, switches)
        return dict(kind="gcs", tgo=sol.tgo, dt=sol.tgo / self.N, p=sol.p, v=sol.v, a=sol.u, m=np.exp(sol.z),
                    sol=sol)

    def _reference(self, s):
        """Référence de la cascade, plus le vent connu que le plan prend en compte."""
        i, p_ref, v_ref, a = super()._reference(s)
        plan = self.plan
        assert plan is not None             # _reference n'est appelée que s'il y a un plan (voir _step)
        dt = plan["dt"]
        tau = min(self.k_since / 50 - i * dt, dt)
        wind = np.array([known_wind(self.spec["wind"], plan["p"][i, 1]), 0.0])
        return i, p_ref + 0.5 * tau * tau * wind, v_ref + tau * wind, a


def make_policy(spec):
    return GCSPolicy(spec)
