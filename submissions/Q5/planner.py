"""Trajectoire optimale le long d'un chemin de régions : la SOCP de guidage + les contraintes A r <= b.

Chaque intervalle [r_k, r_{k+1}] appartient à une région du chemin. On optimise la poussée avec la SOCP, et on
cherche autour d'elle les nœuds de changement de région et le temps de vol.

Vent (wind_robust.py) : le vent connu et la décroissance de la rafale estimée xi0 sont dans la dynamique. Avec
ctrl.use_risk, on ajoute le coût de risque : les contraintes dures restent les faces des régions, le risque porte sur
les vrais obstacles, linéarisés autour d'une trajectoire de référence.
"""
from dataclasses import dataclass
import itertools

import numpy as np

from lander.wind import known_wind
from lcvx import LanderSOCP, LanderSolution
from gcs import Polygon
from wind_robust import RobustLanderSOCP, expected_gust_drift


@dataclass
class PathPlan:
    p: np.ndarray               # positions prévues (N+1, 2), repère du pad
    fuel: float                 # carburant prévu (kg)
    tgo: float                  # temps de vol prévu jusqu'à la porte (s)
    switches: list[int]         # nœuds de changement de région
    solution: LanderSolution
    cost: float                 # critère de choix : carburant (kg), ou coût total de la SOCP avec risque


class PathPlanner:
    def __init__(self, ctrl, regions: list[Polygon], wind: dict, theta_max_deg: float = 25.0):
        """ctrl : la cascade (paramètres, porte, poussée réservée, murs et tube) ; wind : spec["wind"]."""
        self.regions, self.N = regions, ctrl.N
        self.faces = max(len(r.points) for r in regions)
        self.allow_off = ctrl.a_min(ctrl.p.m_dry) >= -ctrl.g      # basse gravité : moteur coupable
        # vent connu : partie constante ici, la partie proportionnelle à y est dans l'argument shear
        self.w0 = float(known_wind(wind, 0.0))
        self.gust_tau = wind["gust_tau"]
        self.xi0 = 0.0                              # rafale estimée au moment du plan (mise à jour par la politique)
        self.hazards, self.tube = ctrl.walls, ctrl.tube
        self.use_risk = ctrl.use_risk
        # glide_from=N : pas de glide slope, les régions protègent déjà du sol
        options = dict(mode="fuel", terminal="gate", theta_max_deg=theta_max_deg, glide_from=self.N,
                       thrust_frac=ctrl.derate, y_gate=ctrl.y_gate, v_gate=ctrl.v_gate, x_tol=ctrl.x_tol,
                       shear=wind["shear"])
        self.socp = (RobustLanderSOCP(ctrl.p, self.N, ctrl.g, n_walls=self.faces, n_risk=self.hazards.n_walls,
                                      w_haz=ctrl.G.w_haz, **options) if self.use_risk
                     else LanderSOCP(ctrl.p, self.N, ctrl.g, n_obs=self.faces, **options))

    def solve(self, path: list[int], switches: list[int], tgo: float, x0, m0: float, certify: bool = True,
              max_slack: float = 1e-6, ref: np.ndarray | None = None) -> LanderSolution | None:
        """Une SOCP. None si infaisable, hors des régions (slack > max_slack) ou non certifiée (si certify).
        ref (N+1, 2) : trajectoire autour de laquelle les obstacles du coût de risque sont linéarisés."""
        # région de l'intervalle k ; LanderSOCP l'impose aux nœuds k et k+1 sous la forme H r >= b
        H = np.zeros((self.faces, self.N + 1, 2))
        b = -np.ones((self.faces, self.N + 1))                    # lignes inutilisées : 0 >= -1
        for k in range(self.N + 1):
            A, bb = self.regions[path[sum(k >= s for s in switches)]].halfplanes()
            H[:len(bb), k] = -A
            b[:len(bb), k] = -bb
        if isinstance(self.socp, RobustLanderSOCP):
            risk_walls, sigma = None, None
            if ref is not None:
                risk_walls = self.hazards(ref)
                sigma = self.tube.sigma_dir(risk_walls[0], np.arange(self.N + 1) * tgo / self.N)
            sol = self.socp.solve(x0, tgo, m0, wind_const=self.w0, xi0=self.xi0, gust_tau=self.gust_tau,
                                  walls=(H, b), sigma=sigma, risk_walls=risk_walls, allow_off=self.allow_off)
        else:
            # accélération moyenne de la rafale attendue sur chaque intervalle (vitesse exacte)
            dt = tgo / self.N
            dv, _ = expected_gust_drift(self.xi0, self.gust_tau, np.arange(self.N) * dt, dt)
            sol = self.socp.solve(x0, tgo, m0, wind=self.w0 + dv / dt, halfspaces=list(zip(H, b)),
                                  allow_off=self.allow_off)
        if sol is None or sol.slack > max_slack:
            return None
        # certificat sans perte ; plancher de 1 m/s^2 pour les nœuds moteur coupé (sigma presque nul)
        if certify and np.max(sol.relaxation_gap / np.maximum(sol.sigma, 1.0)) > 1e-3:
            return None
        return sol

    def polyline_reference(self, points: np.ndarray, switches: list[int], x0) -> np.ndarray:
        """Référence (N+1, 2) le long de la ligne brisée : le point de passage i est atteint au nœud switches[i-1].
        Le début est décalé pour partir de la position mesurée."""
        k = np.arange(self.N + 1)
        nodes = [0, *switches, self.N]
        ref = np.column_stack([np.interp(k, nodes, points[:, i]) for i in (0, 1)])
        return ref + np.outer(np.linspace(1.0, 0.0, self.N + 1), np.asarray(x0[:2]) - ref[0])

    def optimize(self, path: list[int], points: np.ndarray, x0, m0: float) -> PathPlan | None:
        """Meilleur plan le long du chemin ; points = ligne brisée la plus courte (pour initialiser)."""
        N = self.N
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        base = [int(round(f * N)) for f in np.cumsum(lengths)[:-1] / lengths.sum()]
        tried: set = set()
        found: list[PathPlan] = []      # plans successivement meilleurs ; le dernier est le meilleur

        def attempt(switches, tgo) -> bool:
            """Essaie (switches, tgo) ; True si c'est le nouveau meilleur plan."""
            switches = [int(np.clip(s, 1, N - 1)) for s in switches]
            key = (tuple(switches), round(float(tgo), 3))
            if switches != sorted(switches) or key in tried or tgo < 0.6:
                return False
            tried.add(key)
            ref = self.polyline_reference(points, switches, x0) if self.use_risk else None
            sol = self.solve(path, switches, float(tgo), x0, m0, ref=ref)
            if sol is None:
                return False
            cost = getattr(sol, "cost", sol.fuel)
            if found and cost >= found[-1].cost - 1e-5:
                return False
            found.append(PathPlan(sol.p, sol.fuel, float(tgo), switches, sol, cost))
            return True

        # 1. grille grossière
        for offsets in itertools.product((-4, 0, 4), repeat=len(base)):
            for tgo in np.arange(1.5, 9.01, 0.5):
                attempt([s + d for s, d in zip(base, offsets)], tgo)
        if not found:
            return None

        # 2. raffinement local : un seul paramètre modifié à la fois
        for dk, dt in ((2, 0.2), (1, 0.1), (1, 0.05)):
            improved = True
            while improved:
                improved = False
                for sign in (-1, 1):
                    improved |= attempt(found[-1].switches, found[-1].tgo + sign * dt)
                    for i in range(len(base)):
                        switches = list(found[-1].switches)
                        switches[i] += sign * dk
                        improved |= attempt(switches, found[-1].tgo)
        return found[-1]
