from dataclasses import dataclass
import math

import cvxpy as cp
import numpy as np

from controllers import CascadeController
from lander.geometry import closest_point_convex, obstacle_polygons, to_pad
from lander.params import R_Y
from lander.scenarios import W, H


@dataclass
class Polygon:
    points: list[tuple[float, float]]

    #return le vecteur normal vers l'exterieur du polygone du segment entre points[p1] et points[p2]
    def normal(self, p1: int, p2: int) -> tuple[float, float]:
        ax, ay = self.points[p1]
        bx, by = self.points[p2]
        length = math.hypot(bx - ax, by - ay)
        nx, ny = (by - ay) / length, (ax - bx) / length

        # le centre du polygone (convexe) est à l'intérieur : la normale doit pointer à l'opposé
        cx = sum(x for x, _ in self.points) / len(self.points)
        cy = sum(y for _, y in self.points) / len(self.points)
        if nx * (ax - cx) + ny * (ay - cy) < 0:
            nx, ny = -nx, -ny
        return nx, ny

    # déplace chaque arête de |width| : vers l'extérieur si width > 0, vers l'intérieur si width < 0
    def inflate(self, width: float):
        pts: list[tuple[float, float]] = []
        for i in range(len(self.points)):
            i0 = i - 1
            i1 = i
            i2 = (i + 1) % len(self.points)
            px, py = self.points[i1]

            ax, ay = self.normal(i0, i1)    # arête précédente
            bx, by = self.normal(i1, i2)    # arête suivante

            # nouveau sommet = intersection des deux arêtes déplacées
            k = width / (1 + ax * bx + ay * by)
            pts.append((px + k * (ax + bx), py + k * (ay + by)))

        self.points = pts                   # seulement après la boucle


    # étire le polygone selon dir : seules les arêtes dont la normale va dans le sens de dir avancent
    def inflate_dir(self, dir: tuple[float, float]):
        dx, dy = dir
        n = len(self.points)
        pts: list[tuple[float, float]] = []
        for i in range(n):
            ax, ay = self.normal(i - 1, i)          # arête précédente
            bx, by = self.normal(i, (i + 1) % n)    # arête suivante
            d0 = ax * dx + ay * dy
            d2 = bx * dx + by * dy

            px, py = self.points[i]
            stay = min(d0, d2) < -1e-9      # touche une arête opposée à dir
            move = max(d0, d2) > 1e-9       # touche une arête orientée vers dir
            moved = (px + dx, py + dy)

            if stay and move:               # sommet charnière : dédoublé, dans l'ordre du parcours
                pts += [moved, (px, py)] if d0 > d2 else [(px, py), moved]
            elif move:
                pts.append(moved)
            else:
                pts.append((px, py))

        self.points = pts                   # seulement après la boucle

    # forme {p : A p <= b}, une ligne par arête (pour la SOCP)
    def halfplanes(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.points)
        A = np.array([self.normal(i, (i + 1) % n) for i in range(n)])
        b = np.sum(A * np.array(self.points), axis=1)
        return A, b

    # p : un point (2,) ou un tableau de points (k, 2)
    def contains(self, p, tol: float = 1e-7):
        A, b = self.halfplanes()
        return np.all(np.asarray(p) @ A.T <= b + tol, axis=-1)

    # nouveau polygone = partie qui vérifie a . p <= b (None si elle est vide)
    def clip(self, a, b: float) -> "Polygon | None":
        pts: list[tuple[float, float]] = []
        n = len(self.points)
        for i in range(n):
            p, q = self.points[i], self.points[(i + 1) % n]
            sp = a[0] * p[0] + a[1] * p[1] - b
            sq = a[0] * q[0] + a[1] * q[1] - b
            if sp <= 0:
                pts.append(p)
            if sp * sq < 0:                 # l'arête traverse la droite : on ajoute l'intersection
                t = sp / (sp - sq)
                pts.append((float(p[0] + t * (q[0] - p[0])), float(p[1] + t * (q[1] - p[1]))))

        # supprime les sommets confondus (normal() exige des arêtes de longueur non nulle)
        pts = [p for i, p in enumerate(pts) if math.dist(p, pts[i - 1]) > 1e-7]
        return Polygon(pts) if len(pts) >= 3 else None


@dataclass
class Map:
    bounds: tuple[tuple[float, float], tuple[float, float]]
    obstacles: list[Polygon]


def load_map(spec, margin_x=1.15, margin_y=R_Y) -> Map:
    obstacles = []
    for poly in obstacle_polygons(spec)[0]:
        o = Polygon([(float(x), float(y)) for x, y in poly])
        for d in [(margin_x, 0), (-margin_x, 0), (0, margin_y), (0, -margin_y)]:
            o.inflate_dir(d)
        obstacles.append(o)

    lo, hi = to_pad(spec, [[0, 0], [W, H]])
    bounds = ((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1])))
    return Map(bounds, obstacles)


# ellipse {C u + d : ||u|| <= 1} de volume maximal inscrite dans le polygone (problème convexe)
def max_ellipse(poly: Polygon) -> tuple[np.ndarray, np.ndarray] | None:
    C = cp.Variable((2, 2), PSD=True)
    d = cp.Variable(2)
    A, b = poly.halfplanes()
    cons = [cp.norm(C @ a) + a @ d <= bi for a, bi in zip(A, b)]
    cp.Problem(cp.Maximize(cp.log_det(C)), cons).solve(solver=cp.CLARABEL)
    C_opt, d_opt = C.value, d.value
    if C_opt is None or d_opt is None:      # le solveur n'a pas trouvé de solution
        return None
    return np.asarray(C_opt), np.asarray(d_opt)


# coupe les bounds par un demi-plan séparateur par obstacle, le plus proche d'abord (au sens de l'ellipse C, d)
def separate(carte: Map, C: np.ndarray, d: np.ndarray) -> Polygon | None:
    (xmin, ymin), (xmax, ymax) = carte.bounds
    region = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    Ci = np.linalg.inv(C)

    closest = []
    for obs in carte.obstacles:
        P = np.array(obs.points)
        # dans le repère (p - d) @ Ci, l'ellipse devient le disque unité centré en 0
        q, dist, n = closest_point_convex((P - d) @ Ci.T, np.zeros(2))
        closest.append((dist, q, n, P))
    closest.sort(key=lambda c: c[0])

    planes = []
    for dist, q, n, P in closest:
        if any((P @ a >= b - 1e-7).all() for a, b in planes):
            continue                                # obstacle déjà exclu par un plan précédent
        if dist <= 1e-9:
            return None                             # le centre touche l'obstacle
        a = -Ci.T @ n
        a /= np.linalg.norm(a)
        b = a @ (C @ q + d)
        planes.append((a, b))
        region = region.clip(a, b)
        if region is None:
            return None
    return region


# plus grande région convexe libre trouvée autour du point seed (algorithme IRIS)
def grow_region(carte: Map, seed: np.ndarray, iters: int = 4) -> Polygon | None:
    C, d = 1e-2 * np.eye(2), seed
    best = None
    for _ in range(iters):
        region = separate(carte, C, d)
        if region is None or not region.contains(seed):
            break                                   # la région doit toujours contenir la graine
        best = region
        ellipse = max_ellipse(region)
        if ellipse is None:
            break
        C, d = ellipse
    return best


# liste de polygones convexes libres (qui se chevauchent) couvrant la map autour des obstacles
def decompose(carte: Map, seeds=(), max_regions: int = 12, step: float = 0.25) -> list[Polygon]:
    # grille de points libres : sert à choisir les graines et à suivre la couverture
    (xmin, ymin), (xmax, ymax) = carte.bounds
    grid = np.array([(x, y) for x in np.arange(xmin + step / 2, xmax, step)
                            for y in np.arange(ymin + step / 2, ymax, step)])
    obstacles = [np.array(o.points) for o in carte.obstacles]
    clearance = np.array([min(closest_point_convex(P, p)[1] for P in obstacles) for p in grid])
    grid, clearance = grid[clearance > 0.05], clearance[clearance > 0.05]

    regions: list[Polygon] = []
    covered = np.zeros(len(grid), bool)
    tried = np.zeros(len(grid), bool)

    def add(seed):
        region = grow_region(carte, seed)
        if region is not None:
            regions.append(region)
            covered[:] |= region.contains(grid)

    # d'abord les points imposés (départ, porte d'approche)
    for seed in seeds:
        seed = np.asarray(seed, float)
        if not any(r.contains(seed) for r in regions):
            add(seed)

    # puis le point non couvert le plus éloigné des obstacles
    while len(regions) < max_regions and (~covered & ~tried).any():
        i = np.argmax(np.where(~covered & ~tried, clearance, -1.0))
        tried[i] = True
        add(grid[i])
    return regions
