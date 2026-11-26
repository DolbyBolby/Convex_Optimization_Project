from dataclasses import dataclass
import math

import numpy as np
import cvxpy as cp

from lander.geometry import obstacle_polygons, to_pad
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

    def area(self) -> float:
        n = len(self.points)
        total = 0.0
        for i in range(n):
            (px, py), (qx, qy) = self.points[i], self.points[(i + 1) % n]
            total += px * qy - qx * py
        return abs(total) / 2

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

    # polygone qui couvre toute la map, éventuellement restreint à xa <= x <= xb
    def rectangle(self, xa: float | None = None, xb: float | None = None) -> Polygon:
        (xmin, ymin), (xmax, ymax) = self.bounds
        xa = xmin if xa is None else xa
        xb = xmax if xb is None else xb
        return Polygon([(xa, ymin), (xb, ymin), (xb, ymax), (xa, ymax)])


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


# ---------------------------------------------------------------------- découpage de l'espace libre
Line = tuple[float, float]                  # droite y = m x + q, notée (m, q)
HalfPlanes = list[tuple[np.ndarray, float]]  # liste de (a, b) : a . p <= b


@dataclass
class Trapezoid:
    """Morceau d'espace libre entre deux verticales, un sol et un plafond."""
    xa: float
    xb: float
    floor: Line
    ceiling: Line


def clip_all(poly: Polygon, planes: HalfPlanes) -> Polygon | None:
    region: Polygon | None = poly
    for a, b in planes:
        region = region.clip(a, b)
        if region is None:
            break
    return region


def edges(poly: Polygon):
    n = len(poly.points)
    return [(poly.points[i], poly.points[(i + 1) % n]) for i in range(n)]


# abscisses des sommets des obstacles et des croisements d'arêtes (entre elles et avec le haut / bas de la map)
def vertical_cuts(carte: Map) -> list[float]:
    (xmin, ymin), (xmax, ymax) = carte.bounds
    segments = [(k, e) for k, o in enumerate(carte.obstacles) for e in edges(o)]
    segments += [(-1, ((xmin, ymin), (xmax, ymin))), (-2, ((xmin, ymax), (xmax, ymax)))]
    xs = [xmin, xmax] + [x for o in carte.obstacles for x, _ in o.points]

    for i, (k, (a, b)) in enumerate(segments):
        for other, (c, d) in segments[i + 1:]:
            if k == other:
                continue
            rx, ry = b[0] - a[0], b[1] - a[1]
            sx, sy = d[0] - c[0], d[1] - c[1]
            den = rx * sy - ry * sx
            if abs(den) < 1e-12:                    # segments parallèles
                continue
            t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / den
            u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / den
            if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
                xs.append(a[0] + t * rx)

    cuts: list[float] = []
    for x in sorted(x for x in xs if xmin <= x <= xmax):
        if not cuts or x - cuts[-1] > 1e-6:
            cuts.append(x)
    return cuts


# pour chaque tranche verticale, les trapèzes libres (décomposition exacte)
def free_trapezoids(carte: Map) -> list[list[Trapezoid]]:
    (_, ymin), (_, ymax) = carte.bounds
    cuts = vertical_cuts(carte)
    slices = []
    for xa, xb in zip(cuts[:-1], cuts[1:]):
        xm = (xa + xb) / 2                          # aucune arête ne se croise à l'intérieur d'une tranche

        # intervalle occupé par chaque obstacle : ((hauteur en xm, droite) en bas, idem en haut)
        occupied = []
        for obstacle in carte.obstacles:
            hits = []
            for a, b in edges(obstacle):
                if min(a[0], b[0]) < xm < max(a[0], b[0]):
                    m = (b[1] - a[1]) / (b[0] - a[0])
                    q = a[1] - m * a[0]
                    hits.append((m * xm + q, (m, q)))
            if hits:
                occupied.append((min(hits), max(hits)))
        occupied.sort()

        # les trous entre les intervalles occupés sont libres
        free: list[Trapezoid] = []
        top, floor = ymin, (0.0, ymin)
        for low, high in occupied:
            if low[0] > top + 1e-9 and top < ymax:
                ceiling = low[1] if low[0] < ymax else (0.0, ymax)
                free.append(Trapezoid(xa, xb, floor, ceiling))
            if high[0] > top:
                top, floor = high
        if top < ymax - 1e-9:
            free.append(Trapezoid(xa, xb, floor, (0.0, ymax)))
        slices.append(free)
    return slices


# une région convexe libre pour chaque suite de trapèzes voisins
def candidate_regions(carte: Map) -> list[Polygon]:
    slices = free_trapezoids(carte)
    regions = []
    for first, free in enumerate(slices):
        for trapezoid in free:
            stack = [(first, [trapezoid])]
            while stack:
                k, chain = stack.pop()
                # au-dessus de TOUS les sols et sous TOUS les plafonds : convexe et libre
                planes: HalfPlanes = []
                for t in chain:
                    planes.append((np.array([t.floor[0], -1.0]), -t.floor[1]))
                    planes.append((np.array([-t.ceiling[0], 1.0]), t.ceiling[1]))
                region = clip_all(carte.rectangle(chain[0].xa, chain[-1].xb), planes)

                # on arrête la suite si la région n'atteint plus sa dernière tranche
                if region is None or max(x for x, _ in region.points) <= chain[-1].xa + 1e-6:
                    continue
                regions.append(region)

                if k + 1 < len(slices):
                    last, x = chain[-1], chain[-1].xb
                    for t in slices[k + 1]:         # voisins : ouverture commune sur la verticale x
                        low = max(m * x + q for m, q in (last.floor, t.floor))
                        high = min(m * x + q for m, q in (last.ceiling, t.ceiling))
                        if high - low > 1e-3:
                            stack.append((k + 1, chain + [t]))
    return regions


# vrai si les deux polygones ont une surface commune
def overlap(poly: Polygon, other: Polygon, min_area: float = 1e-2) -> bool:
    common = clip_all(poly, list(zip(*other.halfplanes())))
    return common is not None and common.area() > min_area


# agrandit la région en supprimant les faces inutiles pour rester hors des obstacles
def widen(region: Polygon, carte: Map) -> Polygon:
    planes: HalfPlanes = list(zip(*region.halfplanes()))
    j = 0
    while j < len(planes):
        bigger = clip_all(carte.rectangle(), planes[:j] + planes[j + 1:])
        assert bigger is not None                   # retirer une face ne peut qu'agrandir
        # marge de 1e-6 : toucher le bord d'un obstacle est permis
        shrunk = (clip_all(bigger, [(a, b - 1e-6) for a, b in zip(*o.halfplanes())]) for o in carte.obstacles)
        if any(s is not None and s.area() > 1e-8 for s in shrunk):
            j += 1                                  # face nécessaire
        else:
            del planes[j]
    result = clip_all(carte.rectangle(), planes)
    assert result is not None
    return result


# groupes de régions reliées entre elles par des chevauchements
def components(regions: list[Polygon]) -> list[list[Polygon]]:
    left = list(regions)
    groups = []
    while left:
        group = [left.pop()]
        for region in group:                        # la liste grandit pendant le parcours
            linked = [r for r in left if overlap(region, r)]
            group += linked
            left = [r for r in left if r not in linked]
        groups.append(group)
    return groups


# petite liste de polygones convexes libres (qui se chevauchent) couvrant la map autour des obstacles
def decompose(carte: Map, seeds=(), max_regions: int = 12, coverage: float = 0.999,
              step: float = 0.15) -> list[Polygon]:
    # candidates élargies, sans doublons
    unique = {}
    for region in candidate_regions(carte):
        region = widen(region, carte)
        key = tuple(sorted((round(x, 3), round(y, 3)) for x, y in region.points))
        unique[key] = region
    candidates = list(unique.values())

    # grille de points libres pour mesurer la couverture
    (xmin, ymin), (xmax, ymax) = carte.bounds
    grid = np.array([(x, y) for x in np.arange(xmin + step / 2, xmax, step)
                            for y in np.arange(ymin + step / 2, ymax, step)])
    grid = grid[~np.any([o.contains(grid, 1e-9) for o in carte.obstacles], axis=0)]
    covers = np.array([r.contains(grid) for r in candidates])

    chosen: list[Polygon] = []
    covered = np.zeros(len(grid), bool)

    def choose(i):
        chosen.append(candidates[i])
        covered[:] |= covers[i]

    # d'abord les points imposés (départ, porte d'approche) : la plus grande région qui les contient
    for seed in seeds:
        seed = np.asarray(seed, float)
        if not any(r.contains(seed) for r in chosen):
            inside = [i for i, r in enumerate(candidates) if r.contains(seed)]
            if inside:
                choose(max(inside, key=lambda i: candidates[i].area()))

    # couverture gloutonne : la région qui ajoute le plus de points non couverts
    while len(chosen) < max_regions and covered.mean() < coverage:
        gain = (covers & ~covered).sum(axis=1)
        best = int(np.argmax(gain))
        if gain[best] == 0:
            break
        choose(best)

    # régions de liaison tant que le graphe n'est pas connexe (peut dépasser max_regions)
    while len(groups := components(chosen)) > 1:
        def links(r):
            return sum(any(overlap(r, other) for other in group) for group in groups)
        best_link = max(candidates, key=lambda r: (links(r), r.area()))
        if links(best_link) < 2:
            break                                   # espace libre réellement non connexe
        chosen.append(best_link)
    return chosen


# toutes les suites de régions qui vont du départ à la porte
def simple_paths(regions: list[Polygon], start, gate) -> list[list[int]]:
    n = len(regions)
    neighbours = [[j for j in range(n) if j != i and overlap(regions[i], regions[j])] for i in range(n)]
    starts = [i for i in range(n) if regions[i].contains(start)]
    paths: list[list[int]] = []

    def explore(path):
        if regions[path[-1]].contains(gate):
            paths.append(path)
            return
        for j in neighbours[path[-1]]:
            if j not in path and j not in starts:
                explore(path + [j])

    for i in starts:
        explore([i])
    return paths


# ligne brisée la plus courte le long d'un chemin : (longueur, points de passage)
def shortest_polyline(regions: list[Polygon], path: list[int], start, gate):
    points = cp.Variable((len(path) + 1, 2))
    constraints = [points[0] == start, points[-1] == gate]
    for k in range(1, len(path)):
        for i in (path[k - 1], path[k]):          # point dans l'intersection des deux régions
            A, b = regions[i].halfplanes()
            constraints.append(A @ points[k] <= b)
    length = cp.sum(cp.norm(points[1:] - points[:-1], axis=1))
    problem = cp.Problem(cp.Minimize(length), constraints)
    problem.solve(solver=cp.CLARABEL)
    solution = points.value
    if solution is None:
        return np.inf, None
    pts = np.asarray(solution)
    # longueur recalculée à partir des points (problem.value a un type trop vague pour float())
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), pts


# les meilleurs chemins : liste de (longueur, indices des régions, points de passage)
# classés par coût = longueur + region_penalty (en mètres) par région traversée
def find_paths(regions: list[Polygon], start, gate, keep: int = 3, region_penalty: float = 0.25):
    found = []
    for path in simple_paths(regions, start, gate):
        length, points = shortest_polyline(regions, path, start, gate)
        if points is not None:
            found.append((length, path, points))
    found.sort(key=lambda f: f[0] + region_penalty * len(f[1]))
    return found[:keep]
