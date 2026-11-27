"""Robustification contre le vent stochastique (Question 5.2, Phase 3).

Implémentation de la spécification « Guida Robusta in Presenza di Vento Stocastico » (doc convex_opt_wind_gesture).
Les numéros d'équations (1)-(59) cités dans les commentaires sont ceux de ce document.

==================================================================================================================
RÉSUMÉ DE LA DÉMARCHE
==================================================================================================================

0. Point de départ
------------------
On part du contrôleur en cascade de la Question 4 (lander/controllers.py : planificateur SOCP sans perte rejoué
toutes les 0.5 s + loi PD de suivi à 50 Hz + boucle d'attitude). Ce contrôleur ignore le vent : la loi PD subit
toute la dérive, l'erreur de suivi grandit, et près d'une roche ou d'une paroi de canyon cela finit en collision.
Le vent horizontal du simulateur (lander/env.py, équation (4) de l'énoncé) est

        w(t, y) = w_bar + (s/10) (y + l)  +  xi(t),        xi = processus d'Ornstein-Uhlenbeck (sigma_xi, tau),
                  \\_______ connu ________/    \\_ inconnu _/

Convention de la partie déterministe w_det (ATTENTION : elle diffère entre les deux documents)
  * Énoncé (Questions.pdf, éq. (4)) et simulateur : w_det(y) = w_bar + (s/10) (y + l), où y est la hauteur du centre
    du lander dans le repère du pad (y = 0 posé sur le pad), l = LEG_DOWN = 0.6 m, et s la hausse du vent sur 10 m.
    y + l est donc la hauteur du centre AU-DESSUS DU SOL du pad. Le simulateur calcule la même chose en coordonnées
    du monde : shear (y_monde - y_pad) / 10, avec y_monde - y_pad = y + l (lander/env.py, _wind_accel).
  * Doc italien (éq. (1) et (6)) : w_det(y) = w_bar + alpha_w y, sans origine précisée pour la « quota » y.
    Correspondance : alpha_w = s/10 et w_det(y) = (w_bar + alpha_w l) + alpha_w y. Appliquée littéralement avec le y
    du repère du pad, elle oublie le terme constant alpha_w l = s l / 10 : 0.12 m/s^2 dans S05 (s = 2), 0.06 m/s^2 dans
    S10 (s = 1), 0 dans les huit autres scénarios (s = 0). L'estimateur prendrait ce biais pour une rafale permanente.
  * Ce code suit la convention de l'énoncé : partout, w_det est calculé par known_wind(wind, y) de lander/wind.py,
    qui vaut w_bar + s (y + LEG_DOWN) / 10. Dans le SOCP, il est séparé en une partie constante
    known_wind(wind, 0) = w_bar + s l / 10 (termes de dérive dvx, dpx) et une partie s y_k / 10 (argument `shear` de
    LanderSOCP), dont la somme redonne exactement w_bar + (s/10)(y_k + l). Dans le feedforward, seule la différence
    w_det(y) - w_det(y_ref) intervient : le terme constant s'y annule et la convention ne joue pas.
  * Vérifié sur S05 avec rafales coupées : écart maximal entre known_wind et le vent appliqué par le simulateur
    5.6e-17 m/s^2 ; avec w_bar + alpha_w y (y du repère du pad), 0.120 m/s^2 = s l / 10.
  Autres différences de notation, sans effet sur le code : le doc italien note la gravité (0, -g0) avec g0 > 0
  (l'énoncé : (0, g_y) avec g_y < 0) ; son Gamma_k borne ||u_k|| en accélération, c'est le sigma_k = Gamma / m de
  l'énoncé ; ses bornes constantes a_min <= Gamma_k <= a_max sont ici les bornes de Taylor de la Question 1.3.

Plutôt que de traiter tout le vent comme une perturbation aveugle (et de gonfler uniformément les obstacles),
on le DÉCOMPOSE en trois parties, et chacune est confiée à la couche de la cascade qui la gère le mieux :

1. Phase 1 : vent déterministe dans la planification (boucle lente, RobustLanderSOCP)
------------------------------------------------------------------------------------
Tout ce qui est connu ou prévisible EN MOYENNE au moment du replanning entre dans les équations du mouvement du SOCP :
  * vent moyen w_bar et cisaillement s/10 (y + l) : le terme de cisaillement est affine en y_k, donc la
    dynamique reste linéaire (argument `shear` de LanderSOCP) ;
  * la rafale estimée xi_hat_0 à l'instant t_plan : un processus OU décroît en moyenne comme
    E[xi(t_plan + s) | xi_hat_0] = xi_hat_0 e^{-s/tau}. On intègre EXACTEMENT ce profil sur chaque intervalle
    [t_k, t_k+1) (équations (4)-(5)) ; on obtient deux scalaires connus, Delta_v_xi_k et Delta_x_xi_k, que l'on
    place dans les termes de dérive `drift_v`, `drift_r` déjà prévus par LanderSOCP (paramètres dvx, dpx).
La dynamique discrète (6)-(9) reste donc linéaire et exacte (ZOH) : le SOCP incline le lander contre le vent
À L'AVANCE, au lieu de dépenser des marges de sécurité sur une poussée dont on connaît déjà le sens et l'intensité.
La référence interpolée entre deux nœuds (`_reference`) utilise exactement le même modèle de vent, sinon la loi PD
combattrait le vent que le plan a déjà pris en compte.

2. Phase 2 : compensation par anticipation (feedforward) à 50 Hz (OUGustEstimator + _track)
--------------------------------------------------------------------------------------------
Entre deux replannings, de nouvelles rafales arrivent. À chaque pas :
  * estimation de la rafale (10)-(11) : on mesure l'accélération latérale, on lui retire l'accélération des
    moteurs (avec la commande réellement appliquée) et le vent connu, puis on filtre ce résidu avec un filtre
    exponentiel qui respecte la décroissance OU : xi_hat_j = (1 - lambda) phi xi_hat_{j-1} + lambda xi_raw_j ;
  * on calcule la SEULE SURPRISE par rapport à ce que le plan a déjà anticipé (12) :
    Delta_xi(t) = xi_hat(t) - xi_hat_0 e^{-(t - t_plan)/tau} (sinon la rafale serait compensée deux fois) ;
  * loi de commande (13)-(14) : a_cmd = u_plan + Kp (r_ref - r) + Kd (v_ref - v) - [gamma_ff Delta_xi + Delta_w_det, 0],
    où Delta_w_det = w_det(y) - w_det(y_ref) corrige l'écart de cisaillement dû à l'écart d'altitude.
Le gain gamma_ff < 1 (0.7) évite d'amplifier le bruit de l'estimateur. Extension naturelle : dans la descente
terminale et dans le mode de secours (sans plan), xi_plan = 0 et w_plan = 0, donc on compense tout le vent estimé.
Sous la porte il n'y a plus de plan ni de tube : la descente terminale reçoit aussi un maintien latéral plus ferme
(term_gains), car c'est là qu'une dérive résiduelle touche les parois d'un couloir étroit (S03, S05, S08).

3. Phase 3 : coût de risque convexe pour élargir les obstacles (WindTube + RobustLanderSOCP)
------------------------------------------------------------------------------------------
Les phases 1 et 2 réduisent la dérive à la source, mais il reste une dérive résiduelle (rafales futures, retard de
l'estimateur, temps de rotation du corps, couple de turbulence). La phase 3 la quantifie et la traduit en pénalité.
  3.1 Tube d'incertitude (15)-(17), classe WindTube. L'état d'erreur horizontal zeta = (e_x, e_vx, delta_xi) suit,
      sous la loi PD (on prend gamma_ff = 0, choix prudent), le système linéaire zeta_{j+1} = A_cl zeta_j + b_w eps_j.
      Sa covariance obéit à la récursion de Lyapunov P_{j+1} = A_cl P_j A_cl^T + b_w b_w^T (Proposition A.1),
      avec P_0 = diag(0, 0, 0.25 sigma_xi^2). On lit sigma^2_x,wind(t) = P_j[0, 0] et on ajoute les variances
      empiriques du corps rigide sigma_rb (calibrées sur des graines d'entraînement, voir calibrate_sigma_rb).
      Rien ne dépend de la position : la table est calculée UNE SEULE FOIS à la construction (< 1 ms).
  3.2 Projection directionnelle (18) : seule la composante normale au mur compte,
      sigma_{i,k} = sqrt(n_x^2 sigma_x,k^2 + n_y^2 sigma_y,k^2). Une paroi verticale hérite de toute la dérive due
      au vent, le dessus d'une roche (survolé) seulement de la petite incertitude verticale.
  3.3 Pénalité convexe (19)-(20) : la probabilité de collision Phi(-d/sigma) est convexe et décroissante pour
      d >= 0 ; on la minore par le maximum de M = 6 tangentes en z = 0, 0.5, ..., 2.5 (constantes universelles
      ALPHA_TAN, BETA_TAN, Tableau 4). Dans le SOCP, une variable épigraphe q_{i,k} >= 0 par (mur, nœud) vérifie
      q_{i,k} >= alpha_m (n^T r_k - b)/sigma + beta_m (contraintes LINÉAIRES) et W_haz sum q est ajouté au coût.
      À l'optimum q = p_hat(z) (Proposition C.2) ; par la borne de l'union, sum q approche une borne supérieure de la
      probabilité de collision sur toute la descente. Le mur physique n^T r_k >= b - s_{i,k} garde un slack
      d'urgence s >= 0 pénalisé par W_em >> W_haz (pénalité l1 exacte, C.4) : le problème n'est JAMAIS infaisable.
      Avantage clé (Proposition C.3) : dans un couloir étroit (canyon S05) des marges dures z_eps sigma sur les deux
      parois rendraient le SOCP infaisable (Observation B.1) ; la pénalité, elle, centre automatiquement la
      trajectoire sur l'axe du couloir.

4. SOCP final (21)-(32), résolu à chaque replanning
---------------------------------------------------
  min  dt sum_k sigma_k  +  W_haz sum_{i,k} q_{i,k}  +  W_em sum_{i,k} s_{i,k}
  s.c. état initial mesuré, porte d'approche (LanderSOCP terminal="gate"),
       dynamique ZOH exacte avec vent connu + cisaillement + dérive de rafale attendue (6)-(9)      [linéaire]
       ||u_k|| <= sigma_k, bornes de Taylor (Question 1.3), inclinaison        [SOC, relaxation sans perte]
       n_{i,k}^T r_k >= b_{i,k} - s_{i,k}, s >= 0                              [linéaire, demi-plans issus de 5.1]
       q_{i,k} >= alpha_m (n_{i,k}^T r_k - b_{i,k}) / sigma_{i,k} + beta_m, q >= 0                  [linéaire]
Le problème reste un SOCP. Tout ce qui change entre deux résolutions (état, temps, dérive, murs, sigma) est un
cp.Parameter : n/sigma et b/sigma sont des paramètres séparés, donc le problème respecte les règles DPP et n'est
compilé qu'une fois (règle du DPP primer).

5. Interface avec le module obstacles (Section 1.3 du document, Question 5.1)
------------------------------------------------------------------------------
Le module 5.2 ne demande au module 5.1 qu'une chose : pour chaque nœud k, des demi-plans libres n_{i,k}^T r >= b_{i,k}
(||n|| = 1, b incluant déjà l'encombrement du lander). Pour que ce fichier soit autonome, HazardWalls fournit une
implémentation par défaut de type SCP : on linéarise l'extérieur de chaque polygone (rochers, zones interdites,
trapèzes de terrain de lander/geometry.py) autour d'une trajectoire de référence (plan précédent, ou à défaut un
chemin en « L » : transit horizontal puis descente verticale vers la porte), avec le point le plus proche et sa
normale, puis on gonfle par la boîte du lander : b = n^T q + |n_x| R_X + |n_y| R_Y (fonction support de la boîte).
On garde les n_walls murs les plus proches par nœud, sans doublons (le terrain est découpé en trapèzes de 2 m qui
donnent des demi-plans identiques sur un sol plat ; sans ce filtre, ils évinçaient la flèche rocheuse de S10) ;
les emplacements inutilisés sont désactivés (n = 0, b = -1e3, sigma = 1), comme le suggère la Section 7.3.
Glide slope : il est désactivé quand les murs sont actifs (ce sont eux qui protègent du sol ; le cône de 30 deg de
la Question 4 rendait le SOCP infaisable depuis un départ lointain et bas, S10), sauf en basse gravité où il
stabilise les approches balistiques (S07). Pour brancher un autre module 5.1 (par exemple la
décomposition GCS de l'équipe), il suffit de remplacer `self.walls` par un objet appelable path -> (n, b) avec les
mêmes formes.

6. Intégration dans la cascade (WindRobustController, hérite de CascadeController)
---------------------------------------------------------------------------------
  * act        : met à jour l'estimateur de rafale (Phase 2) avant chaque pas et mémorise l'action appliquée ;
  * _solve     : murs (5.1) -> sigma directionnels (3.1, 3.2) -> SOCP robuste avec dérive (Phase 1 + 3.3) ;
  * _search    : même recherche du temps de vol que la Question 4 (grille + section dorée), mais sur le coût total
                 (carburant + risque + slack) ; puis marge de 10 % et une itération SCP autour du meilleur plan ;
  * _replan    : inchangé (horizon décroissant, +15 %, +30 %, puis nouvelle recherche) ;
  * _reference : interpolation ZOH qui inclut le vent connu et la dérive de rafale attendue (Phase 1) ;
  * _track, _terminal, _fallback : loi PD de la Question 4 + feedforward de la Phase 2.

7. Choix des paramètres
-----------------------
  * W_haz = 150 : le score perd environ 150 points sur une collision (-50 au lieu de +100) ; le coût carburant du
    SOCP (dt sum sigma, en m/s) coûte environ 5 m/F ~ 0.29 point par m/s, ce qui donnerait W_haz ~ 500. Mais la borne
    de l'union compte plusieurs fois le même événement (nœuds voisins très corrélés, typiquement 3 à 5 nœuds par
    passage près d'un mur), d'où W_haz ~ 500/3.5 ~ 150, dans la plage [50, 500] du document. W_em = 20 W_haz.
  * lambda = 0.05, EN DEHORS de la plage [0.15, 0.30] du document. Mesure (étape 3 de la Section 7.5) sur
    S01/S03/S05/S08/S10, 4 graines, en rejouant le filtre hors ligne contre la vraie rafale du simulateur
    (diagnostic uniquement, la politique n'y a pas accès) : erreur RMS 0.82 m/s^2 pour lambda = 0.2 (presque autant
    que l'estimation nulle, 0.86), 0.65 pour lambda = 0.05 (optimum, comme l'estimateur fourni). Le résidu est très
    bruité (dispersion des moteurs de Gymnasium, jambes articulées) ; avec lambda = 0.2 le feedforward injectait ce
    bruit dans la commande et poussait le lander contre les parois du canyon.
  * gamma_ff = 0.7 (plage [0.65, 0.80]), sigma_est,0^2 = 0.25 sigma_xi^2 (valeurs du document).
  * sigma_rb = (0.12, 0.05) m : calibré (`python submissions/wind_robust.py calibrate`, 5 scénarios x 10 graines,
    vent coupé : RMS mesurés 0.117 m et 0.050 m). Une valeur plus prudente (0.20 m) a été testée et dégrade S05
    (20 % de succès au lieu de 60 %) : un tube trop large déforme inutilement le plan dans le canyon.
  * term_gains = (0.8, 0.5, 2.5) pour le maintien latéral de la descente terminale (au lieu de (0.3, 0.3, 2.0)).

8. Étude d'ablation (Question 5.3)
-----------------------------------
Les drapeaux use_drift (Phase 1), use_ff (Phase 2), use_risk (Phase 3) et use_walls (évitement 5.1) désactivent une
composante à la fois SANS recompiler un autre problème (on désactive simplement les paramètres correspondants).
Fabriques prêtes à l'emploi : make_policy (complet), make_no_drift, make_no_ff, make_no_risk, make_no_walls.
Résultats mesurés (graines d'entraînement 0-19, 10 scénarios) : complet 71.7 ; sans Phase 1 61.8 ; sans Phase 2 54.6 ;
sans Phase 3 74.1 ; sans murs 24.2 ; cascade de la Question 4 -0.5. La Phase 3 dégrade S05 (70 % de succès contre
90 % sans elle) : le plan centré demande de finir le freinage latéral pile au rebord du canyon, et le retard de
l'attitude (~0.3 s) fait dépasser l'axe (analyse complète : analysis/q53_s05_failure.py).

9. Guide de lecture du code (dans l'ordre où les fonctions sont appelées pendant un pas de simulation)
------------------------------------------------------------------------------------------------------
  lander.evaluate appelle policy.act(obs, info) toutes les 0.02 s :
    WindRobustController.act
      -> OUGustEstimator.update                     (Phase 2 : nouvelle estimation xi_hat de la rafale)
      -> CascadeController.act  (classe mère)       (choix de la phase de vol POWERED / TERMINAL / TOUCHDOWN)
           -> _step (classe mère)                   (toutes les 25 étapes : nouveau plan)
                -> _replan (classe mère) -> _solve  (horizon décroissant)  ou  _search -> _solve (nouvelle recherche)
                     _solve : _reference_path -> HazardWalls (murs 5.1) -> WindTube.sigma_dir (Phase 3.1-3.2)
                              -> RobustLanderSOCP.solve (Phase 1 + Phase 3.3) -> plan (dict)
                -> _track    (POWERED)  : _reference (Phase 1) + loi PD + _wind_feedforward (Phase 2)
                   _terminal (TERMINAL) / _fallback (pas de plan) : loi simple + _wind_feedforward
                -> _to_action (classe mère) : accélération commandée -> (poussée principale, moteurs latéraux)
  Ordre conseillé pour lire le fichier : les fonctions utilitaires (constantes, expected_gust_drift), puis les trois
  briques indépendantes (OUGustEstimator, WindTube, HazardWalls), puis le SOCP (RobustLanderSOCP), puis le contrôleur
  qui assemble le tout (WindRobustController).

Utilisation :
    python -m lander.evaluate --policy submissions/wind_robust.py --split train --n 20
    python -m lander.evaluate --policy submissions.wind_robust:make_no_risk --split train --n 20     # ablation
    python -m lander.evaluate --policy submissions/wind_robust.py --scenarios S05 --seed 3 --gif
    python submissions/wind_robust.py calibrate          # calibration de sigma_rb (étape 2 de la Section 7.5)
    python submissions/wind_robust.py tube               # tube prédit vs dispersion mesurée

Remarque pour le leaderboard : sur le serveur, le package `lander` est celui du kit de départ (sans nos réponses aux
Questions 3 et 4). Si des copies locales `lcvx.py` et `controllers.py` sont placées à côté de submission.py dans le
zip, elles sont importées en priorité (bloc try/except des imports ci-dessous).
"""
# ==================================================================================================================
# IMPORTS
# ==================================================================================================================
import math                 # exp, sqrt, erfc sur des NOMBRES (plus rapide que numpy quand il n'y a qu'un scalaire)
import os
import sys
import time                 # chronométrage de chaque résolution du SOCP (champ solve_time de LanderSolution)
from dataclasses import dataclass

import numpy as np
import cvxpy as cp

# Quand lander.evaluate charge ce fichier (--policy submissions/wind_robust.py), il n'ajoute que le dossier
# submissions/ au chemin de recherche de Python. On ajoute aussi la racine du dépôt (le dossier parent) pour que
# `import lander...` fonctionne. C'est la même astuce que dans submissions/baseline_cascade.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# PARAMS : paramètres physiques du lander (masses, poussée F, F_side, alpha...), lander/params.py.
# R_X = 1.0 m, R_Y = 1.15 m : demi-largeur et demi-hauteur de la boîte qui contient le lander en vol (jambes incluses).
from lander.params import P as PARAMS, R_X, R_Y
# DT = 0.02 s : pas du simulateur et de la boucle de contrôle (50 Hz). Noté h dans le document.
from lander.model import DT
# Fournis par le kit (lander/wind.py) :
#   GustEstimator : estimateur de rafale de base, dont on hérite (on ne change que le filtre) ;
#   known_wind(wind, y) : partie CONNUE du vent à la hauteur y, w_bar + s (y + l) / 10 ;
#   engine_powers(action) : poussée effectivement appliquée par le simulateur (zones mortes incluses).
from lander.wind import GustEstimator, known_wind, engine_powers
# obstacle_polygons(spec) : tous les obstacles (rochers, zones interdites, ET le terrain découpé en trapèzes) sous
# forme de polygones convexes, dans le repère du pad (origine au centre du pad, à la hauteur d'atterrissage).
from lander.geometry import obstacle_polygons

# Le SOCP (LanderSOCP) et la cascade (CascadeController) sont nos réponses aux Questions 3 et 4. Sur le serveur du
# leaderboard, le package `lander` est celui du kit de départ, où ces fonctions lèvent NotImplementedError. On essaie
# donc d'abord des copies locales lcvx.py / controllers.py posées à côté de submission.py dans le zip ; si elles
# n'existent pas (cas local), Python lève ImportError et on prend les versions du package du projet.
try:
    from lcvx import LanderSOCP, LanderSolution, golden_section, is_lossless
    from controllers import CascadeController
except ImportError:
    from lander.lcvx import LanderSOCP, LanderSolution, golden_section, is_lossless
    from lander.controllers import CascadeController


# ==================================================================================================================
# CONSTANTES UNIVERSELLES DE LA PHASE 3.3 (Section 7.1 et Tableau 4 du document)
# ==================================================================================================================
# Idée : la probabilité de toucher un mur situé à une distance d (en nombre d'écarts-types z = d / sigma) vaut
# Phi(-z). Cette fonction n'est pas utilisable telle quelle dans un SOCP. Mais elle est convexe pour z >= 0, donc
# chacune de ses tangentes est EN DESSOUS de la courbe, et le maximum de quelques tangentes en donne une très bonne
# approximation affine par morceaux. Un maximum de fonctions affines s'écrit avec des contraintes linéaires
# (q >= tangente_1, q >= tangente_2, ...), ce qui garde le problème convexe. Ces tangentes ne dépendent d'aucun
# scénario : on les calcule une fois pour toutes ici.

def norm_tail(z):
    """Phi(-z) = P(Z > z) pour Z ~ N(0, 1) : la queue de la loi normale.
    erfc est la fonction d'erreur complémentaire de la bibliothèque standard : Phi(-z) = erfc(z / sqrt 2) / 2."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_pdf(z):
    """Densité phi(z) = e^{-z^2/2} / sqrt(2 pi) de la loi normale standard (c'est aussi -d/dz Phi(-z))."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


# Les M = 6 points de tangence z_m = 0, 0.5, 1.0, 1.5, 2.0, 2.5 (équirépartis, comme dans le Tableau 4).
Z_TAN = np.linspace(0.0, 2.5, 6)
# Tangente à f(z) = Phi(-z) au point z_m : L_m(z) = f(z_m) + f'(z_m) (z - z_m) = alpha_m z + beta_m, avec
# f'(z) = -phi(z). Donc la pente vaut alpha_m = -phi(z_m) (négative : le risque diminue quand on s'éloigne du mur)...
ALPHA_TAN = np.array([-norm_pdf(z) for z in Z_TAN])
# ... et l'ordonnée à l'origine vaut beta_m = Phi(-z_m) - alpha_m z_m = Phi(-z_m) + z_m phi(z_m)    (équation (19)).
BETA_TAN = np.array([norm_tail(z) + z * norm_pdf(z) for z in Z_TAN])
# Point où la dernière tangente (la plus plate) coupe zéro : z_cut = -beta_M / alpha_M ~ 2.85. Au-delà de 2.85 écarts-
# types, toutes les tangentes sont négatives et q = max(0, tangentes) = 0 : le mur est jugé sans danger (Prop. C.1 iii).
# Cette valeur n'est pas utilisée par le code ; elle sert de repère pour interpréter les résultats.
Z_CUT = -BETA_TAN[-1] / ALPHA_TAN[-1]
# Valeur de b qui DÉSACTIVE un emplacement de mur (Section 7.3) : avec n = 0, la contrainte n^T r >= b devient
# 0 >= -1000, toujours vraie, et la marge normalisée z = (0 - (-1000)) / 1 = 1000 rend toutes les tangentes très
# négatives, donc q = 0. On peut ainsi avoir un nombre FIXE d'emplacements dans le SOCP compilé (règle DPP) et n'en
# utiliser qu'une partie.
OFF_B = -1e3


def expected_gust_drift(xi0, gust_tau, t_start, dt):
    """Dérive attendue due à la rafale estimée xi0 (Phase 1, équations (4)-(5)).

    Au moment du replanning on estime la rafale à xi0. Une rafale Ornstein-Uhlenbeck ne disparaît pas d'un coup :
    en moyenne elle décroît comme xi0 e^{-s/tau} (s = temps écoulé depuis le replanning, tau = gust_tau). Sur un
    intervalle [t_start, t_start + dt), cette accélération moyenne ajoute :
        à la vitesse :   dv = intégrale de xi0 e^{-s/tau} ds
                            = xi0 e^{-t_start/tau} tau (1 - e^{-dt/tau})
        à la position :  dx = double intégrale (vitesse puis position)
                            = xi0 e^{-t_start/tau} tau (dt - tau (1 - e^{-dt/tau}))
    Ces deux nombres sont CONNUS au moment de résoudre le SOCP : ils entrent dans la dynamique comme des constantes,
    ce qui la garde linéaire.

    t_start et dt peuvent être des tableaux : un intervalle par nœud du SOCP (t_start = t_k, dt = Delta t).
    dt peut aussi être une durée partielle tau < Delta t : c'est ce qui sert à interpoler la référence entre deux
    nœuds dans WindRobustController._reference. Retourne (dv, dx), en m/s et en m."""
    t_start, dt = np.asarray(t_start, float), np.asarray(dt, float)       # accepte nombres et tableaux
    decay = xi0 * np.exp(-t_start / gust_tau)       # rafale moyenne au DÉBUT de l'intervalle, xi0 e^{-t_start/tau}
    one = 1.0 - np.exp(-dt / gust_tau)              # facteur commun (1 - e^{-dt/tau}) des deux formules
    return decay * gust_tau * one, decay * gust_tau * (dt - gust_tau * one)


# ==================================================================================================================
# PHASE 2 (partie 1) : ESTIMATEUR DE RAFALE, équations (10)-(11)
# ==================================================================================================================
class OUGustEstimator(GustEstimator):
    """Estimateur de rafale du document (Phase 2).

    Principe : à chaque pas on MESURE l'accélération horizontale du lander, (v_x(j) - v_x(j-1)) / DT, et on lui
    retire tout ce que l'on sait expliquer : la poussée des moteurs (avec la commande vraiment appliquée) et le vent
    connu. Ce qui reste (le résidu xi_raw) est la rafale, plus du bruit (dispersion aléatoire des moteurs de
    Gymnasium, efforts des jambes articulées). On filtre ce résidu :
        xi_hat_j = (1 - lambda) phi xi_hat_{j-1} + lambda xi_raw_j,     phi = e^{-DT/tau}.
    Le terme phi xi_hat_{j-1} est la PRÉDICTION de la rafale : entre deux pas, une rafale OU décroît en moyenne
    d'un facteur phi. lambda règle le compromis : grand = réactif mais bruité, petit = lisse mais en retard.
    Le document propose lambda dans [0.15, 0.30] ; on a mesuré que 0.05 est bien meilleur dans ce simulateur
    (voir le point 7 du résumé). L'estimateur du kit (GustEstimator) fait la même chose sans le facteur phi ;
    on en hérite pour réutiliser son état (xi_hat, prev_s), son reset() et ses paramètres."""

    def __init__(self, spec, params=PARAMS, lam=0.05, enabled=True):
        # La classe mère lit dans spec : self.wind = spec["wind"], self.gust_tau = spec["wind"]["gust_tau"], et
        # garde self.p = params (pour F et F_side). Elle appelle aussi reset() : xi_hat = 0, prev_s = None.
        super().__init__(spec, params, enabled=enabled)
        self.lam = lam                                   # gain lambda du filtre (11)
        self.phi = math.exp(-DT / self.gust_tau)         # décroissance moyenne de la rafale en un pas de 0.02 s

    def update(self, s, a_prev):
        """Appelée à chaque pas par WindRobustController.act.
        s : état courant (x, y, vx, vy, theta, omega, m) en unités SI, repère du pad ;
        a_prev : action envoyée au simulateur au pas PRÉCÉDENT (c'est elle qui a produit le changement de vitesse
        que l'on mesure maintenant). Retourne la nouvelle estimation xi_hat (m/s^2)."""
        if self.prev_s is not None and self.enabled:     # au tout premier pas, pas d'état précédent : rien à mesurer
            ps = self.prev_s                             # état au pas précédent
            # Poussée réellement produite : throttle principal pm dans {0} U [0.5, 1] et commande latérale c,
            # en tenant compte des zones mortes du simulateur (a0 <= 0 : moteur coupé ; |a1| <= 0.5 : rien).
            pm, c = engine_powers(a_prev)
            th, m = ps[4], ps[6]                         # inclinaison et masse au pas précédent
            # Accélération horizontale due aux moteurs : le moteur principal pousse le long de l'axe du corps
            # (-sin th, cos th), d'où la composante -F pm sin(th) ; les moteurs latéraux poussent perpendiculairement
            # à l'axe, (cos th, sin th), d'où F_side c cos(th). On divise par la masse pour avoir une accélération.
            a_motor = (self.p.F * pm * -np.sin(th) + self.p.F_side * c * np.cos(th)) / m
            # Résidu (10) : accélération mesurée - moteurs - vent connu (évalué à l'altitude du pas précédent).
            xi_raw = (s[2] - ps[2]) / DT - a_motor - known_wind(self.wind, ps[1])
            # Filtre (11) : prédiction OU (phi xi_hat) corrigée par une fraction lambda de la nouvelle mesure.
            self.xi_hat = (1.0 - self.lam) * self.phi * self.xi_hat + self.lam * xi_raw
        self.prev_s = np.array(s, dtype=float).copy()    # mémorise l'état pour le pas suivant (copie : s peut changer)
        return self.xi_hat


# ==================================================================================================================
# PHASE 3.1 et 3.2 : TUBE D'INCERTITUDE (de combien le lander peut s'écarter du plan ?)
# ==================================================================================================================
class WindTube:
    """Écart-type de l'écart entre la position réelle et la position planifiée, en fonction du temps écoulé depuis le
    replanning (Phase 3.1), puis projeté sur la normale de chaque mur (Phase 3.2).

    Modèle (Annexe A) : on suit l'erreur horizontale zeta = (e_x, e_vx, delta_xi), où e_x = x - x_ref,
    e_vx = v_x - v_x,ref et delta_xi = rafale réelle - rafale prévue par le plan. Sous la loi PD horizontale, l'erreur
    subit l'accélération a_err = -Kp e_x - Kd e_vx + delta_xi. Prudemment, on ignore le feedforward de la Phase 2
    (gamma_ff = 0) : si l'estimateur est en retard, le tube reste valable. Discrétisé sur un pas h :
        zeta_{j+1} = A_cl zeta_j + b_w eps_j       (eps_j ~ N(0, 1) : le bruit qui alimente la rafale OU)
    et la covariance P_j = E[zeta_j zeta_j^T] suit la récursion de Lyapunov
        P_{j+1} = A_cl P_j A_cl^T + b_w b_w^T,     P_0 = diag(0, 0, est_frac sigma_xi^2)
    (au replanning la position et la vitesse sont mesurées, donc d'erreur nulle ; seule l'erreur d'estimation de la
    rafale est déjà présente). La variance de position due au vent est P_j[0, 0]. On y ajoute les variances
    sigma_rb^2 des effets du corps rigide (retard d'attitude, couple de turbulence) mesurées sur des vols sans vent,
    voir calibrate_sigma_rb, ce qui donne l'équation (17).

    Rien de tout cela ne dépend de la position du lander : la table est calculée UNE fois, dans le constructeur."""

    def __init__(self, wind, kp_x, kd_x, sigma_rb=(0.15, 0.10), est_frac=0.25, t_max=15.0, h=DT):
        # wind = spec["wind"] ; kp_x, kd_x = gains PD horizontaux de la cascade ; sigma_rb = (sigma_rb,x, sigma_rb,y)
        # (le contrôleur passe toujours ses valeurs calibrées (0.12, 0.05), la valeur par défaut ici ne sert pas) ;
        # est_frac = part de la variance de la rafale encore inconnue au replanning ; t_max = durée de la table.
        s_xi, tau = wind["gust_std"], wind["gust_tau"]   # écart-type stationnaire et temps de corrélation des rafales
        phi = math.exp(-h / tau)                         # décroissance de la rafale en un pas
        # Matrice A_cl (16), ligne par ligne, obtenue en intégrant exactement a_err (constante sur un pas h) :
        #   e_x(j+1)   = e_x + h e_vx + h^2/2 a_err   ->  (1 - h^2/2 Kp) e_x + (h - h^2/2 Kd) e_vx + h^2/2 delta_xi
        #   e_vx(j+1)  = e_vx + h a_err               ->  -h Kp e_x + (1 - h Kd) e_vx + h delta_xi
        #   delta_xi(j+1) = phi delta_xi + bruit      (la surprise de rafale suit elle-même un processus OU, (34))
        A = np.array([[1 - 0.5 * h * h * kp_x, h - 0.5 * h * h * kd_x, 0.5 * h * h],
                      [-h * kp_x, 1 - h * kd_x, h],
                      [0.0, 0.0, phi]])
        # Le bruit n'entre que par la rafale, avec l'amplitude qui maintient l'écart-type stationnaire à sigma_xi.
        bw = np.array([0.0, 0.0, s_xi * math.sqrt(1 - phi * phi)])
        # Proposition A.1 : la covariance reste bornée si toutes les valeurs propres de A_cl sont de module < 1
        # (c'est le cas avec les gains de la cascade ; l'assertion protège contre un mauvais réglage).
        assert np.max(np.abs(np.linalg.eigvals(A))) < 1.0, "the PD gains must stabilize the error dynamics"
        self.h, self.J = h, int(round(t_max / h))       # J = nombre de pas de 0.02 s dans la table (750 pour 15 s)
        # P : covariance initiale ; Q = b_w b_w^T : covariance ajoutée à chaque pas par le bruit.
        P, Q = np.diag([0.0, 0.0, est_frac * s_xi ** 2]), np.outer(bw, bw)
        var = np.empty(self.J + 1)
        for j in range(self.J + 1):                     # récursion de Lyapunov (15) : 750 produits de matrices 3x3,
            var[j] = P[0, 0]                            # < 1 ms. On ne garde que la variance de position e_x.
            P = A @ P @ A.T + Q
        self.var_x_wind = var                           # sigma_x,wind^2 après j pas (var_x_wind, Section 7.2)
        self.A_cl, self.b_w = A, bw                     # gardés pour inspection (check_tube, rapport)
        self.srb = np.maximum(np.asarray(sigma_rb, float), 0.05)     # plancher de 5 cm (Section 7.5) : évite sigma ~ 0

    def sigma_xy(self, t):
        """(sigma_x, sigma_y) en mètres aux instants t (secondes depuis le replanning, nombre ou tableau).
        C'est la fonction get_sigma_xy de la Section 7.2 : on lit la table au pas rapide j = t / h le plus proche
        (plafonné à J ; au-delà, la variance a de toute façon atteint sa valeur stationnaire)."""
        j = np.minimum(np.round(np.asarray(t, float) / self.h).astype(int), self.J)
        sx = np.sqrt(self.var_x_wind[j] + self.srb[0] ** 2)          # vent horizontal + corps rigide, (17)
        return sx, np.full_like(sx, self.srb[1])                    # verticalement : seulement le corps rigide

    def sigma_dir(self, n, t):
        """Phase 3.2, équation (18) : seule la composante de l'écart PERPENDICULAIRE au mur peut provoquer une
        collision. Pour un mur de normale unitaire n = (n_x, n_y), l'écart-type dans cette direction vaut
            sigma = sqrt(n^T Sigma n) = sqrt(n_x^2 sigma_x^2 + n_y^2 sigma_y^2).
        Une paroi verticale (n = (+-1, 0)) hérite donc de toute la dérive horizontale due au vent, le dessus d'un
        rocher (n = (0, 1)) seulement de la petite incertitude verticale.
        n : (n_walls, N+1, 2) normales des murs à chaque nœud ; t : (N+1,) instants des nœuds.
        Retourne (n_walls, N+1). Un mur désactivé (n = 0) donnerait sigma = 0 et une division par zéro dans le SOCP :
        on lui met sigma = 1 (Section 7.3), sans effet puisque sa marge est de toute façon énorme."""
        sx, sy = self.sigma_xy(t)                        # (N+1,) chacun ; diffusés sur l'axe des murs ci-dessous
        sig = np.sqrt(n[..., 0] ** 2 * sx ** 2 + n[..., 1] ** 2 * sy ** 2)
        return np.where(sig > 1e-9, sig, 1.0)


# ==================================================================================================================
# INTERFACE AVEC LE MODULE 5.1 : DEMI-PLANS LIBRES (n_{i,k}, b_{i,k})
# ==================================================================================================================
# Le document (Section 1.3) suppose que le module d'évitement d'obstacles fournit, pour chaque nœud k, des demi-plans
# n_{i,k}^T r >= b_{i,k} qui décrivent la zone sûre. Le module 5.1 de l'équipe n'étant pas sur cette branche, on en
# fournit un simple, de type SCP (programmation convexe séquentielle) : on « remplace » chaque obstacle par la
# droite qui le touche au point le plus proche de la trajectoire de référence.

def closest_points(poly, pts):
    """Point le plus proche d'un polygone convexe, pour K points à la fois (version vectorisée de
    lander.geometry.closest_point_convex, testée identique ; la version de base coûterait ~80 ms par SOCP).

    poly : sommets (n, 2) d'un polygone convexe, parcourus dans le sens trigonométrique ;  pts : (K, 2).
    Retourne :
      q (K, 2)  point du polygone le plus proche de chaque point ;
      d (K,)    distance signée (positive dehors, négative dedans) ;
      n (K, 2)  normale unitaire qui va du polygone vers le point.
    Le demi-plan {r : n^T (r - q) >= 0} contient tout ce qui est du côté du point, et le polygone est entièrement de
    l'autre côté (c'est un hyperplan d'appui, car le polygone est convexe). Rester dans ce demi-plan garantit donc
    de ne pas toucher l'obstacle : c'est une approximation CONSERVATIVE (intérieure) de « être hors du polygone »."""
    V = np.asarray(poly, float)                                           # sommets (n, 2)
    E = np.roll(V, -1, axis=0) - V                                        # arêtes : E[i] = V[i+1] - V[i]   (n, 2)
    L2 = np.maximum(np.sum(E * E, axis=1), 1e-12)                         # longueur^2 de chaque arête (jamais 0)
    # Normale sortante de chaque arête : pour un parcours trigonométrique, l'extérieur est à DROITE de l'arête,
    # donc la normale est (E_y, -E_x), que l'on normalise.
    nout = np.stack([E[:, 1], -E[:, 0]], axis=1) / np.sqrt(L2)[:, None]
    rel = pts[:, None, :] - V[None]                                       # vecteur sommet i -> point k   (K, n, 2)
    # sd[k, i] : distance signée du point k à la DROITE qui porte l'arête i (> 0 du côté extérieur).
    sd = np.einsum("kni,ni->kn", rel, nout)
    # Projection du point sur le SEGMENT i : paramètre t dans [0, 1] le long de l'arête (0 = sommet i, 1 = sommet i+1).
    t = np.clip(np.einsum("kni,ni->kn", rel, E) / L2, 0.0, 1.0)
    Q = V[None] + t[..., None] * E[None]                                  # point le plus proche sur chaque arête
    d = np.linalg.norm(pts[:, None, :] - Q, axis=2)                       # distance à chaque arête   (K, n)
    K = np.arange(len(pts))                                               # indices 0..K-1 pour l'indexation avancée
    # Cas 1, point à l'extérieur : le point le plus proche du polygone est le plus proche parmi ceux des arêtes,
    # et la normale va de ce point vers le point extérieur.
    i = np.argmin(d, axis=1)
    q_out, d_out = Q[K, i], d[K, i]
    n_out = (pts - q_out) / np.maximum(d_out, 1e-9)[:, None]
    # Cas 2, point à l'intérieur (toutes les distances signées <= 0) : on sort par l'arête la moins pénétrée (celle
    # dont sd est le plus grand, donc le plus proche de 0), selon sa normale sortante.
    ii = np.argmax(sd, axis=1)
    n_in = nout[ii]
    q_in = pts - sd[K, ii][:, None] * n_in                                # projection du point sur cette arête
    inside = np.all(sd <= 0.0, axis=1)
    # On choisit, point par point, le résultat du bon cas.
    return (np.where(inside[:, None], q_in, q_out), np.where(inside, sd[K, ii], d_out),
            np.where(inside[:, None], n_in, n_out))


class HazardWalls:
    """Fournisseur par défaut des demi-plans libres (module 5.1, approche SCP).

    Pour chaque nœud k d'une trajectoire de référence et chaque obstacle (rochers, zones interdites et trapèzes de
    terrain), on calcule le point le plus proche q et la normale n (closest_points), puis on « gonfle » le demi-plan
    de la taille du lander. Le centre r du lander doit être assez loin pour que TOUTE sa boîte [-R_X, R_X] x
    [-R_Y, R_Y] reste du bon côté. Le point de la boîte le plus avancé vers le mur dépasse le centre de
    |n_x| R_X + |n_y| R_Y dans la direction -n (c'est la fonction support de la boîte), d'où :
        n^T r >= b,     b = n^T q + |n_x| R_X + |n_y| R_Y.
    Il peut y avoir beaucoup d'obstacles ; pour garder un SOCP de taille fixe, on ne garde à chaque nœud que les
    n_walls murs les plus proches (et à moins de d_active mètres). Les emplacements restants sont désactivés.

    Utilisation : n, b = walls(path), avec path (N+1, 2) ; n : (n_walls, N+1, 2), b : (n_walls, N+1)."""

    def __init__(self, spec, n_walls=4, d_active=4.0):
        self.polys, self.kinds = obstacle_polygons(spec)                # polygones (repère du pad) et leur type
        # Boîte englobante de chaque polygone : permet d'écarter en une comparaison les obstacles très loin.
        self.lo = [P.min(axis=0) for P in self.polys]                   # coin bas-gauche (x_min, y_min)
        self.hi = [P.max(axis=0) for P in self.polys]                   # coin haut-droit (x_max, y_max)
        self.n_walls, self.d_active = n_walls, d_active

    def __call__(self, path):
        K, n_o = len(path), len(self.polys)        # K = N+1 nœuds, n_o obstacles
        # Pour chaque (obstacle o, nœud k) : distance « gonflée » n^T p - b (< 0 : la boîte touche déjà l'obstacle),
        # normale et offset b du demi-plan. inf = obstacle ignoré à ce nœud.
        dist = np.full((n_o, K), np.inf)
        nrm, off = np.zeros((n_o, K, 2)), np.zeros((n_o, K))
        reach = np.array([R_X, R_Y]) + self.d_active                    # portée utile en x et en y
        for o, poly in enumerate(self.polys):
            # Écart entre chaque nœud et la boîte englobante (0 si le nœud est dedans dans cette direction).
            gap = np.maximum(np.maximum(self.lo[o] - path, path - self.hi[o]), 0.0)
            near = np.all(gap <= reach, axis=1)                         # nœuds assez proches de cet obstacle
            if not near.any():
                continue                                                # obstacle trop loin de toute la trajectoire
            q, _, n = closest_points(poly, path[near])                  # point le plus proche et normale
            b = np.sum(n * q, axis=1) + np.abs(n[:, 0]) * R_X + np.abs(n[:, 1]) * R_Y    # offset gonflé
            nrm[o, near], off[o, near] = n, b
            dist[o, near] = np.sum(n * path[near], axis=1) - b          # marge de la référence par rapport au mur
        # Sortie : n_walls emplacements par nœud, tous désactivés au départ (n = 0, b = OFF_B).
        n_out = np.zeros((self.n_walls, K, 2))
        b_out = np.full((self.n_walls, K), OFF_B)
        order = np.argsort(dist, axis=0)                # pour chaque nœud : obstacles triés par distance croissante
        for k in range(K):
            j = 0                                                       # nombre d'emplacements déjà remplis au nœud k
            for o in order[:, k]:
                if j == self.n_walls or dist[o, k] >= self.d_active:
                    break                                               # plus de place, ou obstacles restants trop loin
                # Le terrain est découpé en trapèzes de 2 m : sur un sol plat, plusieurs segments voisins donnent le
                # MÊME demi-plan (même normale, même offset). On n'en garde qu'un, sinon ces doublons occupent tous
                # les emplacements et un rocher proche (la flèche de S10) disparaît du problème.
                if any(np.abs(n_out[i, k] - nrm[o, k]).max() < 0.05 and abs(b_out[i, k] - off[o, k]) < 0.05
                       for i in range(j)):
                    continue
                n_out[j, k], b_out[j, k] = nrm[o, k], off[o, k]
                j += 1
        return n_out, b_out


# ==================================================================================================================
# LE SOCP ROBUSTE, équations (21)-(32)
# ==================================================================================================================
@dataclass
class RobustSolution(LanderSolution):
    """Solution du SOCP robuste : tous les champs de LanderSolution (positions p, vitesses v, log-masse z,
    accélérations u, carburant...), plus trois champs propres à ce fichier."""
    risk: float = 0.0            # sum q_{i,k} : par la borne de l'union, majore ~ la probabilité de collision (56)
    cost: float = 0.0            # valeur optimale de l'objectif (21) : carburant + W_haz risque + W_em slack
    xi0: float = 0.0             # rafale estimée utilisée par la Phase 1 pour ce plan


class RobustLanderSOCP(LanderSOCP):
    """Le SOCP de la Question 3/4 (LanderSOCP), complété par la Phase 1 et la Phase 3.3.

    On RÉUTILISE tel quel de LanderSOCP (lander/lcvx.py) :
      * les variables r (positions), v (vitesses), z (log-masse), u (accélération de poussée), s (sigma = Gamma/m) ;
      * la dynamique ZOH exacte, avec le cisaillement (argument `shear` : terme (s/10) y_k dans l'accélération
        horizontale) et deux termes de dérive par intervalle, les paramètres dvx (vitesse) et dpx (position) ;
      * la relaxation sans perte ||u_k|| <= sigma_k, les bornes de Taylor, l'inclinaison maximale, la porte
        d'approche, le glide slope et les limites de la carte ;
      * les contraintes de mur AVEC SLACK n^T r_k >= b - s (n_obs = n_walls emplacements, imposées aux nœuds k et
        k+1 pour protéger aussi la corde entre deux nœuds) : c'est la contrainte (30). Leur poids w_slack joue le rôle
        de W_em, le « poids d'urgence ».
    On AJOUTE : les variables de risque q_{i,k}, leurs contraintes (31)-(32) et le terme W_haz sum q dans l'objectif."""

    def __init__(self, p=PARAMS, N=20, g=-10.0, n_walls=4, w_haz=150.0, w_em=None, **kw):
        w_em = 20.0 * w_haz if w_em is None else w_em             # W_em >= 20 W_haz (Section 7.5)
        # Construit tout le SOCP de base avec n_walls emplacements de murs ; **kw transmet les autres réglages
        # (mode, terminal, theta_max_deg, glide_from, thrust_frac, y_gate, v_gate, x_tol, map_limits, shear).
        super().__init__(p, N, g, n_obs=n_walls, w_slack=w_em, **kw)
        self.n_walls, self.w_haz, self.w_em = n_walls, w_haz, w_em
        # Paramètres du risque. On veut z = (n^T r - b) / sigma. Écrire (1/sigma) * (n^T r) multiplierait deux
        # paramètres (1/sigma et n) par une variable (r), ce qui est interdit par les règles DPP (le problème devrait
        # être recompilé à chaque résolution). On donne donc directement Hs = n / sigma et bs = b / sigma : un seul
        # paramètre multiplie la variable, c'est DPP.
        self.Hs = [cp.Parameter((N + 1, 2), name=f"Hs{j}") for j in range(n_walls)]
        self.bs = [cp.Parameter(N + 1, name=f"bs{j}") for j in range(n_walls)]
        cons, risk = [], []
        for Hs, bs in zip(self.Hs, self.bs):                                       # un emplacement de mur à la fois
            q = cp.Variable(N, nonneg=True, name="q")   # q_{i,k} >= 0, nœuds k = 1..N (32) ; nœud 0 = état mesuré
            # Marge normalisée z_k = n_k^T r_k / sigma_k - b_k / sigma_k, pour k = 1..N (vecteur de longueur N).
            # cp.multiply(Hs[1:], r[1:]) multiplie terme à terme (N, 2) x (N, 2) ; la somme sur l'axe 1 donne les
            # produits scalaires n_k^T r_k / sigma_k.
            z = cp.sum(cp.multiply(Hs[1:], self.r[1:]), axis=1) - bs[1:]
            # Épigraphe du maximum des tangentes (31) : q au-dessus de CHACUNE des 6 droites alpha_m z + beta_m.
            # Comme q est minimisé (il est dans l'objectif avec un poids positif), il se colle à la plus haute des
            # droites (ou à 0) : q = max(0, L_1(z), ..., L_6(z)) ~ Phi(-z) à l'optimum (Proposition C.2).
            cons += [q >= a * z + b for a, b in zip(ALPHA_TAN, BETA_TAN)]
            risk.append(cp.sum(q))                          # risque total de cet emplacement sur tout l'horizon
        self.risk = cp.sum(cp.hstack(risk)) if risk else cp.Constant(0.0)     # sum_{i,k} q_{i,k}
        # Objectif (21) = objectif de LanderSOCP (carburant dt sum sigma + W_em sum slack) + W_haz sum q.
        # self.prob.objective.args[0] est l'expression qui était minimisée dans LanderSOCP.
        obj = self.prob.objective.args[0] + w_haz * self.risk
        # Nouveau problème : mêmes contraintes que LanderSOCP + celles du risque. Il remplace self.prob, donc
        # LanderSOCP.set_time et les paramètres hérités continuent de fonctionner tels quels.
        self.prob = cp.Problem(cp.Minimize(obj), self.prob.constraints + cons)
        assert self.prob.is_dcp(dpp=True), "problem must be DPP"       # compilé une seule fois par CVXPY

    def solve(self, x0, tgo, m0=None, wind_const=0.0, xi0=0.0, gust_tau=1.0, walls=None, sigma=None,
              allow_off=False, solver=cp.CLARABEL):
        """Résout le SOCP robuste pour un temps de vol tgo. Même déroulé que LanderSOCP.solve : on donne une valeur
        à tous les paramètres, on résout, on empaquette la solution. Seuls changent les termes de dérive (Phase 1)
        et les paramètres des murs et du risque (Phase 3).

        x0 = (x, y, vx, vy) état mesuré ; tgo = temps de vol jusqu'à la porte (s) ; m0 = masse actuelle (kg) ;
        wind_const = partie CONSTANTE du vent connu, w_bar + s l / 10 (la partie proportionnelle à y est dans
        l'argument `shear` du constructeur) ; xi0 = rafale estimée maintenant ; gust_tau = temps de corrélation ;
        walls = (n, b) de forme (n_walls, N+1, 2) et (n_walls, N+1), ou None (aucun mur) ;
        sigma = écarts-types directionnels (n_walls, N+1), ou None pour désactiver le risque (ablation) ;
        allow_off = basse gravité, où le moteur peut s'éteindre. Retourne une RobustSolution, ou None si échec."""
        N = self.N
        m0 = self.p.m_wet if m0 is None else m0
        # Paramètres qui dépendent du temps de vol : dt, dt^2, alpha dt, points de Taylor, bornes de masse.
        # t = instants des nœuds 0, dt, ..., tgo (depuis maintenant).
        t = self.set_time(tgo, m0)
        if allow_off:                                # basse gravité : le moteur peut être coupé, la poussée minimale
            self.a1.value = np.zeros(N)              # devient 0 (ensemble {0} U [rho1, rho2] relaxé à [0, rho2]), et la
            self.zhi.value = np.full(N + 1, np.log(m0))   # masse peut rester égale à m0 (aucune consommation)
        dt = tgo / N
        self.x0.value = np.asarray(x0, float)
        # Phase 1, équations (6) et (8). Sur chaque intervalle k, LanderSOCP ajoute à la vitesse horizontale dvx[k] et
        # à la position horizontale dpx[k]. On y met :
        #   - le vent connu constant (w_bar + s l / 10), intégré exactement : dt w et dt^2/2 w ;
        #   - la dérive attendue de la rafale qui décroît, (4)-(5), calculée par expected_gust_drift.
        dv_xi, dx_xi = expected_gust_drift(xi0, gust_tau, t[:-1], dt)       # un intervalle par nœud k = 0..N-1
        self.dvx.value = dt * wind_const + dv_xi
        self.dpx.value = 0.5 * dt * dt * wind_const + dx_xi
        # Murs (30) et risque (31). Sans murs : tous les emplacements désactivés (n = 0, b = OFF_B).
        if walls is None:
            walls = (np.zeros((self.n_walls, N + 1, 2)), np.full((self.n_walls, N + 1), OFF_B))
        n, b = walls
        for j in range(self.n_walls):
            self.H[j].value, self.b[j].value = n[j], b[j]            # mur physique avec slack (paramètres hérités)
            if sigma is None:                        # risque désactivé (ablation) : marge énorme -> q = 0 à l'optimum
                self.Hs[j].value, self.bs[j].value = np.zeros((N + 1, 2)), np.full(N + 1, OFF_B)
            else:                                    # n / sigma et b / sigma (voir le constructeur, règle DPP)
                self.Hs[j].value, self.bs[j].value = n[j] / sigma[j][:, None], b[j] / sigma[j]
        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=solver)           # Clarabel, le solveur conique par défaut de CVXPY
        except cp.SolverError:
            return None                              # échec numérique du solveur : traité comme « pas de plan »
        st = time.perf_counter() - t0                # temps de résolution (s)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return None                              # infaisable (ne devrait plus arriver grâce aux slacks)
        z = self.z.value
        fuel = float(np.exp(z[0]) - np.exp(z[-1]))   # carburant planifié = masse initiale - masse finale (kg)
        # Même ordre de champs que LanderSolution : status, tgo, t, p, v, z, u, sigma, fuel (kg), fuel_s (secondes
        # de pleine poussée), landing_error, slack, solve_time, glide_dual (non calculé ici) ; puis nos trois champs.
        return RobustSolution(self.prob.status, tgo, t, self.r.value.copy(), self.v.value.copy(), z.copy(),
                              self.u.value.copy(), self.s.value.copy(), fuel, fuel / (self.p.alpha * self.p.F),
                              float(abs(self.r.value[-1, 0] - self.xt.value)), float(self.slack.value), st, None,
                              risk=float(self.risk.value), cost=float(self.prob.value), xi0=float(xi0))


# ==================================================================================================================
# LE CONTRÔLEUR : la cascade de la Question 4, avec les trois phases de gestion du vent
# ==================================================================================================================
class WindRobustController(CascadeController):
    """Cascade de la Question 4 + gestion du vent en trois phases (voir le résumé en tête de fichier).

    Attributs hérités de CascadeController (lander/controllers.py) utilisés ici :
      self.p, self.g          paramètres du lander et gravité du scénario ;
      self.N                  nombre d'intervalles du SOCP (20) ;
      self.plan               plan courant (dict : tgo, dt, p, v, a, m, sol...) ou None ;
      self.k, self.k_since    nombre de pas depuis le début de l'épisode / depuis l'adoption du plan courant
                              (k_since / 50 = temps écoulé dans le plan, en secondes) ;
      self.phase              "POWERED" (on suit un plan), "TERMINAL" (sous la porte), "TOUCHDOWN" (posé) ;
      self.hover_possible     False en basse gravité (la poussée minimale dépasse le poids) ;
      self.kp_trk, self.kd_trk gains PD de suivi (x, y) ; self.theta_trk : inclinaison maximale commandée ;
      self.tgo_grid, self.tgo_margin   grille de la recherche du temps de vol et marge de 10 % ;
      self.derate, self.y_gate, self.v_gate, self.x_tol   réserve de poussée et définition de la porte ;
      self.G                  le SOCP (remplacé ici par RobustLanderSOCP) ; self.log : journal de l'épisode.
    Méthodes héritées SANS changement : act (à travers super().act), _step, _replan, _go_terminal, _throttle,
    _to_action, _attitude.
    Méthodes redéfinies : reset, act, _solve, _search, _reference, _track, _fallback, _terminal."""
    name = "LCvx cascade + wind robustification"

    def __init__(self, spec, use_drift=True, use_ff=True, use_risk=True, use_walls=True, n_walls=4, w_haz=150.0,
                 gamma_ff=0.7, lam=0.05, sigma_rb=(0.12, 0.05), est_frac=0.25, d_active=4.0, n_scp=2,
                 theta_plan_deg=25.0, gamma_deg=30.0, glide_from=None, term_gains=(0.8, 0.5, 2.5), **kw):
        # Réglages (valeurs et justification au point 7 du résumé) :
        #   use_drift / use_ff / use_risk / use_walls : activer la Phase 1 / 2 / 3 / les murs (étude d'ablation) ;
        #   n_walls : emplacements de murs par nœud ; w_haz : poids W_haz du risque ;
        #   gamma_ff : gain du feedforward de rafale ; lam : gain lambda de l'estimateur ;
        #   sigma_rb : écarts-types du corps rigide (calibrés) ; est_frac : variance initiale de la rafale inconnue ;
        #   d_active : portée des murs (m) ; n_scp : itérations de relinéarisation des murs après la recherche ;
        #   theta_plan_deg, gamma_deg : inclinaison max et angle du glide slope du SOCP (comme la Question 4) ;
        #   glide_from : premier nœud du glide slope (None = choix automatique ci-dessous) ;
        #   term_gains : (kx, vmax, kd) du maintien latéral sous la porte ; **kw : autres réglages de la cascade.
        self.spec, self.wind = spec, spec["wind"]
        self.use_drift, self.use_ff, self.use_risk, self.use_walls = use_drift, use_ff, use_risk, use_walls
        self.gamma_ff, self.n_scp, self.term_gains = gamma_ff, n_scp, term_gains
        # L'estimateur doit exister AVANT super().__init__, car le constructeur de la classe mère appelle self.reset(),
        # et notre reset() réinitialise l'estimateur.
        self.gust = OUGustEstimator(spec, lam=lam)
        # Construit la cascade de la Question 4 (gains, porte, réserve de poussée, SOCP de base...).
        super().__init__(gravity=spec["gravity"], spec=spec, theta_plan_deg=theta_plan_deg, gamma_deg=gamma_deg, **kw)
        # Limites de la carte, calculées exactement comme dans CascadeController : x dans la carte (avec 0.8 m de
        # marge) et y sous le haut de la carte + 1 m, dans le repère du pad (c = abscisse du centre du pad).
        c = 0.5 * (spec["pad"][0] + spec["pad"][1])
        lim = ((0.8 - c, 20.0 - 0.8 - c), 400 / 30.0 + 1.0 - (spec["pad"][2] + 0.6))
        # Glide slope : en Phase 3, ce sont les demi-plans du terrain (HazardWalls) qui protègent du sol. Le cône de
        # 30 deg de la Question 4, imposé dès le nœud 1, rend le SOCP infaisable quand le départ est loin et bas
        # (S10 : x = -13.5 m, y = 7.4 m) ; glide_from = N le désactive (y_N - y_N >= 0). On le garde (réglage de la
        # Question 4) sans murs (ablation) et quand le lander ne peut pas planer : en basse gravité le plan contient
        # des arcs balistiques et le cône garde des approches courtes (S07, 20 graines : 95 % de succès avec le cône
        # contre 80 % sans, à cause de vols qui dépassent 20 s).
        can_hover = self.p.rho1 / self.p.m_dry < abs(self.g)     # poussée minimale < poids (lander vide) ?
        glide_from = (self.N if use_walls and can_hover else 1) if glide_from is None else glide_from
        # Le SOCP de la cascade (self.G) est remplacé par sa version robuste, avec les mêmes réglages que la
        # Question 4. Le cisaillement n'est mis dans la dynamique que si la Phase 1 est active.
        self.G = RobustLanderSOCP(self.p, self.N, self.g, n_walls=n_walls, w_haz=w_haz, mode="fuel",
                                  terminal="gate", theta_max_deg=theta_plan_deg, gamma_deg=gamma_deg,
                                  glide_from=glide_from,
                                  thrust_frac=self.derate, y_gate=self.y_gate, v_gate=self.v_gate, x_tol=self.x_tol,
                                  map_limits=lim, shear=self.wind["shear"] if use_drift else 0.0)
        self.walls = HazardWalls(spec, n_walls, d_active)                         # interface 5.1 (murs)
        # Tube de la Phase 3.1, calculé une fois avec les gains PD HORIZONTAUX de la cascade (indice 0 = axe x).
        self.tube = WindTube(self.wind, self.kp_trk[0], self.kd_trk[0], sigma_rb, est_frac)

    # ------------------------------------------------------------------ début d'épisode et pas de contrôle
    def reset(self):
        """Appelée au début de chaque épisode (et une fois par le constructeur de la classe mère)."""
        super().reset()                              # plan = None, compteurs à 0, journal vide...
        self.gust.reset()                            # xi_hat = 0, pas d'état précédent
        self.a_prev, self.xi_hat = np.zeros(2), 0.0  # aucune action appliquée encore ; rafale supposée nulle
        # Journaux supplémentaires : écart de suivi (x, y), risque et slack de chaque plan, rafale estimée.
        self.log.update(track_xy=[], risk=[], slack=[], xi_hat=[])

    def act(self, obs, info=None):
        """Un pas de contrôle (toutes les 0.02 s).
        1. Phase 2 : mise à jour de l'estimation de rafale, à partir de l'état mesuré et de l'action du pas précédent.
        2. La cascade de la classe mère fait le reste (phase de vol, replanning, suivi, attitude) ; elle appellera nos
           versions de _solve, _search, _track, _terminal, _fallback.
        3. On mémorise l'action renvoyée : l'estimateur en aura besoin au pas suivant."""
        if info is not None:                         # info["state"] = (x, y, vx, vy, theta, omega, m) en SI
            self.xi_hat = self.gust.update(info["state"], self.a_prev)
        self.log["xi_hat"].append(self.xi_hat)
        out = super().act(obs, info)
        self.a_prev = np.asarray(out, float)
        return out

    # ------------------------------------------------------------------ planification robuste
    def _reference_path(self, s, tgo, ref=None):
        """Trajectoire de référence (N+1, 2) autour de laquelle on linéarise les obstacles (module 5.1).

        Le SOCP ne peut pas « voir » un obstacle polygonal directement : HazardWalls le remplace par un demi-plan
        tangent au point le plus proche d'une trajectoire de référence. Plus cette référence ressemble au futur plan,
        meilleurs sont les demi-plans.
          * ref = (plan, temps écoulé dans ce plan) : on reprend l'ancien plan. On le lit (interpolation linéaire) aux
            instants des nouveaux nœuds, puis on décale le début pour qu'il parte de la position MESURÉE ; le décalage
            décroît linéairement jusqu'à 0 au dernier nœud (la fin du plan reste à la porte).
          * ref = None (premier plan de l'épisode) : chemin en « L », transit horizontal à l'altitude actuelle jusqu'à
            la verticale du pad, puis descente verticale vers la porte. Il passe au-dessus des obstacles dans la
            plupart des scénarios, donc les demi-plans obtenus poussent le plan par-dessus plutôt qu'à travers."""
        N = self.N
        if ref is not None:
            plan, elapsed = ref
            tt = np.arange(len(plan["p"])) * plan["dt"]             # instants des nœuds de l'ancien plan
            tk = elapsed + np.arange(N + 1) * tgo / N               # instants des nouveaux nœuds, dans l'ancien plan
            # np.interp lit l'ancien plan aux nouveaux instants (et garde le dernier point au-delà de sa fin).
            path = np.column_stack([np.interp(tk, tt, plan["p"][:, 0]), np.interp(tk, tt, plan["p"][:, 1])])
            return path + np.outer(np.linspace(1.0, 0.0, N + 1), s[:2] - path[0])
        a, gate = np.asarray(s[:2], float), np.array([0.0, self.y_gate])    # départ et porte (repère du pad)
        corner = np.array([0.0, max(a[1], self.y_gate)])            # coin du « L » : au-dessus du pad
        l1, l2 = np.linalg.norm(corner - a), np.linalg.norm(gate - corner)   # longueurs des deux branches
        arc = np.linspace(0.0, 1.0, N + 1) * (l1 + l2)              # N+1 points également espacés le long du L
        f1 = np.clip(arc / max(l1, 1e-9), 0, 1)[:, None]            # avancement sur la 1re branche (0 -> 1)
        f2 = np.clip((arc - l1) / max(l2, 1e-9), 0, 1)[:, None]     # avancement sur la 2e branche (0 -> 1)
        return np.where(arc[:, None] <= l1, a + f1 * (corner - a), corner + f2 * (gate - corner))

    def _solve(self, s, tgo, certify=True, ref=None):
        """Un SOCP robuste pour un temps de vol tgo, depuis l'état mesuré s = (x, y, vx, vy, theta, omega, m).
        Enchaînement : référence -> murs (5.1) -> écarts-types projetés (3.1, 3.2) -> SOCP (Phase 1 + 3.3).
        Même contrat que CascadeController._solve (appelée par _replan et _search de la classe mère) : retourne un
        plan (dict) ou None si le SOCP échoue, ou si certify=True et la relaxation n'est pas sans perte."""
        self.log["n_solves"] += 1
        N = self.N
        t = np.arange(N + 1) * tgo / N                           # instants des nœuds, en secondes depuis maintenant
        if ref is None and self.plan is not None:                # par défaut, on linéarise autour du plan en cours
            ref = (self.plan, self.k_since / 50)
        # Module 5.1 : demi-plans (n, b) le long de la référence (None si les murs sont désactivés).
        walls = self.walls(self._reference_path(s, tgo, ref)) if self.use_walls else None
        # Phase 3.1 + 3.2 : écart-type perpendiculaire à chaque mur, à chaque nœud (None si le risque est désactivé).
        sigma = self.tube.sigma_dir(walls[0], t) if (self.use_risk and walls is not None) else None
        # Phase 1 : rafale actuelle et vent constant connu (0 si la Phase 1 est désactivée : le plan ignore le vent).
        xi0 = self.xi_hat if self.use_drift else 0.0
        w0 = known_wind(self.wind, 0.0) if self.use_drift else 0.0     # w_bar + s l/10 ; s y/10 est dans `shear`
        sol = self.G.solve(s[:4], tgo, s[6], wind_const=w0, xi0=xi0, gust_tau=self.wind["gust_tau"], walls=walls,
                           sigma=sigma, allow_off=not self.hover_possible)
        # is_lossless (Question 3.1) : certificat que ||u_k|| = sigma_k à tous les nœuds, donc que le plan est
        # réalisable par le vrai moteur (tolérance 1e-3, comme dans la cascade).
        if sol is None or (certify and not is_lossless(sol, 1e-3)):
            return None
        self.log["risk"].append(sol.risk)
        self.log["slack"].append(sol.slack)
        # Le plan, au format de la cascade (tgo, dt, p, v, a, m, sol), plus trois clés lues par _reference,
        # _wind_feedforward et _search : la rafale xi0 prise en compte, le coût total, et si le plan contient le vent.
        return dict(kind="robust", tgo=tgo, dt=tgo / N, p=sol.p, v=sol.v, a=sol.u, m=np.exp(sol.z), sol=sol,
                    xi0=xi0, cost=sol.cost, wind=self.use_drift)

    def _search(self, s, refine=6):
        """Recherche du temps de vol (comme dans la Question 4) quand il n'y a pas de plan utilisable.
        1. On résout le SOCP pour chaque temps de vol de la grille self.tgo_grid (1 à 9 s).
        2. Section dorée (golden_section, refine itérations) autour du meilleur point de la grille.
        Différence avec la Question 4 : on compare le COÛT TOTAL (21), pas seulement le carburant : un vol un peu plus
        long qui passe plus loin des murs peut être préférable.
        3. On refait le meilleur plan avec 10 % de temps en plus (marge pour les rotations du corps, comme la
           Question 4), en relinéarisant les murs autour de ce meilleur plan : c'est une itération SCP. On recommence
           n_scp fois au total (chaque fois autour du plan obtenu), ou jusqu'à un échec.
        Retourne le meilleur plan (dict) ou None."""
        plans = {}                                       # temps de vol -> plan, pour tous les SOCP réussis

        def f(t):
            """Fonction minimisée par la recherche : coût total du plan de temps de vol t (inf si échec)."""
            plan = self._solve(s, t)
            if plan is None:
                return np.inf
            plans[t] = plan
            return plan["cost"]

        costs = [f(t) for t in self.tgo_grid]            # 1. grille grossière
        if not np.isfinite(np.min(costs)):
            return None                                  # aucun temps de vol faisable : pas de plan
        i = int(np.argmin(costs))
        # 2. section dorée entre les deux voisins du meilleur point de la grille (bornés aux extrémités de la grille).
        golden_section(f, self.tgo_grid[max(i - 1, 0)], self.tgo_grid[min(i + 1, len(self.tgo_grid) - 1)],
                       max_iter=refine)
        best = min(plans.values(), key=lambda p: p["cost"])      # meilleur plan parmi TOUS ceux évalués
        tgo = best["tgo"] * self.tgo_margin                      # 3. +10 % de temps
        for _ in range(self.n_scp):
            new = self._solve(s, tgo, ref=(best, 0.0))           # murs relinéarisés autour de `best`
            if new is None:
                break                                            # en cas d'échec, on garde le dernier plan valide
            best = new
        return best

    # ------------------------------------------------------------------ suivi du plan + feedforward (Phase 2)
    def _reference(self, s):
        """Position, vitesse et accélération de poussée PLANIFIÉES à l'instant courant.

        Le SOCP ne donne le plan qu'aux nœuds (toutes les dt ~ 0.1-0.4 s) ; le suivi tourne à 50 Hz. Entre deux nœuds
        on intègre exactement la même dynamique que le SOCP sur l'intervalle i : accélération de poussée u_i
        constante + gravité + vent connu à l'altitude du nœud y_i (Phase 1) + dérive partielle de la rafale attendue.
        Sans ce modèle de vent, la référence et le plan ne correspondraient pas, et la loi PD combattrait le vent que
        le plan a justement prévu. Retourne (i, p_ref, v_ref, a) avec a = u_i (accélération de poussée planifiée)."""
        P = self.plan
        t = self.k_since / 50                            # temps écoulé dans le plan (s)
        i = min(int(t / P["dt"]), self.N - 1)            # intervalle courant [t_i, t_i+1)
        tau = min(t - i * P["dt"], P["dt"])              # temps écoulé dans cet intervalle
        a = P["a"][i]                                    # accélération de poussée planifiée (constante, ZOH)
        acc = a + np.array([0.0, self.g])                # + gravité
        if P["wind"]:                                    # + vent connu à y_i, si le plan en tient compte
            acc = acc + np.array([known_wind(self.wind, P["p"][i, 1]), 0.0])
        # Dérive de la rafale entre t_i et t_i + tau (formule (4)-(5) avec une durée partielle tau).
        dv, dx = expected_gust_drift(P["xi0"], self.wind["gust_tau"], i * P["dt"], tau)
        # Intégration exacte à accélération constante, plus la dérive de rafale (horizontale uniquement).
        v_ref = P["v"][i] + tau * acc + np.array([float(dv), 0.0])
        p_ref = P["p"][i] + tau * P["v"][i] + 0.5 * tau * tau * acc + np.array([float(dx), 0.0])
        return i, p_ref, v_ref, a

    def _wind_feedforward(self, s, y_ref=None, t_rel=0.0):
        """Correction horizontale de la Phase 2, équation (13) : -gamma_ff Delta_xi - Delta_w_det (m/s^2).

        Le plan a déjà prévu une partie du vent ; il ne faut compenser que ce qu'il n'a PAS prévu :
          * Delta_xi = xi_hat(t) - xi_plan(t) : la « surprise » de rafale. Le plan a supposé une rafale
            xi_plan(t) = xi_hat_0 e^{-(t - t_plan)/tau} (12) ; ce qui dépasse vient d'une nouvelle rafale. On n'en
            compense que gamma_ff = 70 %, car l'estimation est bruitée.
          * Delta_w_det = w_det(y) - w_det(y_ref) : le plan a prévu le vent connu à l'altitude PLANIFIÉE ; avec le
            cisaillement, si le lander est plus haut ou plus bas que prévu, il subit un autre vent.
        Avec un plan (y_ref donné, appel depuis _track) : comme ci-dessus. Sans plan (y_ref = None : descente terminale,
        mode de secours) : xi_plan = 0 et w_plan = 0, on compense tout le vent estimé. Sans Phase 1 (le plan ignore
        le vent), w_plan = 0 aussi : le feedforward compense alors tout le vent connu."""
        if not self.use_ff:
            return 0.0                                   # Phase 2 désactivée (ablation) : aucune correction
        planned = y_ref is not None and self.plan is not None
        xi_plan = self.plan["xi0"] * math.exp(-t_rel / self.wind["gust_tau"]) if planned else 0.0
        w_plan = known_wind(self.wind, y_ref) if (planned and self.plan["wind"]) else 0.0
        d_xi = self.xi_hat - xi_plan                                 # surprise de rafale
        d_w = float(known_wind(self.wind, s[1])) - float(w_plan)     # écart de cisaillement (et vent non planifié)
        # Signe moins : si le vent pousse vers +x plus que prévu, on commande une accélération vers -x.
        return -self.gamma_ff * d_xi - d_w

    def _track(self, s):
        """Loi de suivi en phase POWERED, équations (13)-(14) :
            a_cmd = u_plan + Kp (r_ref - r) + Kd (v_ref - v)  +  [feedforward de vent, 0].
        L'accélération commandée a_cmd est ensuite transformée en poussée et en inclinaison par _to_action."""
        i, p_ref, v_ref, a = self._reference(s)
        self.log["track_err"].append(float(np.linalg.norm(p_ref - s[:2])))
        self.log["track_xy"].append(s[:2] - p_ref)                   # écart (x, y), pour calibrer sigma_rb
        # Loi PD de la Question 4 (gains différents en x et en y, multiplication terme à terme)...
        a_cmd = a + self.kp_trk * (p_ref - s[:2]) + self.kd_trk * (v_ref - s[2:4])
        # ... plus la correction de vent de la Phase 2, sur l'axe horizontal seulement.
        a_cmd[0] += self._wind_feedforward(s, y_ref=p_ref[1], t_rel=self.k_since / 50)
        # Basse gravité, recopié de la cascade : si le plan coupe presque le moteur (vol balistique), on oriente déjà
        # le lander vers la direction de la prochaine poussée, pour ne pas perdre de temps à tourner ensuite.
        theta_ff = None
        if not self.hover_possible and np.linalg.norm(a) < 0.5 * self.a_min(s[6]):
            nxt = [k for k in range(i, self.N) if np.linalg.norm(self.plan["a"][k]) > 1e-3]
            ak = self.plan["a"][nxt[0]] if nxt else np.array([0.0, 1.0])
            theta_ff = np.arctan2(-ak[0], ak[1])         # inclinaison qui aligne l'axe du moteur sur ak
        return self._to_action(s, a_cmd, theta_ff=theta_ff)

    def _fallback(self, s):
        """Mode de secours (aucun plan valide), recopié de la cascade : on vise une vitesse qui ramène vers le pad et
        fait descendre doucement ; seul ajout, le feedforward de vent (sans plan : tout le vent estimé)."""
        v_ref = np.array([np.clip(-0.5 * s[0], -1.5, 1.5), -np.clip(0.5 * s[1], 0.6, 3.0)])   # vitesse visée
        a_cmd = np.array([0.0, -self.g]) + np.array([1.5, 2.0]) * (v_ref - s[2:4])          # compense g + suivi
        a_cmd[0] += self._wind_feedforward(s)
        engine = self.hover_possible or s[3] < v_ref[1] - 0.3    # basse gravité : moteur seulement si chute trop rapide
        return self._to_action(s, a_cmd, engine=engine)

    def _terminal(self, s):
        """Descente terminale sous la porte (y < y_gate + 0.1 m), à vitesse verticale constante -v_gate.
        Il n'y a plus de plan ni de tube : c'est cette loi qui doit garder le lander sur l'axe du pad.
        Horizontalement, on vise la vitesse v_ref,x = -kx x (bornée à +-vmax) qui ramène vers x = 0, avec un gain kd ;
        on ajoute le feedforward de vent. Le maintien latéral de la Question 4 (kx = 0.3, vmax = 0.3, kd = 2) est trop
        mou dans les couloirs étroits ; term_gains = (0.8, 0.5, 2.5) le raffermit (20 graines : S03 85 -> 100 %,
        S05 60 -> 70 %, S08 55 -> 65 % de succès)."""
        kx, vmax, kd = self.term_gains
        v_ref = np.array([np.clip(-kx * s[0], -vmax, vmax), -self.v_gate])                  # vitesse visée
        a_cmd = np.array([0.0, -self.g]) + np.array([kd, 3.0]) * (v_ref - s[2:4])          # compense g + suivi
        a_cmd[0] += self._wind_feedforward(s)
        engine = self.a_min(s[6]) < -self.g or s[3] < -self.v_gate      # ne peut pas planer : moteur pulsé
        return self._to_action(s, a_cmd, engine=engine)


# ==================================================================================================================
# FABRIQUES : la politique soumise et les variantes de l'étude d'ablation (Question 5.3)
# ==================================================================================================================
# lander.evaluate appelle make_policy(spec) quand on lui donne ce fichier ; avec la syntaxe
# --policy submissions.wind_robust:make_no_risk, il appelle la fabrique nommée.
def make_policy(spec):
    return WindRobustController(spec)                # version complète


def make_no_drift(spec):            # sans Phase 1 : le plan ignore le vent (le feedforward compense tout)
    return WindRobustController(spec, use_drift=False)


def make_no_ff(spec):               # sans Phase 2 : PD seul autour du plan
    return WindRobustController(spec, use_ff=False)


def make_no_risk(spec):             # sans Phase 3 : murs durs sans élargissement probabiliste
    return WindRobustController(spec, use_risk=False)


def make_no_walls(spec):            # sans évitement d'obstacles (ni murs, ni risque ; glide slope de la Question 4)
    return WindRobustController(spec, use_walls=False)


# ==================================================================================================================
# OUTILS HORS LIGNE (Section 7.5) : calibration et vérification du tube. Ils utilisent le simulateur directement,
# ce qui est permis hors ligne sur les graines d'entraînement ; la politique elle-même n'y accède jamais.
# ==================================================================================================================
def calibrate_sigma_rb(sids=("S01", "S02", "S04", "S06", "S09"), seeds=range(10), **kw):
    """Étape 2 de la Section 7.5 : mesurer l'écart de suivi qui ne vient PAS du vent (retard d'attitude, couple de
    turbulence, bruit des moteurs). On rejoue des scénarios avec le vent moyen, le cisaillement et les rafales mis à
    zéro (le couple de turbulence est conservé : il fait partie de sigma_rb), et on mesure l'écart quadratique moyen
    entre la position réelle et la référence. Retourne (sigma_rb,x, sigma_rb,y), avec un plancher de 5 cm."""
    from dataclasses import replace                  # imports locaux : inutiles pour la politique elle-même
    from lander.env import FuelLander
    from lander.evaluate import run_episode
    from lander.scenarios import SCENARIOS
    errs = []
    for sid in sids:
        sc = SCENARIOS[sid]
        # Scenario et Wind sont des dataclasses figées : `replace` en fait une copie modifiée (vent coupé).
        calm = replace(sc, wind=replace(sc.wind, mean=0.0, shear=0.0, gust_std=0.0))
        env, pol = FuelLander(calm), WindRobustController(calm.public_dict(), use_risk=False, **kw)
        for seed in seeds:
            run_episode(env, pol, seed)              # vole un épisode ; pol.log["track_xy"] contient les écarts
            errs += pol.log["track_xy"]              # à récupérer AVANT l'épisode suivant (reset vide le journal)
        env.close()
    e = np.array(errs)                               # (nombre de pas suivis, 2)
    return np.maximum(np.sqrt(np.mean(e ** 2, axis=0)), 0.05)      # RMS par axe, plancher 5 cm


def check_tube(sids=("S01", "S03", "S05", "S10"), seeds=range(5), **kw):
    """Vérification : compare l'écart-type horizontal prédit par le tube (à 0.5 s, la période de replanning, et en
    régime stationnaire) avec l'écart de suivi réellement mesuré en vol, avec et sans feedforward. Comme le tube
    ignore le feedforward (gamma_ff = 0), il devrait majorer la dispersion mesurée avec feedforward."""
    from lander.env import FuelLander
    from lander.evaluate import run_episode
    from lander.scenarios import SCENARIOS
    for sid in sids:
        sc = SCENARIOS[sid]
        row = []
        for use_ff in (False, True):                 # sans puis avec la Phase 2
            env, pol = FuelLander(sc), WindRobustController(sc.public_dict(), use_ff=use_ff, **kw)
            errs, outs = [], []
            for seed in seeds:
                res, _, _ = run_episode(env, pol, seed)
                errs += pol.log["track_xy"]
                outs.append(res["outcome"])
            env.close()
            row.append((np.std(np.array(errs)[:, 0]) if errs else np.nan, outs))    # écart-type mesuré en x
        sx_half, sx_inf = pol.tube.sigma_xy(0.5)[0], pol.tube.sigma_xy(15.0)[0]    # prédictions du tube
        print(f"{sid}: predicted sigma_x {sx_half:.3f} m (0.5 s) / {sx_inf:.3f} m (stationary) | measured "
              f"{row[0][0]:.3f} m without feedforward, {row[1][0]:.3f} m with feedforward | outcomes {row[1][1]}")


# Exécution directe du fichier : python submissions/wind_robust.py [calibrate | tube]
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if cmd == "calibrate":
        print("sigma_rb (x, y) =", calibrate_sigma_rb())
    elif cmd == "tube":
        check_tube()
    else:
        raise SystemExit("usage: python submissions/wind_robust.py [calibrate | tube]")
