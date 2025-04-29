import json
from functools import lru_cache

def detection_fautes(court_json: str,
                     ball_x: float,
                     ball_y: float,
                     player: str = "bottom_player") -> bool:
    """
    Détermine si la balle est IN (True) ou OUT (False).

    Paramètres
    ----------
    court_json : str
        Chaîne JSON contenant la clé ``"points"`` avec les 14 sommets du court.
    ball_x, ball_y : float
        Coordonnées (pixels) du rebond détecté.
    player : str
        - ``"bottom_player"`` : le joueur filmé en bas de l’image  
        - ``"top_player"``    : le joueur filmé en haut de l’image  
        - ``"all"``           : on ignore le sens de jeu et l’on valide
          dès que la balle retombe dans le court de simple, d’un côté
          ou de l’autre du filet.

    Retour
    ------
    bool
        True  → balle bonne  
        False → faute
    """

    # ------------------------------------------------------------------
    # 1) Récupération / mise en cache des sommets et de quelques repères
    # ------------------------------------------------------------------
    X4, Y4, X5, Y5, X6, Y6, X7, Y7, net_y = _get_court_constants(court_json)

    # ------------------------------------------------------------------
    # 2) Fonctions limites dépendant de la perspective
    # ------------------------------------------------------------------
    def singles_x_left(y: float) -> float:
        return X4 + (X5 - X4) * (y - Y4) / (Y5 - Y4)

    def singles_x_right(y: float) -> float:
        return X6 + (X7 - X6) * (y - Y6) / (Y7 - Y6)

    # ------------------------------------------------------------------
    # 3) Tests géométriques
    # ------------------------------------------------------------------
    xl = singles_x_left(ball_y)
    xr = singles_x_right(ball_y)

    # — Dans quel camp (nord / sud) ?
    north_side = ball_y <= net_y

    # — Balle dans le trapèze complet « simple » ?
    in_singles = (xl <= ball_x <= xr) and (Y4 <= ball_y <= Y5)

    # — Balle dans la moitié adverse seulement ?
    in_half = xl <= ball_x <= xr and (
        (north_side and Y4 <= ball_y <= net_y) or
        (not north_side and net_y <= ball_y <= Y5)
    )

    # ------------------------------------------------------------------
    # 4) Décision selon le mode demandé
    # ------------------------------------------------------------------
    if player == "all":
        # On valide dès qu’elle retombe dans le court, quel que soit le côté
        return in_singles

    if north_side and player == "bottom_player":
        return in_half          # bonne balle pour bottom
    if (not north_side) and player == "top_player":
        return in_half          # bonne balle pour top

    # Si la balle retombe dans son propre camp → faute
    return False


# ----------------------------------------------------------------------
# Mise en cache des constantes du terrain
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def _get_court_constants(court_json: str):
    data = json.loads(court_json)
    pts = data["points"]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]

    # Sommets clés (convention des 14 points)
    X4, Y4 = xs[4],  ys[4]   # baseline nord-gauche (simple)
    X6, Y6 = xs[6],  ys[6]   # baseline nord-droite (simple)
    X5, Y5 = xs[5],  ys[5]   # baseline sud-gauche (simple)
    X7, Y7 = xs[7],  ys[7]   # baseline sud-droite (simple)

    # Ligne du filet : 10 % au-dessus du milieu (P12-P13)
    mid_y = (ys[12] + ys[13]) / 2
    net_y = mid_y - 0.10 * (ys[13] - ys[12])

    return X4, Y4, X5, Y5, X6, Y6, X7, Y7, net_y
