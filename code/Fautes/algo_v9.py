import json
from functools import lru_cache

def detection_fautes(court_json: str,
                     ball_x: float,
                     ball_y: float,
                     player: str = "bottom_player") -> bool:
    """
    Décide si la balle est IN (True) ou OUT (False) pour *player*
    dans un match de simple.

    Paramètres
    ----------
    court_json : str
        Chaîne JSON contenant la clé "points" avec les 14 sommets du court
        (dans l’ordre 0→13 de votre convention).
    ball_x, ball_y : float
        Coordonnées (pixel) du rebond détecté.
    player : str
        "bottom_player" ou "top_player".

    Retour
    ------
    bool
        True  → balle bonne pour le joueur
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
    # 3) Test IN/OUT
    # ------------------------------------------------------------------
    north_side = ball_y <= net_y            # True si la balle est dans le camp nord
    xl = singles_x_left(ball_y)
    xr = singles_x_right(ball_y)

    # Balle à l'intérieur du trapèze correspondant ?
    in_half = xl <= ball_x <= xr and (
        (north_side and Y4 <= ball_y <= net_y) or
        (not north_side and net_y <= ball_y <= Y5)
    )

    # Valide uniquement si la balle rebondit dans le camp adverse
    if north_side and player == "bottom_player":
        return in_half           # bonne balle pour bottom
    if (not north_side) and player == "top_player":
        return in_half           # bonne balle pour top
    return False                 # balle dans son propre camp → faute


# ----------------------------------------------------------------------
# Petite aide : mettre en cache les constantes du terrain pour ne pas
#              tout recalculer à chaque appel.
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def _get_court_constants(court_json: str):
    data = json.loads(court_json)
    pts = data["points"]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]

    # Sommets clés (convention des 14 points)
    X4, Y4 = xs[4],  ys[4]   # baseline nord‑gauche (simple)
    X6, Y6 = xs[6],  ys[6]   # baseline nord‑droite (simple)
    X5, Y5 = xs[5],  ys[5]   # baseline sud‑gauche (simple)
    X7, Y7 = xs[7],  ys[7]   # baseline sud‑droite (simple)

    # Ligne du filet : 10 % au‑dessus du milieu des points 12‑13
    mid_y = (ys[12] + ys[13]) / 2
    net_y = mid_y - 0.10 * (ys[13] - ys[12])

    return X4, Y4, X5, Y5, X6, Y6, X7, Y7, net_y
