from shapely.geometry import Point, Polygon

def get_terrain_points(mode):

    if mode == "double":
        print("🎾 Mode sélectionné : DOUBLE")
        terrain_points = [
            (x0, y0), (x1, y1),  # Ligne du haut
            (x3, y3), (x2, y2)  # Ligne du bas
        ]
    else:
        print("🎾 Mode sélectionné : SIMPLE")
        terrain_points = [
            (x4, y4), (x6, y6),  # Ligne du haut
            (x7, y7), (x5, y5)  # Ligne du bas
        ]

    return Polygon(terrain_points)

def verifier_position_balle(terrain_polygon, balle_x, balle_y):

    balle = Point(balle_x, balle_y)

    if terrain_polygon.contains(balle):
        print("La balle est DANS le terrain.")
        return True
    else:
        print("La balle est HORS du terrain.")
        return False


mode_jeu = input("Mode de jeu ? (simple/double) : ").strip().lower()

# Récupérer les coordonnées du terrain en fonction du mode choisi
terrain_polygon = get_terrain_points(mode_jeu)

# Exemple de balle détectée
balle_x, balle_y = 320, 450  # À remplacer par les coordonnées de détection réelle


verifier_position_balle(terrain_polygon, balle_x, balle_y)
