import json
import os

import plotly.graph_objects as go


def load_ball_data(jsonl_path):
    """
    Charge les données de la balle depuis un fichier JSONL.

    Args:
        jsonl_path: Chemin vers le fichier JSONL contenant les données de la balle

    Returns:
        tuple: (frames, ys) - listes des numéros de frames et des positions Y
    """
    frames = []
    ys = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if not data["no_detection"] and data["detections"]:
                frames.append(data["frame"])
                ys.append(data["detections"][0]["Ball_Y"])

    return frames, ys


def detect_rebounds(
    frames, ys, window_size=5, min_direction_change=4, min_frames_between_rebounds=10
):
    """
    Détecte les rebonds en analysant les changements de direction verticale.
    Un rebond est détecté quand la balle descend puis remonte.

    Args:
        frames: Liste des numéros de frames
        ys: Liste des positions Y de la balle
        window_size: Taille de la fenêtre d'analyse
        min_direction_change: Changement minimum de direction pour considérer un rebond
        min_frames_between_rebounds: Nombre minimum de frames entre deux rebonds

    Returns:
        List[dict]: Liste des rebonds détectés avec frame et position Y
    """
    rebounds = []
    last_rebound_frame = -float("inf")

    # On a besoin d'au moins window_size * 2 points pour détecter un rebond
    if len(frames) < window_size * 2:
        return rebounds

    # Analyse de chaque point de la trajectoire
    for i in range(window_size, len(frames) - window_size):
        current_frame = frames[i]

        # Vérifier qu'on est assez loin du dernier rebond
        if current_frame - last_rebound_frame < min_frames_between_rebounds:
            continue

        # Extraction des trajectoires avant et après le point
        pre_trajectory = ys[i - window_size : i]
        post_trajectory = ys[i : i + window_size]

        # S'assurer qu'on a assez de points consécutifs
        if len(pre_trajectory) < 3 or len(post_trajectory) < 3:
            continue

        # Calcul des tendances (descente/montée) avec plus de poids sur les points proches
        pre_diffs = [y1 - y2 for y1, y2 in zip(pre_trajectory[:-1], pre_trajectory[1:])]
        post_diffs = [
            y1 - y2 for y1, y2 in zip(post_trajectory[:-1], post_trajectory[1:])
        ]

        pre_direction = sum(diff * (idx + 1) for idx, diff in enumerate(pre_diffs))
        post_direction = sum(
            diff * (len(post_diffs) - idx) for idx, diff in enumerate(post_diffs)
        )

        # Calculer la variation totale de position
        total_pre_change = pre_trajectory[-1] - pre_trajectory[0]
        total_post_change = post_trajectory[-1] - post_trajectory[0]

        # Détecter un rebond avec des critères plus flexibles
        if (
            # La tendance générale montre une descente puis une montée
            pre_direction < 0
            and post_direction > 0
            # Le changement de direction est suffisant
            and abs(total_pre_change) > min_direction_change
            and abs(total_post_change) > min_direction_change
            # La position Y est cohérente avec un rebond
            and ys[i] > 160
            # Les changements sont relativement continus
            and all(abs(d) < 50 for d in pre_diffs + post_diffs)
            # Le changement total est significatif
            and abs(total_pre_change) + abs(total_post_change) > 8
        ):
            rebounds.append({"frame": current_frame, "y": ys[i]})
            last_rebound_frame = current_frame

    return rebounds


def plot_rebounds(frames, ys, rebounds):
    """
    Génère un graphique interactif montrant la trajectoire de la balle et les rebonds détectés.

    Args:
        frames: Liste des numéros de frames
        ys: Liste des positions Y
        rebounds: Liste des rebonds détectés
    """
    # Création de la figure
    fig = go.Figure()

    # Ajouter la trajectoire de la balle avec des lignes et des points
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=ys,
            mode="lines+markers",
            name="Trajectoire",
            line=dict(color="blue", width=1),
            marker=dict(size=4, color="blue", opacity=0.5),
        )
    )

    # Ajouter les rebonds
    rebound_frames = [r["frame"] for r in rebounds]
    rebound_ys = [r["y"] for r in rebounds]
    fig.add_trace(
        go.Scatter(
            x=rebound_frames,
            y=rebound_ys,
            mode="markers",
            name="Rebonds",
            marker=dict(symbol="star", size=12, color="red"),
        )
    )

    # Personnalisation du graphique
    fig.update_layout(
        title="Trajectoire de la balle et rebonds détectés",
        xaxis_title="Frame",
        yaxis_title="Position Y",
        showlegend=True,
        # Inverser l'axe Y
        yaxis_autorange="reversed",
    )

    # Ajouter une grille
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    # Sauvegarder en HTML
    output_dir = os.path.dirname(__file__)
    fig.write_html(os.path.join(output_dir, "rebounds_plot.html"))


def main():
    """
    Fonction principale pour tester la détection des rebonds
    """
    # Chemin vers le fichier de données
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "output", "balle", "balle.jsonl"
    )

    # Charger les données de la balle
    frames, ys = load_ball_data(data_path)

    # Détecter les rebonds
    rebounds = detect_rebounds(frames, ys)

    # Afficher les rebonds détectés
    print("Rebonds détectés:")
    for rebound in rebounds:
        print(f"Frame {rebound['frame']}: y={rebound['y']}")

    # Générer le graphique interactif
    plot_rebounds(frames, ys, rebounds)
    print(f"\nNombre total de rebonds détectés: {len(rebounds)}")
    print("Le graphique a été sauvegardé dans 'rebounds_plot.html'")


if __name__ == "__main__":
    main()
