#!/usr/bin/env python3
import json
import os


def main():
    # Chemin vers le fichier balle.jsonl dans le même répertoire
    script_dir = os.path.dirname(__file__)
    jsonl_path = os.path.join(script_dir, "balle.jsonl")

    frames = []
    ys = []
    # Lecture des données
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            frame = rec.get("frame")
            dets = rec.get("detections", [])
            if dets:
                y = dets[0].get("Ball_Y")
                if y is not None:
                    frames.append(frame)
                    ys.append(y)

    if not frames:
        print("Aucune donnée Ball_Y trouvée dans balle.jsonl")
        return

    # Détection de points où Y atteint un maximum local (rebond)
    bounce_idxs = []
    for i in range(1, len(ys) - 1):
        prev_dy = ys[i] - ys[i - 1]
        next_dy = ys[i + 1] - ys[i]
        # pic local: balle atteint point le plus bas à l'image (Y max)
        if prev_dy > 0 and next_dy < 0:
            bounce_idxs.append(i)

    # Affichage des résultats
    if bounce_idxs:
        print("Rebonds détectés:")
        for idx in bounce_idxs:
            print(f"  Frame {frames[idx]} : Y = {ys[idx]}")
    else:
        print("Aucun rebond détecté.")

    # Optionnel: utilisation de scipy.signal.find_peaks pour comparaison
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(ys, distance=5)
        print("\nPeaks détectés par find_peaks:")
        for idx in peaks:
            print(f"  Frame {frames[idx]} : Y = {ys[idx]}")
    except ImportError:
        pass

    # Filtrer selon les lignes de fond détectées (terrain) si disponible
    # Cherche le JSON de points terrain (data/terrain_points.json ou terrain_points.json)
    terrain_json = os.path.join(script_dir, "..", "data", "terrain_points.json")
    if not os.path.exists(terrain_json):
        alt = os.path.join(script_dir, "..", "terrain_points.json")
        if os.path.exists(alt):
            terrain_json = alt
    if os.path.exists(terrain_json):
        try:
            with open(terrain_json, "r", encoding="utf-8") as tf:
                tj = json.load(tf)
            pts = tj.get("points", [])
            # Indices 0-1: baseline proche, 2-3: baseline éloignée
            y_top = y_bot = None
            if len(pts) >= 4:
                p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
                if p0 and p1:
                    y_top = (p0["y"] + p1["y"]) / 2
                if p2 and p3:
                    y_bot = (p2["y"] + p3["y"]) / 2
            tol = 30  # tolérance en pixels autour de la ligne
            court_filtered = [
                i
                for i in bounce_idxs
                if (y_top is not None and abs(ys[i] - y_top) <= tol)
                or (y_bot is not None and abs(ys[i] - y_bot) <= tol)
            ]
            print(
                f"Rebonds après filtre terrain : {len(court_filtered)}/{len(bounce_idxs)} retenus (tol={tol})"
            )
            if court_filtered:
                bounce_idxs = court_filtered
        except Exception as e:
            print(f"Erreur filtre terrain: {e}")
    else:
        # Aucun JSON terrain trouvé : fallback sur seuil global bas de l'image
        print(
            f"Aucun fichier terrain JSON trouvé ({terrain_json}), utilisation du seuil global."
        )
        y_max = max(ys)
        threshold_ratio = 0.8
        y_thr = threshold_ratio * y_max
        filtered = [i for i in bounce_idxs if ys[i] >= y_thr]
        print(
            f"Rebonds filtrés (Y>={y_thr:.1f}): {len(filtered)}/{len(bounce_idxs)} retenus"
        )
        bounce_idxs = filtered

    # Génération de la page HTML avec graphique Plotly
    try:
        import plotly.graph_objects as go

        # Préparer la figure
        html_path = os.path.join(script_dir, "statstest.html")
        fig = go.Figure()
        # Courbe de Y vs Frame
        fig.add_trace(go.Scatter(x=frames, y=ys, mode="lines", name="Y vs Frame"))
        # Marqueurs de rebonds
        fig.add_trace(
            go.Scatter(
                x=[frames[i] for i in bounce_idxs],
                y=[ys[i] for i in bounce_idxs],
                mode="markers",
                marker=dict(color="red", size=8),
                name="Rebonds",
            )
        )
        # Inverser axe Y pour correspondance image
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title="Position Y de la balle et détection de rebonds",
            xaxis_title="Frame",
            yaxis_title="Y (pixels)",
            margin=dict(l=40, r=40, t=80, b=40),
        )
        # Sauvegarde et ouverture
        fig.write_html(html_path)
        print(f"Page HTML générée : {html_path}")
        try:
            import webbrowser

            webbrowser.open(f"file://{html_path}")
        except Exception:
            pass
    except ImportError:
        print("Plotly non installé, impossible de générer la page HTML.")


if __name__ == "__main__":
    main()
