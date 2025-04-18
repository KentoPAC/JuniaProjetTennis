import json
import plotly.graph_objects as go
import webbrowser
import os

def main():
    frames = []
    ys = []

    # Lecture ligne à ligne du JSONL
    with open("balle.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            frame = rec.get("frame")
            # on prend la première détection si elle existe
            dets = rec.get("detections", [])
            if dets:
                y = dets[0].get("Ball_Y")
                if y is not None:
                    frames.append(frame)
                    ys.append(y)

    if not frames:
        print("Aucune donnée Ball_Y trouvée dans data.jsonl")
        return

    # Création du graphique interactif sans pandas
    fig = go.Figure(
        data=go.Scatter(x=frames, y=ys, mode="lines", name="Y vs Frame")
    )

    # Inverser l'axe Y
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title="Position Y de la balle en fonction de la frame",
        xaxis_title="Frame",
        yaxis_title="Y (pixels)",
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Enregistrer et ouvrir
    out = "y_vs_frame.html"
    fig.write_html(out)
    print(f"Ouverture de {out}…")
    webbrowser.open("file://" + os.path.realpath(out))


if __name__ == "__main__":
    main()
