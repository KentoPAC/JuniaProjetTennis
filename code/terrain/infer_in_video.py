import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import mode
import json

from terrain.tracknet import BallTrackerNet
from terrain.postprocess import postprocess, refine_kps
from terrain.homography import get_trans_matrix, refer_kps


def read_video(path_video):
    """Lit la vidéo et renvoie les frames et le fps"""
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(imgs, fps, path_output_video):
    """Écrit une vidéo à partir d'une liste de frames"""
    if not imgs:
        print("⚠ Aucune frame à écrire !")
        return
    h, w = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path_output_video, fourcc, fps, (w, h))
    for f in imgs:
        out.write(f)
    out.release()


def infer_terrain(
    model_path: str,
    video_path: str,
    output_json: str,
    duration: float = 5.0,
    use_refine_kps: bool = False,
    use_homography: bool = False,
) -> list:
    """
    Détecte les points clés du terrain et génère :
      - Un JSON de points les plus fréquents
      - Une vidéo annotée avec ces points
    """
    # Chargement modèle
    model = BallTrackerNet(out_channels=15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Terrain] Device: {device}")

    # Lecture vidéo
    frames, fps = read_video(video_path)
    all_points = []
    num_frames = int(fps * duration)
    OUTPUT_W, OUTPUT_H = 640, 360

    # Détection sur premières frames
    for img in tqdm(frames[:num_frames], desc="Terrain Detection"):
        resized = cv2.resize(img, (OUTPUT_W, OUTPUT_H))
        inp = torch.tensor(
            np.rollaxis(resized.astype(np.float32) / 255.0, 2, 0)
        ).unsqueeze(0)
        out = model(inp.to(device).float())[0]
        pred = torch.sigmoid(out).detach().cpu().numpy()

        pts = []
        for k in range(14):
            heat = (pred[k] * 255).astype(np.uint8)
            x, y = postprocess(heat, low_thresh=170, max_radius=25)
            if use_refine_kps and k not in [8, 12, 9] and x and y:
                x, y = refine_kps(img, int(y), int(x))
            pts.append((x, y))

        if use_homography:
            M = get_trans_matrix(pts)
            if M is not None:
                arr = np.array([p for p in pts if p], dtype=np.float32).reshape(
                    -1, 1, 2
                )
                trans = cv2.perspectiveTransform(arr, M)
                pts = [tuple(np.squeeze(p)) for p in trans]
        all_points.append(pts)

    # Calcul du mode
    most_freq = []
    for i in range(14):
        xs = [f[i][0] for f in all_points if f[i][0] is not None]
        ys = [f[i][1] for f in all_points if f[i][1] is not None]
        if xs and ys:
            mx = mode(xs, keepdims=False).mode
            my = mode(ys, keepdims=False).mode
            most_freq.append((int(mx), int(my)))
        else:
            most_freq.append(None)

    # Homographie finale sur point moyens
    if use_homography:
        arr = np.array([p for p in most_freq if p], dtype=np.float32).reshape(-1, 1, 2)
        M = get_trans_matrix(most_freq)
        if M is not None:
            trans = cv2.perspectiveTransform(arr, M)
            most_freq = [tuple(np.squeeze(p)) for p in trans]

    # Écriture JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as jf:
        json.dump(
            {"points": [({"x": p[0], "y": p[1]} if p else None) for p in most_freq]},
            jf,
            indent=4,
        )
    print(f"[Terrain] JSON saved: {output_json}")

    # Génération vidéo annotée
    annotated_path = os.path.splitext(output_json)[0] + "_annotated.mp4"
    frames_upd = []
    for img in frames:
        for idx, p in enumerate(most_freq):
            if p:
                cv2.circle(img, (p[0], p[1]), 5, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    f"P{idx}",
                    (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
        frames_upd.append(img)
    write_video(frames_upd, fps, annotated_path)
    print(f"[Terrain] Video saved: {annotated_path}")

    return most_freq
