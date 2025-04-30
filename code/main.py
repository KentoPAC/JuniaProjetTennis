#!/usr/bin/env python3
# main.py : Pipeline détection tennis avec configuration intégrée

# === Configuration - Modifiez ces variables selon vos besoins ===
FIELD_MODEL = "terrain/model_tennis_court_det.pt"  # Chemin vers le modèle TrackNet
INPUT_VIDEO = "../Vidéos/tennis4_14sec_720p.mp4"  # Chemin vers la vidéo d'entrée
TERRAIN_OUTPUT_DIR = (
    "./output/terrain"  # Dossier de sortie pour la détection de terrain
)
BALL_OUTPUT_DIR = "./output/balle"  # Dossier de sortie pour la détection de balle
DURATION = 5.0  # Durée (s) pour la détection du terrain
USE_REFINE_KPS = False  # Activer refine_kps
USE_HOMOGRAPHY = False  # Activer homography postprocessing
REBOUND_FRAME = 335  # Frame de rebond pour la détection de fautes

import argparse
import json
import os
import time

from Ball.position_ball import ball
from Fautes.algo_v9 import detection_fautes
from terrain.infer_in_video import infer_terrain


def Rebond():
    """
    Simule la détection de rebond.
    """
    time.sleep(1)
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline détection tennis")
    parser.add_argument(
        "--video_path",
        type=str,
        default=INPUT_VIDEO,
        help="Chemin vers la vidéo d'entrée",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION,
        help="Durée pour la détection du terrain en secondes",
    )
    args = parser.parse_args()

    # Récupération des paramètres depuis la configuration
    model_path = FIELD_MODEL
    video_path = args.video_path
    duration = args.duration
    use_refine_kps = USE_REFINE_KPS
    use_homography = USE_HOMOGRAPHY

    os.makedirs(TERRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BALL_OUTPUT_DIR, exist_ok=True)

    # 1) Détection du terrain
    terrain_json = os.path.join(TERRAIN_OUTPUT_DIR, "terrain_points.json")
    print("[1/2] Lancement détection terrain...")
    infer_terrain(
        model_path=model_path,
        video_path=video_path,
        output_json=terrain_json,
        duration=duration,
        use_refine_kps=use_refine_kps,
    )
    print(f"→ JSON terrain généré dans {terrain_json}")

    # après la détection terrain
    with open(terrain_json, "r") as f:
        terrain_str = f.read()

    # 2) Détection de la balle
    print("[2/2] Lancement détection balle...")
    ball(
        output_dir=BALL_OUTPUT_DIR,
        video_path=video_path,
        model_path="Ball/best2.pt",
    )

    # 3) Analyse des fautes après chaque rebond
    if Rebond():
        # Récupération de la position de balle à partir du JSONL
        frame_no = REBOUND_FRAME
        player = "all"
        x = y = None
        ball_jsonl = os.path.join(BALL_OUTPUT_DIR, "balle.jsonl")
        with open(ball_jsonl, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("frame") == frame_no and not item.get("no_detection", True):
                    det = item["detections"][0]
                    x, y = det["Ball_X"], det["Ball_Y"]
                    print(f"Frame {frame_no} : x={x}, y={y}")
                    break
        if x is None or y is None:
            print(f"Aucune détection pour la frame {frame_no}")
        else:
            good = detection_fautes(terrain_str, x, y, player)
            print("Balle IN" if good else "Faute !")

    print("Pipeline terminé.")


if __name__ == "__main__":
    main()
