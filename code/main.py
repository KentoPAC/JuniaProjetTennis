#!/usr/bin/env python3
# main_configured.py : Pipeline détection tennis avec configuration intégrée

# === Configuration - Modifiez ces variables selon vos besoins ===
FIELD_MODEL     = "terrain/model_tennis_court_det.pt"  # Chemin vers le modèle TrackNet
INPUT_VIDEO     = "Vidéos/Test2.mp4"                   # Chemin vers la vidéo d'entrée
TERRAIN_OUTPUT_DIR = "./data/terrain"  # Dossier de sortie pour la détection de terrain
BALL_OUTPUT_DIR    = "./data/balle"    # Dossier de sortie pour la détection de balle
DURATION        = 5.0                                   # Durée (s) pour la détection du terrain
USE_REFINE_KPS  = False                                 # Activer refine_kps
USE_HOMOGRAPHY  = False                                 # Activer homography postprocessing

import os
import time
from terrain.infer_in_video import infer_terrain
from Ball.position_ball import ball
from Fautes.algo_v9 import detection_fautes
import argparse


def Rebond():
    """
    Simule la détection de rebond.
    """
    time.sleep(1)
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline détection tennis")
    parser.add_argument("--video_path", type=str, default=INPUT_VIDEO, help="Chemin vers la vidéo d'entrée")
    parser.add_argument("--duration", type=float, default=DURATION, help="Durée pour la détection du terrain en secondes")
    args = parser.parse_args()

    # Récupération des paramètres depuis la configuration
    model_path       = FIELD_MODEL
    video_path       = args.video_path
    duration         = args.duration
    use_refine_kps   = USE_REFINE_KPS
    use_homography   = USE_HOMOGRAPHY

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

    # 2) Détection de la balle
    print("[2/2] Lancement détection balle...")
    ball(
        output_dir=BALL_OUTPUT_DIR,
        video_path=video_path,
        model_path="Ball/best2.pt",
    )

    # 3) Analyse des fautes après chaque rebond
    # Exemple de boucle : à adapter selon votre source de rebonds
    if Rebond():
        # Remplacez get_last_ball_xy() et player par votre propre logique
        x, y = 412.52, 205.46
        player = "bottom_player"
        good = detection_fautes(terrain_json, x, y, player)
        print("Balle IN" if good else "Faute !")

    print("Pipeline terminé.")


if __name__ == "__main__":
    main()