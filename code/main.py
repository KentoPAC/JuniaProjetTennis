#!/usr/bin/env python3
# main.py : Lance uniquement la détection de terrain (TrackNet)

import argparse
import os
from terrain.infer_in_video import infer_terrain
from Ball.position_ball import ball


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline détection tennis : terrain seulement"
    )
    parser.add_argument(
        "--field_model", required=True, help="Chemin vers model_terrain (tracknet .pt)"
    )
    parser.add_argument(
        "--input_video", required=True, help="Chemin vers la vidéo d'entrée"
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Dossier de sortie pour le JSON"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Durée (s) pour la détection de terrain",
    )
    parser.add_argument(
        "--use_refine_kps", action="store_true", help="Activer refine_kps"
    )
    parser.add_argument(
        "--use_homography",
        action="store_true",
        help="Activer homography postprocessing",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Détection du terrain
    terrain_json = os.path.join(args.output_dir, "terrain_points.json")
    print("[1/2] Lancement détection terrain...")
    infer_terrain(
        model_path=args.field_model,
        video_path=args.input_video,
        output_json=terrain_json,
        duration=args.duration,
        use_refine_kps=args.use_refine_kps,
        use_homography=args.use_homography,
    )
    print(f"→ JSON terrain généré dans {terrain_json}")

    # Détection de la balle
    print("[2/2] Lancement détection balle...")
    ball(
        output_dir=args.output_dir,
        video_path=args.input_video,
        model_path="Ball/best2.pt",
    )

    print("Pipeline terrain terminé.")


if __name__ == "__main__":
    main()
