#!/usr/bin/env python3
# detectionfaute.py : Utilitaire pour vérifier si un rebond est in ou out

import argparse
import json
import os
from Fautes.algo_v9 import detection_fautes


def verifier_faute(terrain_json_path, ball_jsonl_path, frame_no, player="all"):
    """
    Vérifie si un rebond est in ou out pour une frame spécifique.

    Args:
        terrain_json_path: Chemin vers le fichier JSON contenant les points du terrain
        ball_jsonl_path: Chemin vers le fichier JSONL avec les données de la balle
        frame_no: Numéro de la frame à analyser
        player: Joueur concerné ('all' par défaut)

    Returns:
        Tuple (bool, tuple): (True si in/False si out, (x, y) coordonnées de la balle)
    """
    # Charger les données du terrain
    with open(terrain_json_path, "r") as f:
        terrain_str = f.read()

    # Rechercher la position de la balle dans la frame spécifiée
    x = y = None
    with open(ball_jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item.get("frame") == frame_no and not item.get("no_detection", True):
                det = item["detections"][0]
                x, y = det["Ball_X"], det["Ball_Y"]
                break

    if x is None or y is None:
        print(f"Aucune détection pour la frame {frame_no}")
        return None, (None, None)

    # Utiliser la fonction de détection de fautes
    good = detection_fautes(terrain_str, x, y, player)
    return good, (x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérificateur de fautes de tennis")
    parser.add_argument(
        "--terrain",
        type=str,
        default="./output/terrain/terrain_points.json",
        help="Chemin vers le fichier JSON du terrain",
    )
    parser.add_argument(
        "--balle",
        type=str,
        default="./output/balle/balle.jsonl",
        help="Chemin vers le fichier JSONL de la balle",
    )
    parser.add_argument(
        "--frame", type=int, required=True, help="Numéro de la frame à analyser"
    )
    parser.add_argument(
        "--player", type=str, default="all", help="Joueur concerné (all par défaut)"
    )

    args = parser.parse_args()

    good, (x, y) = verifier_faute(args.terrain, args.balle, args.frame, args.player)

    if good is not None:
        print(f"Frame {args.frame} : x={x}, y={y}")
        print("\nBalle IN!! 🎾" if good else "\nFaute!")
