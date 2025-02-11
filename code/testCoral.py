import argparse
import glob
import os
import cv2
import time
import json
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path', required=True)
    parser.add_argument('--labels', help='label file path', required=True)
    parser.add_argument('--camera_idx', type=int, default=0, help='Index of which video source to use.')
    parser.add_argument('--top_k', type=int, default=3, help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.4, help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Chemin du dossier de sauvegarde
    output_dir = "../assets/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Crée le dossier s'il n'existe pas

    # Supprimer les fichiers existants correspondant au motif "detection_*.png"
    files_to_delete = glob.glob(os.path.join(output_dir, "detection_*.png"))
    for file in files_to_delete:
        try:
            os.remove(file)
        except Exception:
            pass  # Supprime les messages d'erreur pour une console simplifiée

    # Supprimer le fichier detections.json s'il existe
    output_json_path = os.path.join(output_dir, "detections.json")
    if os.path.exists(output_json_path):
        os.remove(output_json_path)

    # Charger le flux vidéo de la caméra
    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra.")
        exit()

    frame_counter = 0  # Compteur pour différencier les images sauvegardées

    # Lire les propriétés de la caméra
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Flux vidéo en direct à {fps:.2f} FPS.")

    # Fichier JSON pour enregistrer les détections
    detections_data = {
        "video": "Flux vidéo en direct",
        "fps": fps,
        "total_frames": 0,  # Le nombre total de frames n'est pas connu à l'avance pour un flux en direct
        "duree_total": 0,  # Temps total de traitement
        "detections": []
    }

    # Démarrer le timer pour le temps total de traitement
    start_time = time.time()

    # Lecture frame par frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la lecture du flux vidéo.")
            break

        print(f"Traitement de la frame {frame_counter}...")

        # Démarrer le timer pour le temps de traitement de la frame
        frame_start_time = time.time()

        # Prétraiter l'image pour le modèle (mettre à la bonne taille et normaliser les pixels)
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]

        detection_info = {
            "frame": frame_counter,
            "detections": []
        }

        # Filtrer les prédictions en fonction de la confiance
        for obj in objs:
            bbox = obj.bbox.scale(frame.shape[1] / inference_size[0], frame.shape[0] / inference_size[1])
            x0, y0, x1, y1 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
            label = labels.get(obj.id, obj.id)
            confidence = obj.score

            # Ajouter les coordonnées et la confiance au JSON
            detection_info["detections"].append({
                "x1": x0,
                "y1": y0,
                "x2": x1,
                "y2": y1,
                "label": label,
                "confidence": confidence,
                "temps_detection": 0  # Placeholder pour le temps de détection
            })

            # Dessiner le rectangle autour de l'objet détecté
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Calculer le temps de traitement de la frame
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time

        # Ajouter le temps de détection au JSON
        for detection in detection_info["detections"]:
            detection["temps_detection"] = frame_processing_time

        # Ajouter les détections de la frame au fichier JSON
        detections_data["detections"].append(detection_info)

        # Sauvegarder l'image annotée dans le dossier /assets/
        output_image_path = os.path.join(output_dir, f"detection_{frame_counter}.png")
        cv2.imwrite(output_image_path, frame)

        # Incrémenter le compteur de frames
        frame_counter += 1

    # Calculer le temps total de traitement
    end_time = time.time()
    total_processing_time = end_time - start_time
    detections_data["duree_total"] = total_processing_time

    # Sauvegarder les données de détection dans un fichier JSON
    with open(output_json_path, "w") as json_file:
        json.dump(detections_data, json_file, indent=4)
    print(f"Données de détection sauvegardées dans : {output_json_path}")

    # Libérer les ressources
    cap.release()
    print("Traitement terminé.")

if __name__ == '__main__':
    main()