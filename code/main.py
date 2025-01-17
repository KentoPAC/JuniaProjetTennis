import cv2
from ultralytics import YOLO
import time

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Utiliser le modèle léger pré-entraîné

# Charger une vidéo
cap = cv2.VideoCapture(r"../assets/tennis.mp4")
assert cap.isOpened(), "Erreur lors de la lecture du fichier "

# Paramètres d'optimisation
resize_dimensions = (640, 480)  # Réduire la résolution pour un traitement rapide
skip_frames = 2  # Traiter une frame sur 4 pour économiser les ressources

frame_count = 0
start_time = time.time()

# Identifier l'ID de la balle de tennis dans COCO
sports_ball_id = 32  # ID pour "sports ball"

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Fin de la vidéo ou frame vide.")
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Redimensionner la frame
    im0 = cv2.resize(im0, resize_dimensions)

    # Détection avec YOLO
    results = model(im0, verbose=False)  # Désactive les logs YOLO par défaut

    # Filtrer les résultats pour la balle de tennis (ID 32)
    for result in results:
        boxes = result.boxes  # Accéder aux boîtes de détection

        for box in boxes:
            x1, y1, x2, y2 = box.xywh[0]  # Coordonnées de la boîte
            conf = box.conf[0]  # Confiance
            cls = box.cls[0]  # Classe détectée

            # Si ce n'est pas une balle de tennis, ignorer
            if int(cls) != sports_ball_id:
                continue

            # Afficher uniquement les informations de la balle
            print(f"Balle de tennis détectée à : x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, Confiance={conf:.2f}")

            # Dessiner la boîte de détection sur l'image
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Afficher l'image avec la détection
    cv2.imshow("Balle de Tennis Détectée", im0)

    # Appuyer sur "q" pour quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Fin du traitement
end_time = time.time()
processing_time = end_time - start_time
print(f"Temps de traitement : {processing_time:.2f} secondes")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
