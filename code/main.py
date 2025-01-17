import cv2
from roboflow import Roboflow
import time

# Initialiser l'API Roboflow
rf = Roboflow(api_key="HGDPPCrv2JXA19Dfds89")
project = rf.workspace().project("tennis_ball_yolov8-4qhxa-qjgrh")
model = project.version("1").model

# Charger une vidéo
video_path = "../assets/tennis.mp4"  # Chemin de votre vidéo
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Erreur lors de la lecture du fichier vidéo."

# Paramètres pour le traitement vidéo
resize_dimensions = (640, 480)  # Réduire la résolution pour un traitement rapide
skip_frames = 2  # Traiter une frame sur 2 pour économiser les ressources
fps = 5  # Fréquence d'échantillonnage pour l'API Roboflow

frame_count = 0
start_time = time.time()

# Générer une URL pour la vidéo via l'API
job_id, signed_url, expire_time = model.predict_video(
    video_path,
    fps=fps,
    prediction_type="batch-video",
)

print(f"Analyse vidéo en cours (Job ID : {job_id})...")

# Attendre la complétion de l'analyse vidéo
results = model.poll_until_video_results(job_id)
print("Analyse terminée.")
# Ajouter pour vérifier la réponse de l'API
print("Analyse terminée. Vérification des résultats...")

# Inspecter les résultats
print("Contenu brut de 'results':", results)

# Vérifier si la clé "predictions" existe
if "predictions" not in results:
    print("Erreur : La clé 'predictions' est absente des résultats.")
    if "error" in results:
        print("Détails de l'erreur :", results["error"])
    exit()

# Extraire les prédictions
predictions = results["predictions"]

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Fin de la vidéo ou frame vide.")
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Redimensionner l'image pour correspondre aux dimensions des résultats
    im0 = cv2.resize(im0, resize_dimensions)

    # Filtrer les résultats pour les balles de tennis
    for pred in predictions:
        if pred["class"] == "tennis_ball":  # Vérifier si la classe est "tennis_ball"
            x = int(pred["x"])  # Coordonnée x du centre
            y = int(pred["y"])  # Coordonnée y du centre
            width = int(pred["width"])
            height = int(pred["height"])

            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            print(f"Balle de tennis détectée : x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Dessiner la boîte de détection sur l'image
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Afficher l'image avec les détections
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
