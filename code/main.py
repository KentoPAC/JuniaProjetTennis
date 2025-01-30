import cv2
from ultralytics import YOLO
import os
import glob


output_dir = "../assets/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


files_to_delete = glob.glob(os.path.join(output_dir, "detection_*.png"))
for file in files_to_delete:
    try:
        os.remove(file)
    except Exception as e:
        print(f"Erreur lors de la suppression de {file} : {e}")


model_path = "../code/best.pt"
model = YOLO(model_path)


cap = cv2.VideoCapture("udp://127.0.0.1:5000?fifo_size=1000000&overrun_nonfatal=1", cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

frame_counter = 0
print("Capture vidéo en direct démarrée. Appuyez sur 'q' pour quitter.")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Aucune frame valide reçue. Vérifiez le flux vidéo.")
            break

        try:
            
            results = model.predict(frame, conf=0.4, iou=0.3, verbose=False)

            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    x, y, width, height = (
                        box.xywh[0][0].item(),
                        box.xywh[0][1].item(),
                        box.xywh[0][2].item(),
                        box.xywh[0][3].item(),
                    )
                    x1 = int(x - width / 2)
                    y1 = int(y - height / 2)
                    x2 = int(x + width / 2)
                    y2 = int(y + height / 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Balle ({box.conf.item():.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                
                output_image_path = os.path.join(output_dir, f"detection_{frame_counter}.png")
                cv2.imwrite(output_image_path, frame)

            
            cv2.imshow("Détection en direct", frame)

        except Exception as e:
            print(f"Erreur lors de la détection YOLO : {e}")

        frame_counter += 1

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Capture vidéo arrêtée.")
