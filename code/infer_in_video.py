import os
import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse
import json

def read_video(path_video):
    """ Read video file """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def write_video(imgs_new, fps, path_output_video):
    if not imgs_new:
        print("⚠ Erreur : Aucune frame à écrire dans la vidéo finale !")
        return
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in imgs_new:
        out.write(frame)
    out.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input video')
    parser.add_argument('--output_path', type=str, help='path to output video')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print(f"Utilisation de {device}")

    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360
    
    frames, fps = read_video(args.input_path)
    all_points = []

    for image in tqdm(frames):
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = torch.tensor(np.rollaxis(img.astype(np.float32) / 255., 2, 0)).unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        # Appliquer homographie sur chaque frame si demandé
        if args.use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points_np = np.array([p for p in points if p is not None], dtype=np.float32).reshape(-1, 1, 2)
                points_transformed = cv2.perspectiveTransform(points_np, matrix_trans)
                points = [tuple(np.squeeze(x)) for x in points_transformed]

        all_points.append(points)

    # Convertir en tableau NumPy
    all_points = np.array(all_points, dtype=object)
    num_points = 14  

    # Calculer la moyenne des points après l'homographie
    average_points = []
    for j in range(num_points):
        x_vals = [frame[j][0] for frame in all_points if frame[j][0] is not None]
        y_vals = [frame[j][1] for frame in all_points if frame[j][1] is not None]

        if x_vals and y_vals:
            avg_x = int(np.mean(x_vals))
            avg_y = int(np.mean(y_vals))
            average_points.append((avg_x, avg_y))
        else:
            average_points.append(None)

    # Appliquer une homographie sur les points moyens
    if args.use_homography:
        avg_points_np = np.array([p for p in average_points if p is not None], dtype=np.float32).reshape(-1, 1, 2)
        matrix_trans = get_trans_matrix(average_points)
        if matrix_trans is not None:
            avg_points_transformed = cv2.perspectiveTransform(avg_points_np, matrix_trans)
            avg_points_transformed = [tuple(np.squeeze(x)) for x in avg_points_transformed]
            average_points[:len(avg_points_transformed)] = avg_points_transformed

    # Sauvegarde des points moyens dans un fichier JSON
    json_filename = "average_points.json"
    with open(json_filename, "w") as json_file:
        json.dump({"points": [({"x": p[0], "y": p[1]} if p else None) for p in average_points]}, json_file, indent=4)

    print(f"Points moyens sauvegardés dans {json_filename}")

    # Appliquer les points moyens sur toutes les frames et créer la vidéo finale
    frames_upd = []
    for image in frames:
        for j, point in enumerate(average_points):
            if point is not None:
                image = cv2.circle(image, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 0), thickness=10)
                cv2.putText(image, f"Point {j}", (int(point[0]) + 10, int(point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        frames_upd.append(image)

    write_video(frames_upd, fps, args.output_path)

