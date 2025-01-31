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
    """ Read video file
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
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
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'),
                          fps, (width, height))
    for num in range(len(imgs_new)):
        frame = imgs_new[num]
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

    print(device)

    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360
    
    frames, fps = read_video(args.input_path)
    frames_upd = []
    for image in tqdm(frames):
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        if args.use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        for j in range(len(points)):
            if points[j][0] is not None:
                image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                                  radius=0, color=(0, 0, 255), thickness=10)
                cv2.putText(image, f"Point {j}", (int(points[j][0]) + 10, int(points[j][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        frames_upd.append(image)

    # Créer une liste pour stocker les données des points
    frames_points = []

    # Boucle pour traiter chaque frame
    for frame_idx, image in enumerate(tqdm(frames)):
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        if args.use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        # Stocker les coordonnées des points de la frame actuelle
        frame_data = {
            "frame": frame_idx,
            "points": [{"x": int(p[0]), "y": int(p[1])} if p[0] is not None else None for p in points]
        }
        frames_points.append(frame_data)

        # Dessiner les points et leurs labels sur l'image
        for j, point in enumerate(points):
            if point[0] is not None:
                image = cv2.circle(image, (int(point[0]), int(point[1])),
                                   radius=0, color=(0, 0, 255), thickness=10)
                cv2.putText(image, f"Point {j}", (int(point[0]) + 10, int(point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        frames_upd.append(image)

    # Sauvegarde des points dans un fichier JSON
    json_filename = "points.json"
    with open(json_filename, "w") as json_file:
        json.dump(frames_points, json_file, indent=4)

    print(f"Points sauvegardés dans {json_filename}")

    write_video(frames_upd, fps, args.output_path)
