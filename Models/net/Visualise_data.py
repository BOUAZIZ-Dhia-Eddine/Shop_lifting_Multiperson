import cv2
import csv
import os
import numpy as np
from collections import defaultdict

def load_and_visualize_false_positives(
    feature_csv_path="/home/birali/Desktop/test pour dhia/Dataset_features.csv" ,
    false_positive_csv_path="/home/birali/Desktop/STGCNN/false_positives.csv",
    video_dir="/home/birali/Desktop/test pour dhia/mp4",#path_video = f"/home/birali/Desktop/test pour dhia/mp4/{video_clip_key}.mp4"
    output_dir="/home/birali/Desktop/output_false_positive_videos"
):
    os.makedirs(output_dir, exist_ok=True)

    # Charger les fausses détections (video_clip, person_id)
    false_positives = set()
    with open(false_positive_csv_path, mode='r') as fpf:
        reader = csv.DictReader(fpf)
        for row in reader:
            false_positives.add((row['video_clip'], row['person_id']))

    # Dictionnaire principal pour stocker les infos du dataset complet
    video_data = defaultdict(lambda: defaultdict(list))

    # Charger les données complètes
    with open(feature_csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_clip = row['video_clip']
            person_id = row['person_id']
            if (video_clip, person_id) not in false_positives:
                continue  # On ignore les personnes correctes

            frame_num = int(row['frame_num'])
            frame_data = {
                'class': row['class'],
                'bbox': (
                    float(row['x_min']), 
                    float(row['y_min']), 
                    float(row['x_max']), 
                    float(row['y_max'])
                ),
                'keypoints': [],
            }

            keypoint_names = [
                "Nose", "Neck", "Right_Shoulder", "Right_Elbow", "Right_Wrist",
                "Left_Shoulder", "Left_Elbow", "Left_Wrist", "Right_Hip",
                "Right_Knee", "Right_Ankle", "Left_Hip", "Left_Knee",
                "Left_Ankle", "Right_Eye", "Left_Eye", "Right_Ear", "Left_Ear"
            ]

            for name in keypoint_names:
                x = float(row[f"{name}_x"])
                y = float(row[f"{name}_y"])
                conf = float(row[f"{name}_conf"])
                frame_data['keypoints'].append([x, y, conf])
            
            video_data[video_clip][person_id].append((frame_num, frame_data))

    # Visualisation
    for video_clip, persons in video_data.items():
        print(f"[INFO] Traitement de la vidéo: {video_clip}")
        path_video = os.path.join(video_dir, f"{video_clip}.mp4")

        if not os.path.exists(path_video):
            print(f"[ERREUR] Fichier vidéo introuvable: {path_video}")
            continue

        safe_video_name = video_clip.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f"{safe_video_name}_false_positive.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

        cap = cv2.VideoCapture(path_video)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            for person_id, frames_data in persons.items():
                frame_data = next((fd for fn, fd in frames_data if fn == frame_count), None)

                if frame_data:
                    def denormalize(x, y, img_width=640, img_height=480):
                        return int(x * img_width), int(y * img_height)

                    x_min, y_min, x_max, y_max = frame_data['bbox']
                    x_min, y_min = denormalize(x_min, y_min)
                    x_max, y_max = denormalize(x_max, y_max)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Jaune

                    for kp in frame_data['keypoints']:
                        x, y, conf = kp
                        if conf > 0.1:
                            x, y = denormalize(x, y)
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    label = f"FalsePos - ID:{person_id} Cls:{frame_data['class']}"
                    cv2.putText(frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(frame)
            frame_count += 1

            # Affichage rapide pour debug
            cv2.imshow(f"False Positive View - {video_clip}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if os.path.exists(output_path):
            print(f"[OK] Vidéo annotée sauvegardée: {output_path}")
        else:
            print(f"[ERREUR] Échec sauvegarde pour: {output_path}")

if __name__ == "__main__":
    load_and_visualize_false_positives()
