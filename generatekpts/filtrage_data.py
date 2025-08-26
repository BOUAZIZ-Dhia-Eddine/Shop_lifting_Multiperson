import csv
from collections import defaultdict
from statistics import stdev

from math import sqrt



def is_valid_frame(frame_data, image_width=640, image_height=480, min_box_height=70, min_valid_kpts_ratio=0.88):
    x_min, y_min, x_max, y_max = frame_data['bbox']
    x_min *= image_width
    y_min *= image_height
    x_max *= image_width
    y_max *= image_height

    box_height = y_max - y_min
    box_width = x_max - x_min


    if y_min >= 0.85 * image_height:
        print("*****1 Rejeté: y_min trop bas — personne partiellement visible")
        return False
    
    aspect_ratio = box_height / (box_width + 1e-5)
    if y_max <= 0.15 * image_height:
        if (aspect_ratio < 0.3 and box_height < 0.20 * image_height)or aspect_ratio >=4.5 :
            print("****2 Rejeté: y_max trop proche du haut — détection partielle")
        return False

    if  y_min >= 0.77*image_height:
        if aspect_ratio >=4 or box_width < image_width * 0.1  or  box_height < 0.2 * image_height:
            print("Rejeté: BBox trop proche du bas avec forme anormale")
            return False

    if box_height < min_box_height or box_width < min_box_height * 0.4:
        print("***************3 Rejeté: BBox trop petite ou trop étroite")
        return False

    if aspect_ratio < 0.3 or aspect_ratio > 6:
        print('*******************5 Rejeté: Aspect ratio anormal')
        return False

    keypoints = frame_data['keypoints']
    valid_points = [
        (x, y) for (x, y, conf) in keypoints
        if 0 <= x <= image_width and 0 <= y <= image_height and conf > 0.0
    ]

    if len(valid_points) / len(keypoints) < min_valid_kpts_ratio:
        print("Rejeté: trop peu de keypoints valides")
        return False

    if len(valid_points) >= 16:
        x_vals = [p[0] for p in valid_points]
        y_vals = [p[1] for p in valid_points]
        std_x, std_y = stdev(x_vals), stdev(y_vals)

        bbox_diag = sqrt(box_width ** 2 + box_height ** 2)
        density_thresh = bbox_diag * 0.1  # seuil strict

        if std_x < density_thresh and std_y < density_thresh:
            print("Rejeté: keypoints trop regroupés (densité anormale)")
            return False
    return True


def load_and_filter_csv(
    csv_path='/home/birali/Desktop/Data_check/data/Skeleton_data_output.csv',
    output_csv_path='/home/birali/Desktop/Data_check/data/filtred_data_version4.csv',
    output_rejected_path='/home/birali/Desktop/Data_check/data/filtred_data_rejetev4.csv',
    max_invalid_frames=2
):
    video_data = defaultdict(lambda: defaultdict(list))

    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

        for row in rows:
            video_clip = row['video_clip']
            if "carrefour" in video_clip:
                continue
            person_id = row['person_id']
            frame_num = int(row['frame_num'])

            frame_data = {
                'class': row['class'],
                'bbox': (
                    float(row['x_min']),
                    float(row['y_min']),
                    float(row['x_max']),
                    float(row['y_max'])
                ),
                'keypoints': []
            }

            keypoint_names = [
                "Nose", "Neck", "Right_Shoulder", "Right_Elbow", "Right_Wrist",
                "Left_Shoulder", "Left_Elbow", "Left_Wrist", "Right_Hip",
                "Right_Knee", "Right_Ankle", "Left_Hip", "Left_Knee",
                "Left_Ankle", "Right_Eye", "Left_Eye", "Right_Ear", "Left_Ear"
            ]

            for name in keypoint_names:
                x = int(float(row[f"{name}_x"]) * 640)
                y = int(float(row[f"{name}_y"]) * 480)
                conf = float(row[f"{name}_conf"])
                frame_data['keypoints'].append([x, y, conf])

            video_data[video_clip][person_id].append((frame_num, frame_data, row))

    filtered_rows = []
    rejected_rows = []
    nb_valid_videos = 0
    nb_rejected_videos = 0

    for video_clip, persons in video_data.items():
        video_has_valid_person = False

        for person_id, frames in persons.items():
            nb_invalid_row_person = 0
            for _, frame_data, _ in frames:
                if not is_valid_frame(frame_data):
                    nb_invalid_row_person += 1

            if nb_invalid_row_person <= max_invalid_frames:
                video_has_valid_person = True
                for _, _, original_row in frames:
                    filtered_rows.append(original_row)
            else:
                for _, _, original_row in frames:
                    rejected_rows.append(original_row)

        if video_has_valid_person:
            nb_valid_videos += 1
        else:
            nb_rejected_videos += 1

    if filtered_rows:
        with open(output_csv_path, mode='w', newline='') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=filtered_rows[0].keys())
            writer.writeheader()
            writer.writerows(filtered_rows)

    if rejected_rows:
        with open(output_rejected_path, mode="w", newline='') as fichier:
            writer = csv.DictWriter(fichier, fieldnames=rejected_rows[0].keys())
            writer.writeheader()
            writer.writerows(rejected_rows)

    print(f"{len(filtered_rows)} lignes valides enregistrées dans {output_csv_path}")
    print(f"{len(rejected_rows)} lignes rejetées enregistrées dans {output_rejected_path}")
    print(f" Statistiques :")


if __name__ == "__main__":
    load_and_filter_csv()