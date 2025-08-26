import pandas as pd
from collections import Counter
import numpy as np
import os 


path_data = "/home/birali/Desktop/Data_check/data/filtred_data_last_version.csv"
data = pd.read_csv(path_data)


data["class"] = data["class"].replace({4: 3, 7: 3, 10: 3})
video_clips = data["video_clip"].unique()

np.random.seed(42) 
np.random.shuffle(video_clips)

unique_classes = data.groupby(['video_clip', 'person_id'])['class'].first()
total_class_counts  = Counter(unique_classes)
val_target_class_counts = {cls: int(count * 0.2) for cls, count in total_class_counts.items()}
print(val_target_class_counts)

val_videos = []
train_videos = []
val_class_counts = Counter()
train_class_counts = Counter()
classes_fully_filled = set()


def class_dist_difference(current, target):
    diff = 0
    for cls in target:
        if target[cls] == 0:
            continue
        diff += abs(current.get(cls, 0) - target[cls]) / target[cls]
    return diff


for vid in video_clips:
    video_data = data[data["video_clip"] == vid]
    class_video_counts = Counter(video_data["class"])

 
    only_filled_classes = all(cls in classes_fully_filled for cls in class_video_counts)


    if only_filled_classes:
        train_videos.append(vid)
        train_class_counts += class_video_counts
        continue

    contains_filled_class = any(cls in classes_fully_filled for cls in class_video_counts)

    if contains_filled_class:

        val_part = video_data[~video_data["class"].isin(classes_fully_filled)]
        train_part = video_data[video_data["class"].isin(classes_fully_filled)]

        if not val_part.empty:
            val_videos.append(vid)
            val_class_counts += Counter(val_part["class"])
        if not train_part.empty:
            train_videos.append(vid)
            train_class_counts += Counter(train_part["class"])
        continue


    temp_val_counts = val_class_counts + class_video_counts
    current_diff = class_dist_difference(val_class_counts, val_target_class_counts)
    new_diff = class_dist_difference(temp_val_counts, val_target_class_counts)

    if new_diff < current_diff:
        val_videos.append(vid)
        val_class_counts = temp_val_counts


        for cls, count in val_class_counts.items():
            if count >= val_target_class_counts[cls]:
                classes_fully_filled.add(cls)
    else:
        train_videos.append(vid)
        train_class_counts += class_video_counts

val_df = data[data["video_clip"].isin(val_videos)]
train_df = data[data["video_clip"].isin(train_videos)]
output_dir = os.path.dirname("/home/birali/Desktop/Data_check/data/")


train_csv_path = os.path.join(output_dir, "train_split.csv")
val_csv_path = os.path.join(output_dir, "val_split.csv")

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)

print("Validation class distribution:", Counter(val_df["class"]))
print("Train class distribution:", Counter(train_df["class"]))
print("Classes remplies dans val:", classes_fully_filled)
