import numpy as np
import pandas as pd
from yolopose_detector import PoseCompiler
import os 
import csv


class KeypointMatcher:
    def __init__(self, csv_path="/home/birali/Desktop/test pour dhia/test.csv", margin=0.03):
        self.margin = margin
        self.dict_csv = self.Organise_Person(csv_path)
        self.persons={}
        self.high_score=[0,1,2,3,4]
        self.midle_score=[5,6,7,8,9,10,11,12]
        self.low_score=[13,14,15,16]
    
    def load_csv(self,csv_path):
            return pd.read_csv(csv_path)
    def Organise_Person(self,csv_path):
            df=self.load_csv(csv_path)
            dict_csv=dict()
            info=()
            for row in df.itertuples(index=True, name="Row"):
                info=(row.video_name+"/"+str(row.clip_name),row.id)
                
                if len(dict_csv)==0 or info[0] not in dict_csv :
                    dict_csv[info[0]]={info[1]:([row.box0,row.box1,row.box2,row.box3],row.class_)}
                else :
                    if info[1] not in dict_csv[info[0]] :
                        dict_csv[info[0]][info[1]]=([row.box0,row.box1,row.box2,row.box3],row.class_)
            return dict_csv
    
    def _calculate_median_keypoints(self, keypoints_dict):
        """Calcule les keypoints médians pour chaque personne détectée"""
        median_kps = {}
        for person_id, kps_list in keypoints_dict.items():
            # Convertir en array numpy (n_frames, 17, 3)
            kps_array = np.array(kps_list)
            # Calculer la médiane sur l'axe des frames
            median_kps[person_id] = np.median(kps_array, axis=0)
        return median_kps
    
    def _expand_bbox(self, bbox):
        """Étend la bbox avec une marge proportionnelle"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        margin_x = width * self.margin
        margin_y = height * self.margin
        
        return [
            max(0, bbox[0] - margin_x),
            max(0, bbox[1] - margin_y),
            min(1, bbox[2] + margin_x),
            min(1, bbox[3] + margin_y)
        ]
    
    def _is_keypoint_inside_bbox(self, kp, bbox):
        """Vérifie si un keypoint est à l'intérieur d'une bbox"""
        x, y, _ = kp
        return (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3])
    
    def _get_bbox_center(self, bbox):
        """Centre de la bbox : (x1, y1, x2, y2)"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _get_keypoints_center(self, keypoints):
        """Centre moyen des keypoints valides"""
        valid_kps = [(kp[0], kp[1]) for kp in keypoints if kp[0] > 0 and kp[1] > 0]
        if not valid_kps:
            return (0, 0)
        return tuple(np.mean(valid_kps, axis=0))

    def _calculate_coverage_score(self, median_kps, bbox):
        """Score de couverture des keypoints dans une bbox étendue"""
        expanded_bbox = self._expand_bbox(bbox)
        inside_count = 0
        nb_keypoint = 0

        for i, kp in enumerate(median_kps):
            if self._is_keypoint_inside_bbox(kp, expanded_bbox):
                nb_keypoint += 1
                if i in self.high_score:
                    inside_count += 0.8
                elif i in self.midle_score:
                    inside_count += 0.6
                else:
                    inside_count += 0.5

        return (inside_count / len(median_kps), nb_keypoint)

    def match_persons(self, video_key, detected_keypoints):
        if video_key not in self.dict_csv:
            raise ValueError(f"Video {video_key} not found in CSV data")

        gt_bboxes = self.dict_csv[video_key]
        median_kps_dict = self._calculate_median_keypoints(detected_keypoints)

        matches = {}
        used_detected_ids = set()

        for gt_id, gt_bbox in gt_bboxes.items():
            best_score = -1
            best_distance = float('inf')
            best_detected_id = None

            gt_center = self._get_bbox_center(gt_bbox[0])

            for detected_id, median_kps in median_kps_dict.items():
                if detected_id in used_detected_ids:
                    continue

                score, nbkeypoint = self._calculate_coverage_score(median_kps, gt_bbox[0])
                kp_center = self._get_keypoints_center(median_kps)

                # Distance entre centres
                dist = np.linalg.norm(np.array(gt_center) - np.array(kp_center))

                if score > best_score or (score == best_score and dist < best_distance):
                    best_score = score
                    best_distance = dist
                    best_detected_id = detected_id

            if best_score >= 0.5:
                matches[gt_id] = (best_detected_id, best_score)
                used_detected_ids.add(best_detected_id)
            else:
                matches[gt_id] = (None, 0)

        return matches
    
    
    def process_all_videos(self):
        for video_clip_key in self.dict_csv.keys():
            print(f"\nTraitement de la vidéo: {video_clip_key}")
            path_video = f"/home/birali/Desktop/test pour dhia/mp4/{video_clip_key}.mp4"
            
            if not os.path.exists(path_video):
                print(f"Fichier vidéo introuvable: {path_video}")
                continue
            
            detections={}
            
            video_compiler = PoseCompiler(path_video)
            video_compiler.VideoLecture()
            
            trace_tracker,mva=video_compiler.get_skeleton(True)
            if trace_tracker and mva :
                detections=self.match_persons(video_clip_key,video_compiler.trace_tracker)
                
                for id_in_csv ,value in detections.items():
                    if value[0]!= None:
                        bboxe,class_=self.dict_csv[video_clip_key ][id_in_csv]
                        if video_clip_key not in self.persons.keys():
                            self.persons[video_clip_key]={id_in_csv:(class_,bboxe,trace_tracker[value[0]],mva[value[0]])}
                        elif id_in_csv not in self.persons[video_clip_key].keys():
                            self.persons[video_clip_key][id_in_csv]=(class_,bboxe,trace_tracker[value[0]],mva[value[0]])

        return self.persons
                
    def mettre_a_jour_csv(self, csv_path="/home/birali/Desktop/test pour dhia/Dataset_features.csv"):
        # Créer les en-têtes du CSV
        headers = ["video_clip", "person_id", "frame_num","class"]
        
        # Noms des keypoints (18 points)
        keypoint_names = [
            "Nose", "Neck", "Right_Shoulder", "Right_Elbow", "Right_Wrist",
            "Left_Shoulder", "Left_Elbow", "Left_Wrist", "Right_Hip",
            "Right_Knee", "Right_Ankle", "Left_Hip", "Left_Knee",
            "Left_Ankle", "Right_Eye", "Left_Eye", "Right_Ear", "Left_Ear"
        ]
        
        headers.extend(["x_min", "y_min", "x_max","y_max"])
        # Ajouter les colonnes pour x, y, confidence de chaque keypoint
        for name in keypoint_names:
            headers.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
        
        # Ajouter les colonnes pour les vitesses
        for name in keypoint_names:
            headers.append(f"{name}_vitesse")
        
        # Ajouter les colonnes pour les angles (seulement ceux spécifiés)
        angles_features_indices = [0, 1, 12, 9, 6, 3]  # Indices des keypoints pour les angles
        angle_names = [keypoint_names[i] for i in angles_features_indices]
        for name in angle_names:
            headers.append(f"{name}_angle")
        
        # Créer le nouveau fichier CSV
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            
            # Parcourir toutes les vidéos et personnes
            for video_clip, persons in self.persons.items():
                for person_id, (class_,bboxe,keypoints_list, mva_list) in persons.items():
                    # Vérifier que les deux listes ont la même longueur
                    if len(keypoints_list) != len(mva_list):
                        print(f"Erreur: longueurs inégales pour {video_clip} personne {person_id}")
                        continue
                    
                    # Pour chaque frame
                    for frame_num in range(len(keypoints_list)):
                        row_data = [video_clip, person_id, frame_num,class_,bboxe[0],bboxe[1],bboxe[2],bboxe[3]]
                        # Ajouter x, y, confidence pour chaque keypoint
                        keypoints = keypoints_list[frame_num]  # shape (18, 3)
                        for kp in keypoints:
                            row_data.extend([kp[0], kp[1], kp[2]])
                        
                        # Ajouter les vitesses pour chaque keypoint
                        mva = mva_list[frame_num]  # shape (18, 3) - magnitude, vitesse, angle
                        for kp in mva:
                            row_data.append(kp[1])  # vitesse est à l'index 1
                        
                        # Ajouter seulement les angles spécifiés
                        for idx in angles_features_indices:
                            row_data.append(mva[idx][2])  # angle est à l'index 2
                        
                        writer.writerow(row_data)
        
        print(f"Données enregistrées dans {csv_path}")
            
                
                


# Exemple d'utilisation
if __name__ == "__main__":
  km=KeypointMatcher()
  p=km.process_all_videos()
  
  km.mettre_a_jour_csv()