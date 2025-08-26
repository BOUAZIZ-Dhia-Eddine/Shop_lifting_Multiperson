
import cv2
import numpy as np
import os
import pathlib
import sys
from ultralytics import YOLO
import copy


# Chemin du dossier courant
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import Processing



class PoseCompiler:
    def __init__(self, path_video, model_path="yolov8x-pose.pt", SlidingFrame=30, seuille_supp=14,max_personne=3):
        self.path_video = path_video
        self.max_personne=max_personne
        self.last_frame = -1
        self.Sliding_Frame = SlidingFrame
        self.trace_tracker = dict()
        self.bounding_boxe_list=dict()
        self.seuille_supp = seuille_supp
        self.mapping_openpose_to_yolo = [0, 17, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16, 1, 2, 3, 4]
        # Variables pour le calcul des FPS
        self.frame_count = 0
        self.fps = 0
        self.model =YOLO(model_path)
        self.body_connections = [
            (1, 3), (2, 4), (0, 1), (0, 2),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        self.body_interpolation = [
    (1, 3), (2, 4), (0, 1), (0, 2), 
    (4,6),(3,5),# Connexions supplémentaires
    # Bras droit
    (5, 7),  # Épaule gauche -> Coude gauche
    (7, 9),(9,5) , # Coude gauche -> Poignet gauche
    # Bras gauche
    (6, 8),(6,14),(5,13) ,# Épaule droite -> Coude droit
    (8, 10),(10,6),  # Coude droit -> Poignet droit
    # Tronc
    (5, 6),  # Épaule gauche -> Épaule droite
    (5, 11),  # Épaule gauche -> Hanche gauche
    (6, 12),  # Épaule droite -> Hanche droite
    (11, 12),  # Hanche gauche -> Hanche droite
    # Jambe droite
    (11, 13),  # Hanche gauche -> Genou gauche
    (13, 15),  # Genou gauche -> Cheville gauche
    # Jambe gauche
    (12, 14),  # Hanche droite -> Genou droit
    (14, 16) , (12, 8), (11, 7),(0,6),(0,5) # Genou droit -> Cheville droite
]


        # Initialisation du VideoWriter
        self.video_writer = None
        self.neighbor_dict = self.build_neighbors_dict(self.body_interpolation )

    def build_neighbors_dict(self, body_connections):
        neighbors = {i: [] for i in np.unique(body_connections)}
        for a, b in body_connections:
            neighbors[a].append(b)
            neighbors[b].append(a)
        return neighbors

    def build_neighbors_dict(self, body_connections):
        neighbors = {i: [] for i in np.unique(body_connections)}
        for a, b in body_connections:
            neighbors[a].append(b)
            neighbors[b].append(a)
        return neighbors
    def is_missing(self, kp):
        return (kp[0] == 0 and kp[1] == 0) or kp[2] == 0

    def reflect_point(self, pt, shoulder_left, shoulder_right):
        axis = shoulder_right[:2] - shoulder_left[:2]
        axis_unit = axis / np.linalg.norm(axis)
        center = (shoulder_left[:2] + shoulder_right[:2]) / 2
        v = pt - center
        proj = np.dot(v, axis_unit) * axis_unit
        perp = v - proj
        reflected = pt - 2 * perp
        return reflected

    def correct_missing_arm_by_symmetry(self, keypoints):
        left_shoulder, left_elbow, left_wrist = 5, 7, 9
        right_shoulder, right_elbow, right_wrist = 6, 8, 10

        left_missing = all(self.is_missing(keypoints[i]) for i in [left_shoulder, left_elbow, left_wrist])
        right_missing = all(self.is_missing(keypoints[i]) for i in [right_shoulder, right_elbow, right_wrist])

        if self.is_missing(keypoints[5]) or self.is_missing(keypoints[6]):
            return keypoints  # impossible de corriger sans les deux épaules

        if left_missing and not right_missing:
            for li, ri in zip([5, 7, 9], [6, 8, 10]):
                kp = keypoints[ri, :2]
                reflected = self.reflect_point(kp, keypoints[5], keypoints[6])
                keypoints[li, :2] = reflected
                keypoints[li, 2] = 0.6
        elif right_missing and not left_missing:
            for ri, li in zip([6, 8, 10], [5, 7, 9]):
                kp = keypoints[li, :2]
                reflected = self.reflect_point(kp, keypoints[5], keypoints[6])
                keypoints[ri, :2] = reflected
                keypoints[ri, 2] = 0.6

        return keypoints

    def interpolate_keypoints(self, keypoints):
        keypoints=self.correct_missing_arm_by_symmetry(keypoints)
        invalid = ((keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)) | (keypoints[:, 2] == 0)
        if np.all(invalid):
            return None

        for i in np.where(invalid)[0]:
            valid_neighbors = [keypoints[n] for n in self.neighbor_dict.get(i, []) if not invalid[n]]
            if valid_neighbors:
                valid_neighbors = np.array(valid_neighbors)
                keypoints[i, :2] = valid_neighbors[:, :2].mean(axis=0)
                keypoints[i, 2] = max(keypoints[i, 2], 0.5 if len(valid_neighbors) >= 2 else 0.3)
                invalid[i] = False
            else:
                return None
        return keypoints
    def plot_keypoints(self, features, image_array, n_w, n_h):
        for tup in self.body_connections:
            feat1 = tuple((features[tup[0], :2] * [n_w, n_h]).astype(int))
            feat2 = tuple((features[tup[1], :2] * [n_w, n_h]).astype(int))
            cv2.circle(image_array, feat1, 3, (255, 0, 0), -1)
            cv2.circle(image_array, feat2, 3, (255, 0, 0), -1)
            cv2.line(image_array, feat1, feat2, (0, 0, 255), 2)
        return image_array

    def Track_Image(self, image_array, n_h, n_w):
        results = self.model.track(
            source=image_array,
            conf=0.3,
            classes=0,
            show=False,
            save=False,
            tracker='C:/Users/PC_DHIA/Desktop/pfe_action_detection/utils/botsortMediaPipe.yaml',
            persist=True
        )

        dict_results = dict()
        image_array=image_array
        for result in results:
            if result.boxes.id is not None and result.keypoints is not None:
                for obj_id, keypoints, conf_keypoints in zip(result.boxes.id, result.keypoints.xy, result.keypoints.conf):
                    keypoints = keypoints.cpu().numpy()
                    conf_keypoints = conf_keypoints.cpu().numpy()
                    features = np.array([
                        (float(kp[0] / n_w), float(kp[1] / n_h), float(conf))
                        for kp, conf in zip(keypoints, conf_keypoints)
                    ])
                    
                    features = self.interpolate_keypoints(features)
                    all_valid = not np.any(((features[:, 0] == 0) & (features[:, 1] == 0)) | (features[:, 2] == 0))
                    if features is not None and  all_valid :
                        print("*",features)
                        image_array = self.plot_keypoints(features, image_array, n_w, n_h)
                        dict_results[int(obj_id)] = features
        self.make_18_points(dict_results)
        
                        
        return dict_results, image_array

    def calcule_distance(self, keypoints):
        if not self.trace_tracker:
            return None

        keypoints_xy = keypoints[:, :2]
        best_match = None
        best_similarity = -float('inf')
        box1 = [
            np.min(keypoints_xy[:, 0]), 
            np.min(keypoints_xy[:, 1]), 
            np.max(keypoints_xy[:, 0]), 
            np.max(keypoints_xy[:, 1])
        ]
        def calculate_center_distance(box1, last):
            """Calcule la distance normalisée entre les centres des boxes"""
            box2 = [
                np.min(last[:, 0]),
                np.min(last[:, 1]),
                np.max(last[:, 0]),
                np.max(last[:, 1])
            ]
            center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
            center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
            distance = np.linalg.norm(center1 - center2)
            return distance / np.sqrt(2) 
        
        first_person_id = next(iter(self.trace_tracker))
        last_kp_first = self.trace_tracker[first_person_id][-1][:, :2]
        median_dist = np.median(np.linalg.norm(last_kp_first - keypoints_xy, axis=1))
        center_sim = calculate_center_distance(box1, last_kp_first)
        dist_threshold = median_dist + (1 - center_sim)  

        def calculate_similarity(box1, last):
            """Calcule un score de similarité composite normalisé entre 0 et 1"""
            box2 = [
                np.min(last[:, 0]),
                np.min(last[:, 1]),
                np.max(last[:, 0]),
                np.max(last[:, 1])
            ]
            
            # Calcul IoU
            xi1 = max(box1[0], box2[0])
            yi1 = max(box1[1], box2[1])
            xi2 = min(box1[2], box2[2])
            yi2 = min(box1[3], box2[3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            
            box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
            box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
            iou = inter_area / (box1_area + box2_area - inter_area + 1e-5)
            
            def aspect_ratio_sim(b1, b2):
                w1, h1 = b1[2]-b1[0], b1[3]-b1[1]
                w2, h2 = b2[2]-b2[0], b2[3]-b2[1]
                r1 = w1 / (h1 + 1e-5)
                r2 = w2 / (h2 + 1e-5)
                return 1 - abs(r1 - r2)/(max(r1, r2) + 1e-5)
            
            ratio_sim = aspect_ratio_sim(box1, box2)
            
            weights = {'iou': 0.6, 'ratio': 0.4}
            return weights['iou'] * iou + weights['ratio'] * ratio_sim
        for person_id, traces in self.trace_tracker.items():
            last_kp = traces[-1][:, :2]
            
            current_median_dist = np.median(np.linalg.norm(last_kp - keypoints_xy, axis=1))
            current_center_dist = calculate_center_distance(box1, last_kp)
            combined_dist = current_median_dist + current_center_dist
            
            if combined_dist < dist_threshold:
                dist_threshold = combined_dist
            
            similarity = calculate_similarity(box1, last_kp)
            if combined_dist == dist_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = (person_id, similarity)

        return best_match if best_similarity > 0.4 else None
    
    def ignore_video(self):
        for key in list(self.trace_tracker.keys()):
            if len(self.trace_tracker[key]) <= self.seuille_supp:
                del self.trace_tracker[key] 
        
    
    def Update(self, dict_results, frame_actuelle):
        if frame_actuelle <= self.last_frame:
            return None
        if not dict_results:  # Vérifie si le dictionnaire est vide
           return None
        for values in dict_results.values():
            if values.ndim == 2 and values.shape[1] == 3:
                break
        else:
            return None

        for detected_id, keypoints in dict_results.items():
            # Trouver la personne existante la plus proche
            closest_person = self.calcule_distance(keypoints)
            if closest_person is not None:
                # Mettre à jour la trace existante
                existing_id = closest_person[0]
                self.trace_tracker[existing_id].append(keypoints)
            else:
                if detected_id in self.trace_tracker.keys():
                   self.trace_tracker[detected_id].append(keypoints)
                else :
                   self.trace_tracker[detected_id] = [keypoints]        
        self.last_frame = frame_actuelle

    def VideoLecture(self, target_width=1680, target_height=1680):
            
            cap = cv2.VideoCapture(self.path_video)
            if not cap.isOpened():
                print("Impossible d'ouvrir la vidéo...")
                return

            total_nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #frame_delay = 1 / 60  # Délai pour chaque frame à 60 FPS

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = "C:/Users/PC_DHIA/Desktop/videos_pose_estimation/second_2134voir2.mp4"
            self.video_writer = cv2.VideoWriter(output_video_path, fourcc, 60, (target_width, target_height))
            
            for comp_frame in range(total_nb_frame):
                ret, frame = cap.read()
                if not ret:
                    print("Fin de la vidéo")
                    break

                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                n_h, n_w, _ = frame.shape
                dict_results, image_array= self.Track_Image(frame,1680,1680)
                self.Update(dict_results, comp_frame + 1)  
        # Sauvegarde de la frame
                
                self.video_writer.write(image_array)
                #cv2.imshow("Pose Estimation", image_array)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
    def make_18_points(self,dict_result):
        for key  , value in dict_result.items():
             key_point =(value[5,:]+value[6,:])/2
             key_point=key_point.reshape((1,3))
             value=np.vstack((value,key_point))
             
             value=self.convert_YoloPose_vers_OpenPose(value)
             
             dict_result[key]=value
             
    def convert_YoloPose_vers_OpenPose(self,keypoints_yolo):
        return keypoints_yolo[self.mapping_openpose_to_yolo]
    def get_skeleton(self, Vitesse=False,type=0):
        
        self.ignore_video()
        processor = Processing()
        self.trace_tracker = processor.Ajout_Padding(self.trace_tracker, dt=5, type=type, Sliding_Frame=self.Sliding_Frame)
        for key,value in self.trace_tracker.items():
            B = len(value)
            if B > self.Sliding_Frame:
                    self.trace_tracker[key]=value[0:self.Sliding_Frame]
        if Vitesse:
            copie_tracker = copy.deepcopy(self.trace_tracker)
            
            return (self.trace_tracker ,processor.CalculeVitesse(copie_tracker))
        return self.trace_tracker
    def calcule_confscore_personne(self, personne):
        return np.sum([np.mean(frame[:, 2]) for frame in personne])

    def modif_tracking(self):
        
        if len(self.trace_tracker) <= self.max_personne:
            return None
        else:
            liste_des_score = []
            for key, personne in self.trace_tracker.items():
                score = self.calcule_confscore_personne(personne)
                liste_des_score.append((key, score))
            
            # Trier les scores décroissants et ne garder que les top max_personne
            liste_des_score = sorted(liste_des_score, key=lambda x: x[1], reverse=True)[:self.max_personne]

            # Extraire uniquement les clés à garder
            liste_des_score = [tup[0] for tup in liste_des_score]

            # Supprimer les clés à ne pas garder
            for key in list(self.trace_tracker.keys()):
                if key not in liste_des_score:
                    del self.trace_tracker[key]
               
# Point d'entrée du programme
if __name__ == "__main__":
    video_source = "C:/Users/PC_DHIA/Desktop/videos_pose_estimation/second_2134.mp4"
    p=PoseCompiler(video_source)
    p.VideoLecture()
    
    '''for key in p.trace_tracker.keys():
        print("====>",key,"===",len(p.trace_tracker[key]))
    p.modif_tracking()
    print((p.trace_tracker.keys()))
    tt,v = p.get_skeleton(True)
    print("Résultats du tracking:", len(tt[1]))
    for key in tt.keys():
        print("====>",key,len(tt[key]))
    print("********************************")

    print("Résultats du tracking:", len(v[1]))  '''

