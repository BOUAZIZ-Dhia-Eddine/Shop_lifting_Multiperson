import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm
import pandas as pd
import math
#from net.Stgcnn_withattentionm import Model
from net.STGCNDataset import STGCNDataset 
from net.st_gcnn_withatt import Model
#from net.ConvLstmShopLifting import MultiPersonContextLSTM

class Load_data:
    def __init__(self,csv_path,cache_dir="/home/birali/Desktop/projectpfe_action_recognition/STGCNN/net/cache"):
        self.csv_path=csv_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.JOINT_MAPPING = {
        'Nose': 0, 'Neck': 1,
        'Right_Shoulder': 2, 'Right_Elbow': 3, 'Right_Wrist': 4,
        'Left_Shoulder': 5, 'Left_Elbow': 6, 'Left_Wrist': 7,
        'Right_Hip': 8, 'Right_Knee': 9, 'Right_Ankle': 10,
        'Left_Hip': 11, 'Left_Knee': 12, 'Left_Ankle': 13,
        'Right_Eye': 14, 'Left_Eye': 15,
        'Right_Ear': 16, 'Left_Ear': 17
    }

        self.ANGLE_NAMES = {
        'Nose_angle': 0, 'Neck_angle': 1, 'Left_Knee_angle': 6,
        'Right_Knee_angle': 3, 'Left_Elbow_angle': 12, 'Right_Elbow_angle': 9
    }
    def _get_cache_path(self):
        
        # Génère un nom de fichier unique basé sur le CSV (ex: "train.csv" → "cache/train.pt")
        filename = os.path.basename(self.csv_path).replace(".csv", ".pt")
        print(filename)
        return os.path.normpath(os.path.join(self.cache_dir, filename))
    def csv_to_stgcn_tensor2(self, num_frames=30, num_joints=18, num_features=2, max_persons=3):
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            print("Chargement depuis le cache :", cache_path)
            cached_data = torch.load(cache_path,weights_only=False)
            print(cached_data['features'].shape)
            return (
                cached_data['features'],  
                cached_data['labels'],
                cached_data['class_to_idx']
            )

        df = pd.read_csv(self.csv_path)
        df = df.sort_values(['video_clip', 'person_id', 'frame_num'])
        df["class"] = df["class"].replace({4: 0, 7: 0, 10: 0})
        df["class"] = df["class"].replace({2:1})
        unique_classes = df['class'].unique()
        print(unique_classes)
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        unique_videos = df['video_clip'].unique()
        video_id_to_index = {vid: i for i, vid in enumerate(unique_videos)}
        num_videos = len(unique_videos)
        num_videos = len(unique_videos)

        features_tensor = np.zeros((num_videos, num_features, num_frames, num_joints, max_persons))
        labels = np.full((num_videos, max_persons), -1)

        df = df[df['frame_num'] < num_frames]

        person_assignments = {vid: {} for vid in unique_videos}
        excess_persons = []

        grouped = df.groupby(['video_clip', 'person_id'])
        
        for (video_id, person_id), group in tqdm(grouped, total=len(grouped), desc="Processing CSV"):
            vid_idx = video_id_to_index[video_id]
            action_class = group['class'].iloc[0]  
            
            if person_id not in person_assignments[video_id]:
                if len(person_assignments[video_id]) >= max_persons:
                    excess_persons.append((video_id, person_id, class_to_idx[action_class]))
                    continue
                    
                person_assignments[video_id][person_id] = len(person_assignments[video_id])
                labels[vid_idx, person_assignments[video_id][person_id]] = class_to_idx[action_class]
                
            person_idx = person_assignments[video_id][person_id]

            for frame_idx, row in group.iterrows():
                frame_num = int(row['frame_num'])
                features = np.zeros((num_joints, num_features))
                
                for joint_name, joint_idx in self.JOINT_MAPPING.items():
                    features[joint_idx, 0] = row[f'{joint_name}_x']
                    features[joint_idx, 1] = row[f'{joint_name}_y']
                    #features[joint_idx, 2] = row[f'{joint_name}_vitesse']

                '''for joint_name, joint_idx in self.ANGLE_NAMES.items():
                    features[joint_idx, 3] = row[joint_name]'''
                
                features_tensor[vid_idx, :, frame_num, :, person_idx] = features.T

        for video_id in unique_videos:
            vid_idx = video_id_to_index[video_id]
            current_count = len(person_assignments[video_id])
            available_slots = max_persons - current_count
            
            while available_slots > 0 and excess_persons:
                
                ex_video_id, ex_person_id, ex_class = excess_persons.pop(0)
                ex_person_data = df[(df['video_clip'] == ex_video_id) & 
                                    (df['person_id'] == ex_person_id)]
                
                if ex_person_data.empty:
                    continue
                    
                new_pid = f"excess_{ex_video_id}_{ex_person_id}"
                person_assignments[video_id][new_pid] = max_persons - available_slots
                labels[vid_idx, person_assignments[video_id][new_pid]] = ex_class

                for frame_num, row in ex_person_data.groupby('frame_num').first().iterrows():
                    features = np.zeros((num_joints, num_features))
                    
                    for joint_name, joint_idx in self.JOINT_MAPPING.items():
                        features[joint_idx, 0] = row[f'{joint_name}_x']
                        features[joint_idx, 1] = row[f'{joint_name}_y']
                        #features[joint_idx, 2] = row[f'{joint_name}_vitesse']
                    
                    '''for joint_name, joint_idx in self.ANGLE_NAMES.items():
                        features[joint_idx, 3] = row[joint_name]'''
                    
                    features_tensor[vid_idx, :, frame_num, :, person_assignments[video_id][new_pid]] = features.T
                
                available_slots -= 1

        remaining = len(excess_persons)
        if remaining > 0:
            num_extra_videos = math.ceil(remaining/ max_persons)  #(remaining + max_persons - 1) // max_persons
            
            new_features = np.zeros((num_extra_videos, num_features, num_frames, num_joints, max_persons))
            new_labels = np.full((num_extra_videos, max_persons), -1)
            
            for i in range(num_extra_videos):
                for j in range(max_persons):
                    if not excess_persons:
                        break
                        
                    ex_video_id, ex_person_id, ex_class = excess_persons.pop(0)
                    ex_person_data = df[(df['video_clip'] == ex_video_id) & 
                                    (df['person_id'] == ex_person_id)]
                    
                    if ex_person_data.empty:
                        continue
                        
                    new_labels[i, j] = ex_class

                    for frame_num, row in ex_person_data.groupby('frame_num').first().iterrows():
                        features = np.zeros((num_joints, num_features))
                        
                        for joint_name, joint_idx in self.JOINT_MAPPING.items():
                            features[joint_idx, 0] = row[f'{joint_name}_x']
                            features[joint_idx, 1] = row[f'{joint_name}_y']
                            #features[joint_idx, 2] = row[f'{joint_name}_vitesse']
                        
                        '''for joint_name, joint_idx in self.ANGLE_NAMES.items():
                            features[joint_idx, 3] = row[joint_name]'''
                        
                        new_features[i, :, frame_num, :, j] = features.T

            features_tensor = np.concatenate((features_tensor, new_features), axis=0)
            labels = np.concatenate((labels, new_labels), axis=0)
            
        features_tensor = torch.from_numpy(features_tensor).float()
        labels = torch.from_numpy(labels).long()

        result = {
                'features': features_tensor,
                'labels': labels,
                'class_to_idx': class_to_idx,
                'config': {
                    'num_frames': num_frames,
                    'num_joints': num_joints,
                    'num_features': num_features,
                    'max_persons': max_persons
                }
            }

        try:
                torch.save(result, cache_path)
                print(f"Données sauvegardées dans le cache: {cache_path}")
        except Exception as e:
                print(f"Erreur lors de la sauvegarde du cache: {e}")

        return features_tensor, labels, class_to_idx
class ModelTester:
    def __init__(self, config, model_weights_path, csv_path):
        self.config = config
        self.model_weights_path = model_weights_path
        self.csv_path = csv_path
        
        # Charger les données
        self.data_loader = Load_data(csv_path)
        self.features, self.labels, self.class_to_idx = self.data_loader.csv_to_stgcn_tensor2()
        
        # Inverser le mapping classe → index pour l'affichage
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(self.idx_to_class)
        # Initialiser le modèle
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialise le modèle avec les poids sauvegardés"""
        model = Model(
            in_channels=self.config['in_channels'],
            num_class=self.config['num_class'],
            graph_args=self.config['graph_args'],
            edge_importance_weighting=self.config['edge_importance_weighting'],
            max_persons=self.config['max_persons']
        ).to(self.config['device'])
        #model=MultiPersonContextLSTM().to(self.config['device'])
        
        # Charger les poids sauvegardés
        if os.path.exists(self.model_weights_path):
            model.load_state_dict(torch.load(self.model_weights_path, map_location=self.config['device']))
            print(f"Modèle chargé depuis {self.model_weights_path}")
        else:
            raise FileNotFoundError(f"Fichier de poids {self.model_weights_path} introuvable")
            
        model.eval()  # Mode évaluation
        return model
    
    def get_test_dataloader(self):
        """Crée le DataLoader pour les données de test"""
        dataset = STGCNDataset(self.features, self.labels)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=2
        )
        return dataloader
    
    def run_inference(self):
        """Exécute l'inférence complète et calcule les métriques"""
        test_loader = self.get_test_dataloader()
        
        all_preds = []
        all_labels = []
        person_metrics = []  # Pour stocker les métriques par personne
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.config['device'])
                y_batch = y_batch.to(self.config['device'])
                
                outputs = self.model(x_batch)
                outputs = outputs.view(-1, self.config['num_class'])
                y_batch = y_batch.view(-1)
                
                # Filtrer les padding (-1)
                mask = y_batch != -1
                outputs = outputs[mask]
                y_batch = y_batch[mask]
                
                if len(y_batch) > 0:
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
        
        # Calcul des métriques globales
        if len(all_labels) > 0:
            global_accuracy = 100 * accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            
            # Métriques par classe
            class_metrics = {}
            for cls_idx in self.class_to_idx.values():
                cls_name = self.idx_to_class[cls_idx]
                cls_mask = np.array(all_labels) == cls_idx
                
                if sum(cls_mask) > 0:
                    cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(
                        cls_mask,
                        np.array(all_preds) == cls_idx,
                        average='binary',
                        zero_division=0
                    )
                    cls_accuracy = 100 * accuracy_score(
                        np.array(all_labels)[cls_mask],
                        np.array(all_preds)[cls_mask]
                    )
                    
                    class_metrics[cls_name] = {
                        'accuracy': cls_accuracy,
                        'precision': cls_precision,
                        'recall': cls_recall,
                        'f1': cls_f1,
                        'support': sum(cls_mask)
                    }
                else:
                    class_metrics[cls_name] = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'support': 0
                    }
            
            # Matrice de confusion# Remplacez par votre chemin
            self._plot_confusion_matrix( np.array(all_labels),np.array(all_preds))
            
            return {
                'global_accuracy': global_accuracy,
                'global_precision': precision,
                'global_recall': recall,
                'global_f1': f1,
                'class_metrics': class_metrics
            }
        else:
            raise ValueError("Aucune donnée valide pour l'évaluation")
    
    def _plot_confusion_matrix(self, true_labels, pred_labels):
        """Génère et sauvegarde la matrice de confusion"""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(12, 10))
        
        # Obtenir les noms de classes dans l'ordre des indices
        class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies étiquettes")
        plt.title("Matrice de Confusion")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig("confusion_matrix_inference.png")
        plt.close()
        print("Matrice de confusion sauvegardée sous confusion_matrix_inference.png")

    def print_metrics(self, metrics):
        """Affiche les métriques de manière lisible"""
        print("\n=== Métriques Globales ===")
        print(f"Accuracy: {metrics['global_accuracy']:.2f}%")
        print(f"Precision: {metrics['global_precision']:.4f}")
        print(f"Recall: {metrics['global_recall']:.4f}")
        print(f"F1-Score: {metrics['global_f1']:.4f}")
        
        print("\n=== Métriques par Classe ===")
        for cls_name, cls_metrics in metrics['class_metrics'].items():
            print(f"\nClasse: {cls_name}")
            print(f"  Support: {cls_metrics['support']}")
            print(f"  Accuracy: {cls_metrics['accuracy']:.2f}%")
            print(f"  Precision: {cls_metrics['precision']:.4f}")
            print(f"  Recall: {cls_metrics['recall']:.4f}")
            print(f"  F1-Score: {cls_metrics['f1']:.4f}")


if __name__ == "__main__":
    # Configuration (doit correspondre à celle utilisée pendant l'entraînement)
        config = {
  
            'in_channels': 2,  # x, y, vitesse
            'num_class': 2,  # Nombre de classes
            'graph_args': {'strategy':'spatial'},
            'edge_importance_weighting': True,
            'max_persons': 3,
            'num_features':2,
            'batch_size': 80,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        '''config = {
  
            'in_channels': 2,  # x, y, vitesse
            'num_class': 2,  # Nombre de classes
            'graph_args': {'strategy':'spatial'},
            'edge_importance_weighting': True,
            'max_persons': 1,
            'num_features':2,
            'batch_size': 80,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }'''
        '''config = {
  
            'in_channels': 3,  # x, y, vitesse
            'num_class': 2,  # Nombre de classes
            'graph_args': {'strategy':'spatial'},
            'edge_importance_weighting': True,
            'max_persons': 3,
            'num_features':3,
            'batch_size': 80,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }'''

        # Initialiser et exécuter le test
        tester = ModelTester(
            config=config,
            #model_weights_path="/home/birali/Desktop/STGCNN/best_model_s.pth",
            #->model_weights_path="/home/birali/Desktop/projectpfe_action_recognition/STGCNN/single_personne_stgcn.pth", 
            #  # Remplacez par votre chemin/home/birali/Desktop/STGCNN/ahahahahahahah.pth
            #->model_weights_path="/home/birali/Desktop/STGCNN/best_model_s.pth",
            model_weights_path="/home/birali/Desktop/projectpfe_action_recognition/STGCNN/multipersonne.pth",
            csv_path="/home/birali/Desktop/test pour dhia/Dataset_features.csv"       
        )

        metrics = tester.run_inference()
        tester.print_metrics(metrics)