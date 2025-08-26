

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
import os 
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transforms.DataAugmenter import DataAugmenter
from net.Stgcnn_withattentionm import Model
from net.focal_loss import FocalLoss
from net.EarlyStopping import EarlyStopping
from torch.utils.data import DataLoader
from net.STGCNDataset import STGCNDataset 
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns 
class Load_data:
    def __init__(self,csv_path,cache_dir="/home/birali/Desktop/STGCNN/net/cache"):
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
        return os.path.normpath(os.path.join(self.cache_dir, filename))
    def csv_to_stgcn_tensor2(self, num_frames=30, num_joints=18, num_features=3, max_persons=3):
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

        unique_classes = df['class'].unique()
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        print(class_to_idx)
        unique_videos = df['video_clip'].unique()
        video_id_to_index = {vid: i for i, vid in enumerate(unique_videos)}
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
        
        


# ======================================================================
# 1. CONFIGURATION
# ======================================================================
class Train_stgcn:
    def __init__(self,csv,features_t, labels_t, class_to_idx_t,features_v,labels_v,
                 class_to_idx_v,batch_size=42,num_class=2,max_persons=3,learning_rate=7e-4 ,epochs=100):
        self.csv_val=csv
        self.features_t=features_t
        self.labels_t=labels_t
        self.class_to_idx_t=class_to_idx_t
        self.features_v=features_v
        self.labels_v=labels_v
        self.class_to_idx_v=class_to_idx_v


    # Hyperparamètres
        self.config = {
        'batch_size': batch_size,
        'num_frames': 30,
        'num_joints': 18,
        'num_features':3,
        'max_persons':3 ,#max_persons,
        'in_channels': 3,
        'num_class':num_class,
        'learning_rate':learning_rate,
        'epochs': epochs,
        'graph_args': {'strategy':'spatial'},
        'edge_importance_weighting':True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
        

    # ======================================================================
    # 2. FONCTIONS TO CREATE TENSORS
    # ======================================================================

    def get_dataloader(self,Train=True):
        if Train :
            augmenter = DataAugmenter(max_persons=self.config['max_persons'])

            features, labels = augmenter.augment(self.features_t, self.labels_t)
            dataset = STGCNDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True,pin_memory=True,num_workers=8,persistent_workers=True, 
    prefetch_factor=2  )
            return dataloader
        else : 
            dataset = STGCNDataset(self.features_v, self.labels_v)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False,pin_memory=True,num_workers=8,persistent_workers=True, 
    prefetch_factor=2  )
            return dataloader
        
    
    # ======================================================================
    # 3. FONCTIONS CONVERT TO PYTORCH TENSOR
    # ======================================================================

    def plot_loss(self,train_losses,val_losses):
        epochs = [e for e, _ in train_losses]
        train_loss_values = [l for _, l in train_losses]
        val_loss_values = [l for _, l in val_losses]

        plt.plot(epochs, train_loss_values, label='Train Loss')
        plt.plot(epochs, val_loss_values, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train vs Validation Loss')
        plt.grid(True)
        plt.show()

    def initialize_model(self,train_loader):
        """Initialise le modèle et les composants d'entraînement"""
        model = Model(
            in_channels=self.config['in_channels'],
            num_class=self.config['num_class'],
            graph_args=self.config['graph_args'],
            edge_importance_weighting=self.config['edge_importance_weighting'],
            max_persons=self.config['max_persons']
        ).to(self.config['device'])
        alpha=torch.tensor([0.8,1])
        criterion = FocalLoss(alpha=alpha, gamma=2, reduction='mean')
        #label_smoothing=0.1,
        #weights = torch.tensor([0.85, 1], dtype=torch.float32).to(self.config['device'])
        #criterion = nn.CrossEntropyLoss(weight=weights,reduction='mean').to(self.config['device'])
        #optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=5e-2
            )
        total_steps = len(train_loader) * self.config['epochs']
        warmup_steps = int(0.1 * total_steps) 
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
        '''scheduler = CosineAnnealingLR(
    optimizer,
    T_max=self.config['epochs'],   
    eta_min= 4e-5                  
)'''

        return model, criterion, optimizer, scheduler
    def train_epoch(self, model, criterion, optimizer, train_loader, scheduler):
        torch.manual_seed(42)
        model.train()
        total_loss = 0
        num_batches_with_loss = 0

        all_preds = []
        all_labels = []

        class_correct = {cls: 0 for cls in self.class_to_idx_t.values()}
        class_total = {cls: 0 for cls in self.class_to_idx_t.values()}

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(self.config['device'])
            y_batch = y_batch.to(self.config['device'])
            optimizer.zero_grad()

            outputs = model(x_batch)
            outputs = outputs.view(-1, self.config['num_class'])
            y_batch = y_batch.view(-1)

            mask = y_batch != -1
            outputs = outputs[mask]
            y_batch = y_batch[mask]

            if len(y_batch) > 0:
                loss = criterion(outputs, y_batch)
                n_valid_samples = mask.sum().item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                num_batches_with_loss += n_valid_samples
                total_loss += loss.item() * n_valid_samples

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

                for cls in self.class_to_idx_t.values():
                    cls_mask = (y_batch == cls)
                    if cls_mask.any():
                        class_correct[cls] += (preds[cls_mask] == y_batch[cls_mask]).sum().item()
                        class_total[cls] += cls_mask.sum().item()

        avg_loss = total_loss / max(1, num_batches_with_loss)

        if len(all_labels) > 0:
            global_accuracy = 100 * accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )

            class_metrics = {}
            for cls_name, cls_idx in self.class_to_idx_t.items():
                if class_total[cls_idx] > 0:
                    cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(
                        np.array(all_labels) == cls_idx,
                        np.array(all_preds) == cls_idx,
                        average='binary',
                        zero_division=0
                    )
                    class_metrics[cls_name] = {
                        'accuracy': 100 * class_correct[cls_idx] / class_total[cls_idx],
                        'precision': cls_precision,
                        'recall': cls_recall,
                        'f1': cls_f1,
                        'support': class_total[cls_idx]
                    }
                else:
                    class_metrics[cls_name] = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'support': 0
                    }
        else:
            global_accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
            class_metrics = {cls: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
                            for cls in self.class_to_idx_t.keys()}

        return avg_loss, global_accuracy, precision, recall, f1, class_metrics


    def evaluate(self, model, criterion, val_loader):
        torch.manual_seed(42)
        model.eval()
        total_loss = 0
        num_batches_with_loss = 0
        all_preds = []
        all_labels = []

        class_correct = {cls: 0 for cls in self.class_to_idx_v.values()}
        class_total = {cls: 0 for cls in self.class_to_idx_v.values()}

        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(self.config['device'])
                y_batch = y_batch.to(self.config['device'])

                outputs = model(x_batch)
                outputs = outputs.view(-1, self.config['num_class'])
                y_batch = y_batch.view(-1)

                mask = y_batch != -1
                outputs = outputs[mask]
                y_batch = y_batch[mask]

                if len(y_batch) > 0:
                    loss = criterion(outputs, y_batch)
                    n_valid_samples = mask.sum().item()
                    num_batches_with_loss += n_valid_samples
                    total_loss += loss.item() * n_valid_samples

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

                    for cls in self.class_to_idx_v.values():
                        cls_mask = (y_batch == cls)
                        if cls_mask.any():
                            class_correct[cls] += (preds[cls_mask] == y_batch[cls_mask]).sum().item()
                            class_total[cls] += cls_mask.sum().item()

        avg_loss = total_loss / max(1, num_batches_with_loss)

        if len(all_labels) > 0:
            global_accuracy = 100 * accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )

            class_metrics = {}
            for cls_name, cls_idx in self.class_to_idx_v.items():
                if class_total[cls_idx] > 0:
                    cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(
                        np.array(all_labels) == cls_idx,
                        np.array(all_preds) == cls_idx,
                        average='binary',
                        zero_division=0
                    )
                    class_metrics[cls_name] = {
                        'accuracy': 100 * class_correct[cls_idx] / class_total[cls_idx],
                        'precision': cls_precision,
                        'recall': cls_recall,
                        'f1': cls_f1,
                        'support': class_total[cls_idx]
                    }
                else:
                    class_metrics[cls_name] = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'support': 0
                    }
        else:
            global_accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
            class_metrics = {cls: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
                            for cls in self.class_to_idx_v.keys()}

        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_to_idx_v.keys(),
                    yticklabels=self.class_to_idx_v.keys())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()

        return avg_loss, global_accuracy, precision, recall, f1, class_metrics
    def main(self):
        torch.manual_seed(42)
        if 'cuda' in self.config['device']:
            torch.cuda.empty_cache()
        
        train_loader = self.get_dataloader()
        val_loader = self.get_dataloader(Train=False)

        model, criterion, optimizer, scheduler = self.initialize_model(train_loader)
        #model, criterion, optimizer= self.initialize_model(train_loader)
        early_stopping = EarlyStopping(patience=10, delta=0.01)
        
        self.train_losses, self.val_losses = [], []
        self.train_metrics, self.val_metrics = [], []

        for epoch in range(self.config['epochs']):
            # Entraînement avec métriques
            train_loss, train_acc, train_prec, train_rec, train_f1, train_class_metrics = self.train_epoch(
                model, criterion, optimizer, train_loader
            , scheduler)
            
            if 'cuda' in self.config['device']:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Validation avec métriques
            val_loss, val_acc, val_prec, val_rec, val_f1, val_class_metrics = self.evaluate(
                model, criterion, val_loader
            )
            
            early_stopping(val_loss)
            
            # Stockage des métriques
            self.train_losses.append((epoch, train_loss))
            self.val_losses.append((epoch, val_loss))
            self.train_metrics.append((train_acc, train_prec, train_rec, train_f1))
            self.val_metrics.append((val_acc, val_prec, val_rec, val_f1))

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                torch.save(model.state_dict(), 'best_model_s.pth')
                print(f"  *--> Best model saved (val_loss: {val_loss:.4f})")
                break
                
            if val_loss == early_stopping.val_loss_min:
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"  --> Best model saved (val_loss: {val_loss:.4f})")
            
            #scheduler.step()
            
            # Affichage détaillé
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Prec: {train_prec:.2f} | Rec: {train_rec:.2f} | F1: {train_f1:.2f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Prec: {val_prec:.2f} | Rec: {val_rec:.2f} | F1: {val_f1:.2f}")
            
            print("\nTrain Class Metrics:")
            for cls, metrics in train_class_metrics.items():
                print(f"  {cls}: Acc={metrics['accuracy']:.2f}% | Prec={metrics['precision']:.2f} | Rec={metrics['recall']:.2f} | F1={metrics['f1']:.2f} | Support={metrics['support']}")
            
            print("\nVal Class Metrics:")
            for cls, metrics in val_class_metrics.items():
                print(f"  {cls}: Acc={metrics['accuracy']:.2f}% | Prec={metrics['precision']:.2f} | Rec={metrics['recall']:.2f} | F1={metrics['f1']:.2f} | Support={metrics['support']}")

        if epoch == self.config['epochs']-1:
            torch.save(model.state_dict(), 'last_model.pth')
            print("  --> Last model saved!")

        print("\nTraining complete.")   

if __name__ == "__main__":
        l=Load_data("/home/birali/Desktop/Data_check/data/train_splitOrig2.csv")
        features_tensor_t, labels_t, class_to_idx_t=l.csv_to_stgcn_tensor2()
        print(class_to_idx_t)
        l.csv_path="/home/birali/Desktop/Data_check/data/val_splitOrig2.csv"
        csv=l.csv_path
        features_tensor_v, labels_v, class_to_idx_v=l.csv_to_stgcn_tensor2()
        stgcnM=Train_stgcn(csv,features_tensor_t, labels_t, class_to_idx_t,features_tensor_v, labels_v, class_to_idx_v,learning_rate=7e-4)
        stgcnM.main()
        stgcnM.plot_loss(stgcnM.train_losses,stgcnM.val_losses)
