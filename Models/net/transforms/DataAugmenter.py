
import os 
import sys
import pathlib
import torch
import math
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transforms.PoseTransformation import Horizontal_flip,skeleton_translation,Randomrotation,GaussianNoiseTransform
class DataAugmenter:
    def __init__(self, max_persons):
        self.max_persons = max_persons
        self.flip = Horizontal_flip(p=1)
        self.translate = skeleton_translation()
        self.rotation = Randomrotation(max_angle_deg=10)
        self.rotation2= Randomrotation(max_angle_deg=10)
        self.Noise=GaussianNoiseTransform(std=0.005)
        
    def _generateTransformations(self,features_x,labels):

        augmented=[]
        for vid in range(features_x.shape[0]) :
            
            if len(features_x.shape) < 5:
                    raise ValueError(f"Expected a 5D tensor but got shape {features_x}")
            for person_id in range(features_x.shape[4]):
                if labels[vid,person_id] !=-1 : 
                    original=features_x[vid,:,:,:,person_id]
                    label=labels[vid,person_id]
                    augmented.extend([
                            (self.flip.apply_transformation(original.clone()), label),
                            (self.translate.apply_transformation(original.clone()), label),
                            (self.rotation.apply_transformation(original.clone()), label),

                            # Combinaisons de 2 transformations (ordre important)
                            (self.flip.apply_transformation(self.translate.apply_transformation(original.clone())), label),
                            (self.translate.apply_transformation(self.flip.apply_transformation(original.clone())), label),

                            (self.flip.apply_transformation(self.rotation.apply_transformation(original.clone())), label),
                            (self.rotation.apply_transformation(self.flip.apply_transformation(original.clone())), label),

                            (self.translate.apply_transformation(self.rotation.apply_transformation(original.clone())), label),
                            (self.rotation.apply_transformation(self.translate.apply_transformation(original.clone())), label),

                            # Combinaisons de 3 transformations (6 permutations possibles)
                            (self.flip.apply_transformation(self.translate.apply_transformation(self.rotation.apply_transformation(original.clone()))), label),
                            (self.flip.apply_transformation(self.rotation.apply_transformation(self.translate.apply_transformation(original.clone()))), label),

                            (self.translate.apply_transformation(self.flip.apply_transformation(self.rotation.apply_transformation(original.clone()))), label),
                            (self.translate.apply_transformation(self.rotation.apply_transformation(self.flip.apply_transformation(original.clone()))), label),

                            (self.rotation.apply_transformation(self.flip.apply_transformation(self.translate.apply_transformation(original.clone()))), label),
                            (self.rotation.apply_transformation(self.translate.apply_transformation(self.flip.apply_transformation(original.clone()))), label),
                        #(self.Noise.apply_transformation(original.clone()), label),
                        #(self.Noise.apply_transformation(self.rotation2.apply_transformation(self.flip.apply_transformation(original.clone()))), label),
                        #(self.Noise.apply_transformation(self.flip.apply_transformation(self.rotation2.apply_transformation(original.clone()))), label),
                        #(self.flip.apply_transformation(self.translate.apply_transformation(self.Noise.apply_transformation(original.clone()))), label),
                        #(self.flip.apply_transformation(self.Noise.apply_transformation(self.translate.apply_transformation(original.clone()))), label),
                        #(self.Noise.apply_transformation(self.translate.apply_transformation(original.clone())), label),
                        #(self.translate.apply_transformation(self.Noise.apply_transformation(original.clone())), label),
                        #(self.flip.apply_transformation(self.Noise.apply_transformation(original.clone())), label),
                        #(self.Noise.apply_transformation(self.flip.apply_transformation(original.clone())), label),
                        #(self.rotation2.apply_transformation(self.Noise.apply_transformation(original.clone())), label),
                        #(self.Noise.apply_transformation(self.rotation2.apply_transformation(original.clone())), label),
                    ])
                    
        return augmented 
    def _insertTransformations(self,features,labels,agmented):
        available_slots = [(v, p) for v in range(labels.shape[0])
                           for p in range(labels.shape[1]) if labels[v, p] == -1]

        for slot, (aug_feat, aug_label) in zip(available_slots, agmented):
            v_idx, p_idx = slot
            features[v_idx, :, :, :, p_idx] = aug_feat
            labels[v_idx, p_idx] = aug_label

        remaining = agmented[len(available_slots):]
        if remaining:
            #num_new = (len(remaining) + self.max_persons - 1) // self.max_persons
            num_new=math.ceil(len(remaining) / self.max_persons)
            new_features = torch.zeros((num_new, *features.shape[1:]), dtype=features.dtype)
            new_labels = torch.full((num_new, self.max_persons), -1, dtype=labels.dtype)

            for i, (feat, label) in enumerate(remaining):
                v_idx = i // self.max_persons
                p_idx = i % self.max_persons
                new_features[v_idx, :, :, :, p_idx] = feat
                new_labels[v_idx, p_idx] = label

            features = torch.cat([features, new_features], dim=0)
            labels = torch.cat([labels, new_labels], dim=0)

        return features, labels
    
        
    def augment(self, features_x, labels):
        augmented = self._generateTransformations(features_x, labels)
        return self._insertTransformations(features_x.clone(), labels.clone(), augmented)

        
