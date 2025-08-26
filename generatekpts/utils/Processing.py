import numpy as np
import os
import pathlib
import sys


folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils.Interpolation import Interpolation
from utils.KalmanFilter import KalmanFilter

class Processing: 
    def __init__(self):
        pass
    
    
    def calculate_angle(self,a,b,c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)/180.0
    def extract_angles_from_keypoints(self,keypoints):
        """
        keypoints: numpy array of shape (18, 2) → [x, y] pour chaque point OpenPose
        returns: dictionnaire d'angles utiles
        """
        angles = {}

        # Coudes
        angles[3] = self.calculate_angle(keypoints[2,0:2], keypoints[3,0:2], keypoints[4,0:2])
        angles[6]  = self.calculate_angle(keypoints[5,0:2], keypoints[6,0:2], keypoints[7,0:2])
        
        # Genoux
        angles[9] = self.calculate_angle(keypoints[8,0:2], keypoints[9,0:2], keypoints[10,0:2])
        angles[12]  = self.calculate_angle(keypoints[11,0:2], keypoints[12,0:2], keypoints[13,0:2])

        # Hanche / tronc
        angles[1] = self.calculate_angle(keypoints[11,0:2], keypoints[1,0:2], keypoints[8,0:2])

        # Mid-Hip
        mid_hip = (keypoints[8,0:2] + keypoints[11,0:2]) / 2
        angles[0] = self.calculate_angle(keypoints[0,0:2], keypoints[1,0:2], mid_hip)

        return angles
        
    def CalculeVitesse(self,copie_tracker):
        for key, value in copie_tracker.items():
            if len(value) < 2:  
                
                value[0][:, 0] *= value[0][:, 1]  # Magnitude
                value[0][:, 1] = 0  # Vitesse = 0
                continue  

            for i in range(len(value)):  
                angles=self.extract_angles_from_keypoints(value[i])
                value[i][:,2]=0
                for j in angles.keys(): 
                        value[i][j, 2] = angles[j]
                                      
                if i == 0:
                    
                    value[i][:, 0] *= value[i][:, 1]  # Magnitude
                    value[i][:, 1] = 0  
                else:
                    
                    dx = value[i][:, 0] - value[i - 1][:, 0]
                    dy = value[i][:, 1] - value[i - 1][:, 1]
                    
                    
                    value[i][:, 0] *= value[i][:, 1]  # Magnitude
                    value[i][:, 1] = np.abs(dx * dy) / 5  # Vitesse
                

        return copie_tracker
    
    #verification sliding_window 
    def Ajout_Padding(self,trace_tracker,dt,type=0,Sliding_Frame=30):
        
        #simple Padding add last frame
        if type == 0:
            for key, value in trace_tracker.items():
                length = len(value)
                if length < Sliding_Frame:
                    derniere_pos = value[-1]
                    padding = np.tile(derniere_pos, (Sliding_Frame - length, 1, 1))
                    trace_tracker[key] = np.vstack([value, padding])  
            return trace_tracker
        if type==1 : 
            trace_tracker=self.CalculeVitesse(trace_tracker)
            kalman = KalmanFilter(dt,Sliding_Frame)
            
            return kalman.apply_kalman_filter(trace_tracker)
        if type==2 :
            interpolation=Interpolation(Sliding_Frame)
        
            return interpolation.complete_missing_frames_with_spline(trace_tracker)
            
        

if __name__ == "__main__":
    proc=Processing()



