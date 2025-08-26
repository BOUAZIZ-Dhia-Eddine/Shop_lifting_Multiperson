import numpy as np
import cv2 



class KalmanFilter :
    def __init__(self,dt,T):
        self.dt=dt 
        self.T=T
    def initialize_kalman_filter(self,dt=0.01):


        kf = cv2.KalmanFilter(4, 3)


        kf.transitionMatrix = np.array([
            [1, 0, self.dt * np.cos(0), 0],  # x = x + v * cos(θ) * dt
            [0, 1, self.dt * np.sin(0), 0],  # y = y + v * sin(θ) * dt
            [0, 0, 1, 0],               
            [0, 0, 0, 1]                
        ], dtype=np.float32)

        
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],  # On mesure x
            [0, 1, 0, 0],  # On mesure y
            [0, 0, 1, 0]   # On mesure v
        ], dtype=np.float32)

        
        kf.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)

        
        kf.measurementNoiseCov = 1e-4 * np.eye(3, dtype=np.float32)

        
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def apply_kalman_filter(self,trace_tracker):
        
        for person_id, frames in trace_tracker.items():
            num_nodes = frames[0].shape[0]  
            num_frames = len(frames)        

            
            kalman_filters = [self.initialize_kalman_filter() for _ in range(num_nodes)]

            for frame_idx in range(self.T):
                if frame_idx < num_frames:
                    
                    current_frame = frames[frame_idx]
                    for node_idx in range(num_nodes):
                       
                        measurement = current_frame[node_idx, :].astype(np.float32).reshape(3, 1)  # Mesure (x, y, v)
                        kalman_filters[node_idx].correct(measurement)  # Mise à jour du filtre
                        predicted_state = kalman_filters[node_idx].predict()  
                else:
                    
                    predicted_frame = np.zeros((num_nodes, 3), dtype=np.float32)
                    for node_idx in range(num_nodes):
                        predicted_state = kalman_filters[node_idx].predict() 
                      
                        predicted_frame[node_idx, :] = predicted_state[:3].flatten()
                    trace_tracker[person_id].append(predicted_frame)

        return trace_tracker


'''trace_tracker = {
    "person_id_1": [
        np.array([[1.0, 2.0, 0.5], [1.2, 2.2, 0.5], [1.4, 2.4, 0.5]]),
        np.array([[1.6, 2.6, 0.55], [1.8, 2.8, 0.55], [2.0, 3.0, 0.55]]),
        np.array([[2.2, 3.2, 0.6], [2.4, 3.4, 0.6], [2.6, 3.6, 0.6]]),
        np.array([[2.8, 3.8, 0.65], [3.0, 4.0, 0.65], [3.2, 4.2, 0.65]]),
        np.array([[3.4, 4.4, 0.7], [3.6, 4.6, 0.7], [3.8, 4.8, 0.7]]),
        np.array([[4.0, 5.0, 0.75], [4.2, 5.2, 0.75], [4.4, 5.4, 0.75]]),
        np.array([[4.6, 5.6, 0.8], [4.8, 5.8, 0.8], [5.0, 6.0, 0.8]]),
        np.array([[5.2, 6.2, 0.85], [5.4, 6.4, 0.85], [5.6, 6.6, 0.85]]),
        np.array([[5.8, 6.8, 0.9], [6.0, 7.0, 0.9], [6.2, 7.2, 0.9]])
    ],
    "person_id_2": [
        np.array([[3.0, 4.0, 0.7], [3.1, 4.1, 0.7], [3.2, 4.2, 0.7]]),
        np.array([[3.3, 4.3, 0.7], [3.4, 4.4, 0.7], [3.5, 4.5, 0.7]]),
        np.array([[3.6, 4.6, 0.7], [3.7, 4.7, 0.7], [3.8, 4.8, 0.7]]),
        np.array([[3.9, 4.9, 0.7], [4.0, 5.0, 0.7], [4.1, 5.1, 0.7]]),
        np.array([[4.2, 5.2, 0.7], [4.3, 5.3, 0.7], [4.4, 5.4, 0.7]]),
        np.array([[4.5, 5.5, 0.7], [4.6, 5.6, 0.7], [4.7, 5.7, 0.7]]),
        np.array([[4.8, 5.8, 0.7], [4.9, 5.9, 0.7], [5.0, 6.0, 0.7]]),
        np.array([[5.1, 6.1, 0.7], [5.2, 6.2, 0.7], [5.3, 6.3, 0.7]]),
        np.array([[5.4, 6.4, 0.7], [5.5, 6.5, 0.7], [5.6, 6.6, 0.7]])
    ],
    "person_id_3": [
        np.array([[2.0, 1.0, 0.8], [2.1, 1.1, 0.8], [2.2, 1.2, 0.8]]),
        np.array([[2.3, 1.3, 0.8], [2.4, 1.4, 0.8], [2.5, 1.5, 0.8]]),
        np.array([[2.6, 1.6, 0.8], [2.7, 1.7, 0.8], [2.8, 1.8, 0.8]]),
        np.array([[2.9, 1.9, 0.8], [3.0, 2.0, 0.8], [3.1, 2.1, 0.8]]),
        np.array([[3.2, 2.2, 0.8], [3.3, 2.3, 0.8], [3.4, 2.4, 0.8]]),
        np.array([[3.5, 2.5, 0.8], [3.6, 2.6, 0.8], [3.7, 2.7, 0.8]]),
        np.array([[3.8, 2.8, 0.8], [3.9, 2.9, 0.8], [4.0, 3.0, 0.8]]),
        np.array([[4.1, 3.1, 0.8], [4.2, 3.2, 0.8], [4.3, 3.3, 0.8]]),
        np.array([[4.4, 3.4, 0.8], [4.5, 3.5, 0.8], [4.6, 3.6, 0.8]])
    ],
    "person_id_4": [
        np.array([[4.0, 4.0, 0.6], [3.8, 3.8, 0.55], [3.6, 3.6, 0.5]]),
        np.array([[3.4, 3.4, 0.45], [3.2, 3.2, 0.4], [3.0, 3.0, 0.35]]),
        np.array([[2.8, 2.8, 0.3], [2.6, 2.6, 0.25], [2.4, 2.4, 0.2]]),
        np.array([[2.2, 2.2, 0.15], [2.0, 2.0, 0.1], [1.8, 1.8, 0.05]]),
        np.array([[1.6, 1.6, 0.0], [1.4, 1.4, -0.05], [1.2, 1.2, -0.1]]),
        np.array([[1.0, 1.0, -0.15], [0.8, 0.8, -0.2], [0.6, 0.6, -0.25]]),
        np.array([[0.4, 0.4, -0.3], [0.2, 0.2, -0.35], [0.0, 0.0, -0.4]]),
        np.array([[-0.2, -0.2, -0.45], [-0.4, -0.4, -0.5], [-0.6, -0.6, -0.55]]),
        np.array([[-0.8, -0.8, -0.6], [-1.0, -1.0, -0.65], [-1.2, -1.2, -0.7]])
    ],
    "person_id_5": [
        np.array([[5.0, 1.0, 0.9], [5.2, 1.2, 1.0], [5.4, 1.4, 1.1]]),
        np.array([[5.6, 1.6, 1.2], [5.8, 1.8, 1.3], [6.0, 2.0, 1.4]]),
        np.array([[6.2, 2.2, 1.5], [6.4, 2.4, 1.6], [6.6, 2.6, 1.7]]),
        np.array([[6.8, 2.8, 1.8], [7.0, 3.0, 1.9], [7.2, 3.2, 2.0]]),
        np.array([[7.4, 3.4, 2.1], [7.6, 3.6, 2.2], [7.8, 3.8, 2.3]]),
        np.array([[8.0, 4.0, 2.4], [8.2, 4.2, 2.5], [8.4, 4.4, 2.6]]),
        np.array([[8.6, 4.6, 2.7], [8.8, 4.8, 2.8], [9.0, 5.0, 2.9]]),
        np.array([[9.2, 5.2, 3.0], [9.4, 5.4, 3.1], [9.6, 5.6, 3.2]]),
        np.array([[9.8, 5.8, 3.3], [10.0, 6.0, 3.4], [10.2, 6.2, 3.5]])
    ]
}
kf=KalmanFilter(5,10)
trace_tracker = kf.apply_kalman_filter(trace_tracker)

print(trace_tracker)'''