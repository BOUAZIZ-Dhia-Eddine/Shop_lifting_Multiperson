import numpy as np
from scipy.interpolate import CubicSpline

class Interpolation :
    def __init__(self,T):
        self.T=T 

    def complete_missing_frames_with_spline(self,trace_tracker):
        completed_tracker = {}
        
        for person_id, frames in trace_tracker.items():
            num_frames = len(frames)
            num_nodes = frames[0].shape[0] 
            
           
            if num_frames >= self.T:
                completed_tracker[person_id] = frames
                continue
            
            
            new_frames = frames.copy()  
            
            
            if num_frames < 2:
                last_frame = frames[-1]  
                new_frames = [last_frame] * self.T
           
            else:
                time = np.arange(num_frames)  
                new_time = np.linspace(0, num_frames - 1, self.T)  
                
                new_frames_array = np.zeros((self.T, num_nodes, 3))
                

                for node in range(num_nodes):
  
                    x = np.array([frame[node, 0] for frame in frames])
                    y = np.array([frame[node, 1] for frame in frames])
                    v = np.array([frame[node, 2] for frame in frames])
                    

                    cs_x = CubicSpline(time, x)
                    cs_y = CubicSpline(time, y)
                    cs_v = CubicSpline(time, v)
                    

                    new_frames_array[:, node, 0] = cs_x(new_time)
                    new_frames_array[:, node, 1] = cs_y(new_time)
                    new_frames_array[:, node, 2] = cs_v(new_time)
                

                new_frames = [new_frames_array[t] for t in range(self.T)]
            
           
            completed_tracker[person_id] = new_frames
        
        return completed_tracker
