from torch.utils.data import Dataset
import torch
class STGCNDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (numpy array): Shape (N, C, T, V, M)
            labels (numpy array): Shape (N, M)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
