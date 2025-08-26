import random
import torch
from abc import ABC, abstractmethod

class PersonAwareTransform(ABC):
    def __init__(self):
        self.joint_pairs = [
            (2, 5), (3, 6), (4, 7),
            (8, 11), (9, 12), (10, 13),
            (14, 15), (16, 17)
        ]

    @abstractmethod
    def apply_transformation(self, person_features):
        raise NotImplementedError

    def __call__(self, data):
        features, labels = data
        for v_idx in range(features.shape[0]):
            for p_id in range(features.shape[4]):
                if labels[v_idx, p_id] != -1:
                    features[v_idx, :, :, :, p_id] = self.apply_transformation(
                        features[v_idx, :, :, :, p_id]
                    )
        return features, labels

class Horizontal_flip(PersonAwareTransform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def apply_transformation(self, person_features):  # shape (4, 30, 18)
        if random.random() < self.p:
            person_features[0, :, :] = 1.0 - person_features[0, :, :]
            for right, left in self.joint_pairs:
                person_features[:, :, [right, left]] = person_features[:, :, [left, right]]
        return person_features

class skeleton_translation(PersonAwareTransform):
    def __init__(self):
        super().__init__()

    def apply_transformation(self, person_features):
        x = person_features[0]
        y = person_features[1]
        mask = x != 0

        if not torch.any(mask):
            return person_features

        xmin, xmax = torch.min(x[mask]), torch.max(x[mask])
        ymin, ymax = torch.min(y[mask]), torch.max(y[mask])

        marge_right = 1.0 - xmax
        marge_left = -xmin
        marge_bottom = 1.0 - ymax
        marge_top = -ymin

        dx = torch.empty(1).uniform_(marge_left, marge_right).item()
        dy = torch.empty(1).uniform_(marge_top, marge_bottom).item()

        person_features[0] = torch.clamp(x + dx, 0.0, 1.0)
        person_features[1] = torch.clamp(y + dy, 0.0, 1.0)

        return person_features

class Randomrotation(PersonAwareTransform):
    def __init__(self, max_angle_deg=15):
        super().__init__()
        self.max_angle_deg = max_angle_deg

    def apply_transformation(self, person_features):
        x = person_features[0]
        y = person_features[1]
        mask = x != 0

        if not torch.any(mask):
            return person_features

        xG = torch.mean(x[mask])
        yG = torch.mean(y[mask])

        max_angle_rad = torch.deg2rad(torch.tensor(float(self.max_angle_deg)))
        angle = torch.empty(1).uniform_(-max_angle_rad, max_angle_rad).item()
        cos_a = torch.cos(torch.tensor(angle))
        sin_a = torch.sin(torch.tensor(angle))

        x_shifted = x - xG
        y_shifted = y - yG

        x_rotated = cos_a * x_shifted - sin_a * y_shifted + xG
        y_rotated = sin_a * x_shifted + cos_a * y_shifted + yG

        if (
            torch.all((x_rotated[mask] >= 0) & (x_rotated[mask] <= 1)) and
            torch.all((y_rotated[mask] >= 0) & (y_rotated[mask] <= 1))
        ):
            person_features[0] = x_rotated
            person_features[1] = y_rotated

        return person_features

class GaussianNoiseTransform(PersonAwareTransform):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def apply_transformation(self, person_features):
        x = person_features[0]
        y = person_features[1]
        mask = x != 0

        if not torch.any(mask):
            return person_features

        noise_x = torch.normal(mean=0.0, std=self.std, size=(1,18))
        noise_y = torch.normal(mean=0.0, std=self.std, size=(1,18))

        x_noisy = torch.clamp(x + noise_x * mask, 0.0, 1.0)
        y_noisy = torch.clamp(y + noise_y * mask, 0.0, 1.0)

        person_features[0] = x_noisy
        person_features[1] = y_noisy

        return person_features
