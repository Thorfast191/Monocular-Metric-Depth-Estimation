import os, cv2, random
import numpy as np
from torch.utils.data import Dataset
from .transforms import to_tensor

class KITTIDataset(Dataset):
    def __init__(self, root, split, subset_ratio=1.0, seed=42):
        self.img_dir = f"{root}/{split}/image"
        self.dep_dir = f"{root}/{split}/depth"
        self.files = sorted(os.listdir(self.img_dir))

        if subset_ratio < 1.0:
            random.seed(seed)
            n = int(len(self.files) * subset_ratio)
            self.files = random.sample(self.files, n)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(f"{self.img_dir}/{self.files[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(
            f"{self.dep_dir}/{self.files[idx]}",
            cv2.IMREAD_UNCHANGED
        ).astype(np.float32)

        depth /= 256.0  # METERS

        return to_tensor(img, depth)