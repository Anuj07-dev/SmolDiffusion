from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape
        print(f"sprite shape: {self.sprites_shape}")
        print(f"labels shape: {self.slabel_shape}")
        self.transform = transform
        self.null_context = null_context
   
    def __len__(self):
        return len(self.sprites)
    
    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        return self.sprites_shape, self.slabel_shape