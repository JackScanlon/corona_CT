import torch

class COVID_Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y, transform=None):
    super(COVID_Dataset, self).__init__()
    
    self.X = X
    self.y = y
    self.transform = transform

  def __getitem__(self, idx):
    X = self.X[idx]
    if self.transform:
      X = self.transform(self.X[idx])
    return X, self.y[idx]

  def __len__(self):
    return len(self.X)
