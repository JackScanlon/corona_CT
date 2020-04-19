from PIL import Image 
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

DATA_TRANSFORMS = {
  'train': transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomResizedCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=[0, 0.3]),
    transforms.ToTensor()
  ]),
  'validation': transforms.Compose([
    transforms.Grayscale(),
    transforms.CenterCrop(100),
    transforms.ToTensor()
  ])
}

def load_images_from_dir(f_dir):
  results = []
  for f_name in os.listdir(f_dir):
     img = Image.open(os.path.join(f_dir, f_name))
     if img is not None:
       results.append(img)
  return results

def get_data_loader(dataset, batch_size=16, shuffle=True):
  return torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle
  )
