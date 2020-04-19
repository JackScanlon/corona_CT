from covid_dataset import COVID_Dataset
from model import CNN
import numpy as np
import utils
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit

IMAGES = {
  'positive': utils.load_images_from_dir('./data/CT_COVID'),
  'negative': utils.load_images_from_dir('./data/CT_NonCOVID')
}

VAL_IMAGES = {
  'positive': utils.load_images_from_dir('./data/validate/COVID'),
  'negative': utils.load_images_from_dir('./data/validate/NONCOVID')
}

DEVICE = torch.cuda.is_available() and 'cuda' or 'cpu'

def train(model, device, train_loader, optimiser, epoch):
    # pass
    return 

def test(model, device, test_loader):
    # pass
    return 

if __name__ == '__main__':
  training_labels = np.hstack((np.ones(len(IMAGES['positive'])), np.zeros(len(IMAGES['negative']))))
  training_dataset = COVID_Dataset(
    X=IMAGES['positive'] + IMAGES['negative'], 
    y=torch.tensor(training_labels.tolist(), dtype=torch.long, device=DEVICE), 
    transform=utils.DATA_TRANSFORMS['train']
  )
  training_data_loader = utils.get_data_loader(training_dataset)

  validation_labels = np.hstack((np.ones(len(VAL_IMAGES['positive'])), np.zeros(len(VAL_IMAGES['negative']))))
  validation_dataset = COVID_Dataset(
      X=VAL_IMAGES['positive'] + VAL_IMAGES['negative'], 
      y=torch.tensor(validation_labels.tolist(), dtype=torch.long, device=DEVICE), 
      transform=utils.DATA_TRANSFORMS['validation']
  )
  validation_data_loader = utils.get_data_loader(validation_dataset)

  optimiser = nn.CrossEntropyLoss()
  # pass