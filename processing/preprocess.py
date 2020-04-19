import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageStat
import os

def get_images(f_dir):
  result = [ ]
  for cur_file in os.listdir(f_dir):
      if cur_file.endswith('.png') or cur_file.endswith('.jpg'):
          result.append(os.path.join(f_dir, cur_file))

  return result

def preprocess(classes, size):
  for folder in classes:
      images = get_images(folder)
      for i, img in enumerate(images):
          img = Image.open(img).convert('L').resize(size, Image.ANTIALIAS)
          img.save('./data/' + folder.split('/')[2] + '/img-%d.png' % i, 'PNG')

if __name__ == "__main__":
  preprocess([
    './raw_data/CT_COVID',
    './raw_data/CT_NonCOVID'
  ], (100, 100))