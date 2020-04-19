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

def get_brightness(img):
   img = Image.open(img).convert('L')
   return ImageStat.Stat(img).mean[0]

def explore_brightness(class1, class2):
  imgs1 = get_images(class1)
  imgs2 = get_images(class2)

  m1, m2 = [], []
  for img in imgs1:
    brightness = get_brightness(img)
    m1.append(brightness)
  for img in imgs2:
    brightness = get_brightness(img)
    m2.append(brightness)
  
  std1 = np.std(m1)
  std2 = np.std(m2)
  
  plt.figure()
  plt.plot(m1)
  plt.show()

  plt.figure()
  plt.plot(m2)
  plt.show()

  m1 = np.mean(m1)
  m2 = np.mean(m2)

  print('Mean of COVID: {co}  |  Mean of Non-COVID: {noco}'.format(co=m1, noco=m2))
  print(' STD of COVID: {co}  |   STD of Non-COVID: {noco}'.format(co=std1, noco=std2))


if __name__ == "__main__":
  explore_brightness(
    './raw_data/CT_COVID',
    './raw_data/CT_NonCOVID'
  )