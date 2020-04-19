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

def explore_images(class1, class2):
  class1 = get_images(class1)
  class2 = get_images(class2)

  imgs1 = pd.DataFrame([np.asarray(Image.open(path)) for path in class1], columns=['image'])
  imgs2 = pd.DataFrame([np.asarray(Image.open(path)) for path in class2], columns=['image'])

  img_df = pd.concat([
      imgs1,
      imgs2
  ])
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.bar(['COVID', 'Non-COVID'], [len(imgs1), len(imgs2)])
  plt.show()

  img_df['width'] = img_df['image'].apply(lambda x: x.shape[0])
  img_df['height'] = img_df['image'].apply(lambda x: x.shape[1])

  hist = img_df.hist(bins=20)
  plt.setp(hist)
  plt.show()

  print(img_df[['height', 'width']].describe())


if __name__ == "__main__":
  explore_images(
    './raw_data/CT_COVID',
    './raw_data/CT_NonCOVID'
  )