import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random
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
          img = Image.open(img)
          img.save(folder + '/img-%d.png' % (len(images) + i), 'PNG')


def split_data(imgs):
    # 80, 20 split for [Training, Validation]
    random.shuffle(imgs)
    df = pd.DataFrame(imgs)
    return np.split(df.sample(frac=1), [int(.8*len(df))])

def split_directories(classes):
    class1, class2 = [ ], [ ]

    for i, folder in enumerate(classes):
        images = get_images(folder)
        if i == 0:
            class1 = images
        else:
            class2 = images
    
    if len(class1) != len(class2):
        lesser = min(len(class1), len(class2))
        if len(class1) == lesser:
            random.shuffle(class2)
            class2 = class2[:len(class1)]
        else:
            random.shuffle(class1)
            class1 = class1[:len(class2)]
    
    return [
        split_data(class1),
        split_data(class2)
    ]

if __name__ == "__main__":
    #preprocess([
    #    './raw_data/CT_COVID',
    #    './raw_data/CT_NonCOVID'
    #], (100, 100))

    split = split_directories([
        './data/CT_COVID',
        './data/CT_NonCOVID'
    ])

    for _, f in split[0][0].iterrows():
        os.replace(f[0], "./data/train/COVID/%s" % f[0].split('\\')[-1])
    for _, f in split[1][0].iterrows():
        os.replace(f[0], "./data/train/NONCOVID/%s" % f[0].split('\\')[-1])
    for _, f in split[0][1].iterrows():
        os.replace(f[0], "./data/validate/COVID/%s" % f[0].split('\\')[-1])
    for _, f in split[1][1].iterrows():
        os.replace(f[0], "./data/validate/NONCOVID/%s" % f[0].split('\\')[-1])