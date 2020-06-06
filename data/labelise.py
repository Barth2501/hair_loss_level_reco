import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
from PIL import Image

ds = tfds.load('celeb_a', split='train',shuffle_files=True)
ds = ds.batch(1)
count=800590
for example in tfds.as_numpy(ds):
    image, label = example["image"], example["attributes"]
    if label['Male'][0]==True:
        plt.imshow(image[0])
        plt.show()
        image = Image.fromarray(image[0])
        level = input("which level ?")
        try:
            level = int(level)
            if level==1:
                image.save('./{}/first_{}.jpg'.format('first',count))
            elif level==2:
                image.save('./{}/second_{}.jpg'.format('second',count))
            elif level==3:
                image.save('./{}/third_{}.jpg'.format('third',count))
            elif level==0:
                image.save('./{}/zero_{}.jpg'.format('zero',count))
            else:
                continue
            count+=1
        except:
            continue