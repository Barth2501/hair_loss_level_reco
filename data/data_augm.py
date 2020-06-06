import cv2
from imgaug import augmenters as iaa
import os
from absl import app


def main(argv):
    flip = iaa.Fliplr()
    blur = iaa.GaussianBlur(sigma=(0.8,1.5))

    data_dir = ['zero','first','second','third']

    for dir in data_dir:
        images_train_dir = os.listdir('./{}'.format(dir))

        for i, image in enumerate(images_train_dir):
            img = cv2.imread('./{}/'.format(dir) + image)
            print(i)
            flipped = flip.augment_image(img)
            blurred = blur.augment_image(img)
            flipped_blurred = blur.augment_image(flipped)
            cv2.imwrite('./{}/flipped_'.format(dir) + image,flipped)
            cv2.imwrite('./{}/blurred_'.format(dir) + image,blurred)
            cv2.imwrite('./{}/flip+blur_'.format(dir) + image,flipped_blurred)

if __name__ == "__main__":
    app.run(main)    