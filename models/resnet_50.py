import tensorflow_hub as hub

from tensorflow.keras import Model
from  tensorflow.keras.layers import Dense, Softmax


# Define our model with keras model subclassing
class ResNet_50(Model):
    def __init__(self, fine_tune=False):
        super(ResNet_50, self).__init__()
        self.backbone = hub.KerasLayer(
            'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
            trainable=fine_tune,
            #output_shape=[1280],
        )
        self.dense = Dense(units=4)
        self.softmax = Softmax()
    
    def call(self, x):
        h = self.backbone(x)
        h = self.dense(h)
        y = self.softmax(h)
        return y
