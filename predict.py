import tensorflow as tf

from absl import app, flags, logging
from models.mobilenet import MobileNet
from models.resnet_50 import ResNet_50

def main(argv):
    model = ResNet_50()
    model.load_weights('./saved_model/mobilenet_001000_bs180.tf').expect_partial()
    #model = tf.keras.models.load_model('./saved_model/mobilenet_000145_bs50.h5')
    image = tf.io.read_file(FLAGS.image)
    img_raw = tf.image.decode_jpeg(image)
    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img,(224,224))
    image = tf.cast(img, tf.float32) / 256
    
    print(model.predict(image))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string('image','./third_800142.jpg',"")
    app.run(main)    