import tensorflow as tf
from layers import Residual


class Generator(tf.keras.Model):
    
    def __init__(self, cfg, num_filters=64):
        super(Generator, self).__init__()

        self.cfg = cfg

        # part 1
        self.conv1 = tf.keras.layers.Conv2D(64, 9, strides=1, padding='same')
        self.relu1 = tf.keras.layers.ReLU()

        # part 2 - Resnet blocks
        self.residual1 = Residual(cfg, num_filters)
        self.residual2 = Residual(cfg, num_filters)
        self.residual3 = Residual(cfg, num_filters)
        self.residual4 = Residual(cfg, num_filters)

        # part 3
        self.conv2 = tf.keras.layers.Conv2D(64, 9, strides=1, padding='same')
        self.bm1 = tf.keras.layers.BatchNormalization()
        
        # part 4
        self.convt1 = tf.keras.layers.Conv2DTranspose(num_filters*4, 3, strides=2, padding='same')
        self.relu2 = tf.keras.layers.ReLU()

        # part 5
        self.convt2 = tf.keras.layers.Conv2DTranspose(num_filters*4, 3, strides=2, padding='same')
        self.relu3 = tf.keras.layers.ReLU()

        # part 6
        self.out = tf.keras.layers.Conv2D(cfg.num_channels, 3, strides=1, padding='same', activation='tanh')

    def call(self, x_in):
        x1a = self.conv1(x_in)
        x1b = self.relu1(x1a)
        x2a = self.residual2(self.residual1(x1b))
        x2b = self.residual4(self.residual3(x2a))
        x3 = self.bm1(self.conv2(x2b))
        x4 = self.relu2(self.convt1(x3 + x1b))
        x5 = self.relu3(self.convt2(x4))
        return self.out(x5)
  

class Discriminator(tf.keras.Model):
    
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x):
        out = self.pipeline(x)
        out = tf.sigmoid(out)
        return out
