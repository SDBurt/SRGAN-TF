import tensorflow as tf
import numpy as np
import h5py, os, cv2, time, PIL, argparse
from tqdm import trange
from datetime import datetime
from pathlib import Path
from config import get_config
from models import Generator, Discriminator
from processing import get_dataset, get_testset
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

# Model
parser.add_argument('--learning_rate', type=float, default=1e-4, help="optimizer learning rate")
parser.add_argument('--beta1', type=float, default=0.9, help="beta_1 variable for Adam optimizer")
parser.add_argument('--num_channels', type=int, default=3, help="number of output channels")

# Training
parser.add_argument('--train_buf', type=int, default=600, help="buffer size for training input")
parser.add_argument('--batch_size', type=int, default=10, help="batch size for training input")
parser.add_argument('--epochs', type=int, default=10, help="number of training epochs")
parser.add_argument('--pretrain_epochs', type=int, default=10, help="number of training epochs")

# Processing
parser.add_argument("--hr_size", type=int, default=1280, help="size of image before cropping")
parser.add_argument("--crop_size", type=int, default=96, help="cropped image size (square)")

# Data
parser.add_argument('--save_dir', type=str, default='./saves/', help="directory for saves")
parser.add_argument('--log_dir', type=str, default='./logs/', help="directory for tensorboard logs")
parser.add_argument("--log_freq", type=int, default=5, help="how often a log should occur (global steps)")
parser.add_argument('--extension', type=str, default=None, help="extension for logs, defaults to datetime")

# Testing / Evaluation
parser.add_argument('--test_buf', type=int, default=1, help="buffer size for testing input")
parser.add_argument("--test_size", type=int, default=768, help="size of test images")
parser.add_argument('--image_dir', type=str, default='./images/', help="image folder for saved images (per epoch)")
parser.add_argument('--num_examples_to_generate', type=int, default='16', help="number of images for grid")
parser.add_argument('--make_gif', type=bool, default=False, help="T/F make gif from ./images/")

cfg = parser.parse_args()

class SRGAN(object):

    def __init__(self, cfg):
        super(SRGAN, self).__init__()
        
        self.dataset = get_dataset(cfg)
        self.testset = get_testset(cfg)

        # Training stats
        self.global_step = 0
        
        # Models
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(cfg.learning_rate, cfg.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(cfg.learning_rate, cfg.beta1)

        # Loss
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(cfg.crop_size,cfg.crop_size) + (3,))
        self.vgg_features = tf.keras.models.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block4_conv4').output)

        # Build writers for logging
        self.build_writers()


    def build_writers(self):
        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)

        if not Path(cfg.image_dir).is_dir():
            os.mkdir(cfg.image_dir)

        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.log_path = cfg.log_dir + cfg.extension
        self.writer = tf.summary.create_file_writer(self.log_path)
        self.writer.set_as_default()

        self.save_path = cfg.save_dir + cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'

    def log_scalar(self, name, scalar):
        if self.global_step % cfg.log_freq == 0:
            tf.summary.scalar(name, scalar, step=self.global_step)

    def log_img(self, name, img, outputs=1):
        if self.global_step % (cfg.log_freq*4) == 0:
            tf.summary.image(name, img, step=self.global_step, max_outputs=outputs)


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.bce_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.bce_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, disc_generated_output, sr, hr):

        LAMBDA = 10e-3
        gen_loss = self.bce_loss(tf.ones_like(disc_generated_output), disc_generated_output)

        sr_vgg_logits = self.vgg_features(tf.keras.applications.vgg19.preprocess_input(sr))
        hr_vgg_logits = self.vgg_features(tf.keras.applications.vgg19.preprocess_input(hr))

        #sr_max = np.array(sr_vgg_logits).max()/2
        #hr_max = np.array(hr_vgg_logits).max()/2

        #vgg_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error((hr_vgg_logits/hr_max), (sr_vgg_logits/sr_max)))
        vgg_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(hr_vgg_logits, sr_vgg_logits))

        l1_loss = tf.reduce_mean(tf.abs(hr - sr))

        total_gen_loss =  vgg_loss + l1_loss + LAMBDA*gen_loss
        return total_gen_loss, vgg_loss, l1_loss, LAMBDA*gen_loss

    def train_step(self, ds, hr):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            sr = self.generator(ds)
            disc_real_output = self.discriminator(hr)
            disc_generated_output = self.discriminator(sr)

            gen_loss, vgg_loss, l1_loss, adv_loss = self.generator_loss(disc_generated_output, sr, hr)

            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        self.log_scalar("SRGAN-Generator/TOTAL", gen_loss)
        self.log_scalar("SRGAN-Generator/VGG", vgg_loss)
        self.log_scalar("SRGAN-Generator/L1", l1_loss)
        self.log_scalar("SRGAN-Generator/ADV", adv_loss)
        self.log_scalar("SRGAN-Discriminator/TOTAL", disc_loss)

        self.log_img("SRGAN/DS", (ds+1.)/2.)
        self.log_img("SRGAN/SR", (sr+1.)/2.)
        self.log_img("SRGAN/HR", (hr+1.)/2.)


    def pretrain_step(self, ds, hr):
        
        with tf.GradientTape() as gen_tape:

            sr = self.generator(ds)
            gen_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(hr, sr))

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        self.log_scalar("Pretrain/Generator", gen_loss)
        self.log_img("Pretrain/DS", (ds+1.)/2.)
        self.log_img("Pretrain/SR", (sr+1.)/2.)    
        self.log_img("Pretrain/HR", (hr+1.)/2.)

    def pretrain(self):
        print('Pretraining')
        self.global_step = 0
        for epoch in trange(cfg.pretrain_epochs):

            for ds, hr in self.dataset:
                self.pretrain_step(ds, hr)
                self.global_step += 1

    def train(self):
        print('Training')
        self.global_step = 0
        for epoch in trange(cfg.epochs):

            for ds, hr in self.dataset:
                self.train_step(ds, hr)
                self.global_step += 1

            for ds, hr in self.testset.take(1):
                self.generate_and_save_images(epoch, ds, hr)

        if cfg.make_gif == True:
            self.make_gif()

    def generate_and_save_images(self, epoch, ds, hr):
        sr = self.generator(ds)
        visual = np.array((sr[0]+1)/2.0)
        im = PIL.Image.fromarray(visual, mode='RGB')
        im.save('{}image_at_epoch_{:04d}.png'.format(cfg.image_dir,epoch))

        self.log_img("Test/SR", (sr+1)/2)
        self.log_img("Test/DS", (ds+1)/2)
        self.log_img("Test/HR", (hr+1)/2)

    def make_gif(self):
        anim_file = 'gif_srgan.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(cfg.image_dir+'image*.png')
            filenames = sorted(filenames)
            last = -1
            for i,filename in enumerate(filenames):
                frame = 2*(i**0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)


def main():
    
    srgan = SRGAN(cfg)
    srgan.pretrain()
    srgan.train()



if __name__ == '__main__':
    main()

