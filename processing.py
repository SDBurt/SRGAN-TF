import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

CROP_SIZE = 96 # Default
FACTOR = 4
HR_SIZE = 1024
TEST_SIZE = 512

# Get the dataset for this project
def get_dataset(cfg):

    CROP_SIZE = cfg.crop_size
    HR_SIZE = cfg.hr_size


    # filenames = [img for img in glob.glob('./data/HR/*.png')]
    # images = []
    # for img in filenames:
    #     n = cv2.imread(img)

    #     images.append(n)

    train_dataset = tf.data.Dataset.list_files('./data/HR/*.png', shuffle=True)
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(cfg.batch_size)
    return train_dataset.take(cfg.train_buf)


# Get the testset for this project
def get_testset(cfg):

    TEST_SIZE = cfg.test_size

    test_dataset = tf.data.Dataset.list_files('./data/LR/*.JPG',shuffle=True)
    test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(cfg.test_buf)
    return test_dataset.take(cfg.test_buf)

def load_image_train(image_file):
    images = load(image_file)
    images, _ = resize(images, HR_SIZE, HR_SIZE)
    images = random_crop(images, CROP_SIZE)
    # blurred = blur([images])
    downsized, original = resize(images, CROP_SIZE//FACTOR, CROP_SIZE//FACTOR)
    return normalize(downsized), normalize(original)

def blur(images):
    batch = tf.pack([images], name="Packed")
    convolved = tf.nn.conv2d(batch, (3,3), strides=[1, 1, 1, 1], padding='VALID')
    return convolved

def load_image_test(image_file):

    images = load(image_file)
    images, _ = resize(images, TEST_SIZE, TEST_SIZE)
    cropped = center_crop(images, 0.3)
    #images, _ = resize(images, TEST_SIZE, TEST_SIZE)
    return normalize(cropped), normalize(images)

def resize(original, height, width):
	resized = tf.image.resize(original, [height, width], method=tf.image.ResizeMethod.BICUBIC)
	return resized, original


def normalize(images):
    images = (images / 127.5) - 1
    return images

def random_crop(image, crop_size):
    size = [crop_size, crop_size, 3]
    cropped_image = tf.image.random_crop(image, size=size)
    return cropped_image

def center_crop(image, fraction):
    cropped_image = tf.image.central_crop(image, fraction)
    return cropped_image

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    return tf.cast(image, tf.float32)
