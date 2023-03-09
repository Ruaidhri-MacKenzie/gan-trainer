import tensorflow as tf

import os
import time
import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# DATASET
BUFFER_SIZE = 80 # Training set sample count
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 4
IMAGE_FILE_TYPE = "png"

# LOSS AND OPTIMIZER
LAMBDA = 100
LEARNING_RATE = 2e-4
BETA_1 = 0.5

# DIRECTORIES
CHECKPOINT_DIR = "./training_checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
LOG_DIR = "logs/"

# TRAINING
BATCH_SIZE = 1 # Batch size of 1 produced better results for U-Net in original pix2pix
TRAIN_STEPS = 1000

AUTOTUNE = tf.data.AUTOTUNE

# Define the directory paths for the source and target images
source_dir = "./odyssey-sprites/train/source/"
target_dir = "./odyssey-sprites/train/target/"

# Create a list of file paths for the source and target images
source_paths = [os.path.join(source_dir, fname) for fname in os.listdir(source_dir)]
target_paths = [os.path.join(target_dir, fname) for fname in os.listdir(target_dir)]

# Define the preprocessing function for the images
def preprocess_image(image):
    image = tf.image.decode_png(image, channels=4)
    image = tf.image.resize_with_crop_or_pad(image, 64, 64)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image

# Create a TensorFlow dataset from the file paths
dataset = tf.data.Dataset.from_tensor_slices((source_paths, target_paths))

# Load the images using the file paths and the preprocessing function
dataset = dataset.map(lambda source_path, target_path: (tf.io.read_file(source_path), tf.io.read_file(target_path)))
dataset = dataset.map(lambda source_image, target_image: (preprocess_image(source_image), preprocess_image(target_image)))

# Shuffle and batch the dataset
dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

# Prefetch the dataset for improved performance
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# Print the first batch of images to verify that the dataset was loaded correctly
for source_images, target_images in dataset.take(1):
    print(f"Source images shape: {source_images.shape}, Target images shape: {target_images.shape}")


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[64, 64, 4])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 32, 32, 64)
    downsample(128, 4),  # (batch_size, 16, 16, 128)
    downsample(256, 4),  # (batch_size, 8, 8, 256)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(IMAGE_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[64, 64, 4], name='input_image')
  tar = tf.keras.layers.Input(shape=[64, 64, 4], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 64, 64, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 16, 16, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 8, 8, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
  return tf.keras.Model(inputs=[inp, tar], outputs=last)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

summary_writer = tf.summary.create_file_writer(LOG_DIR + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, steps):
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if step % 1000 == 0:
      if step != 0:
        print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')
      start = time.time()
      print(f"Step: {step // 1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step + 1) % 10 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 1000 == 0:
      checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

fit(dataset, steps=TRAIN_STEPS)

# Restore the latest checkpoint and save models
# checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
# generator.save("./models/generator")
# discriminator.save("./models/discriminator")
