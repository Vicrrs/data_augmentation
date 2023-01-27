import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.utils import load_img
from matplotlib import pyplot as plt

img = load_img(r'D:\dataset_mercosul\A\A (12).png')
plt.imshow(img)
plt.show()
data = img_to_array(img)
samples = np.expand_dims(data, 0)

# horizontal flipping
datagen = ImageDataGenerator(horizontal_flip=True)
_ = datagen.flow(samples, batch_size=1)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for i in range(3):
    batch = _.next()
    image = batch[0].astype('uint8')
    plt.subplot(1, 3, i + 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()

# vertical flipping
datagen = ImageDataGenerator(vertical_flip=True)
_ = datagen.flow(samples, batch_size=1)
for i in range(3):
    batch = _.next()
    image = batch[0].astype('uint8')
    plt.subplot(130 + 1 + i)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()

datagen = ImageDataGenerator(rotation_range=90)
_ = datagen.flow(samples, batch_size=1)
for i in range(3):
    batch = _.next()
    image = batch[0].astype('uint8')
    plt.subplot(130 + 1 + i)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()

datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
_ = datagen.flow(samples, batch_size=1)
for i in range(3):
    batch = _.next()
    image = batch[0].astype('uint8')
    plt.subplot(130 + 1 + i)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()

datagen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.5,
                             rotation_range=45)
_ = datagen.flow(samples, batch_size=1)
for i in range(9):
    batch = _.next()
    image = batch[0].astype('uint8')
    plt.subplot(330 + 1 + i)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()
