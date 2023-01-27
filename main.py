import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image

data_gen = ImageDataGenerator(rotation_range=40,
                              width_shift_range=[-100, 100],
                              height_shift_range=[-100, 100],
                              shear_range=0.2,
                              zoom_range=[0.5, 1.0],
                              horizontal_flip=True,
                              fill_mode='nearest')

data_gen.flow_from_directory(directory=r"D:\dataset_mercosul\A",
                             target_size=(224, 224),
                             batch_size=32,
                             class_mode='binary',
                             save_to_dir=r"D:\dataset_mercosul\A_data")

X = np.random.rand(100, 28, 28, 3)
y = np.random.randint(0, 2, 100)

image_count = 0
for images, labels in data_gen.flow(X, y, batch_size=32):
    for i in range(len(images)):
        images = np.uint8(images)
        images = np.squeeze(images)
        img = Image.fromarray(images[i])
        img.save(r'D:\dataset_mercosul\A_data\image_{}.png'.format(i))
        image_count += 1
        if image_count >= 32:
            break
    if image_count >= 32:
        break
