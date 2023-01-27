import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.utils import load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

path = r'D:\dataset_mercosul\B'

image_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

i = 0
for image_path in image_list:
    img = load_img(image_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    for batch in datagen.flow(x, batch_size=1, save_to_dir=r'D:\dataset_mercosul\B_data', save_prefix='a1',
                              save_format='jpeg'):
        i += 1
        if i > 1000:
            break
