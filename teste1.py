from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.utils import load_img

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')

img = load_img(r'D:\dataset_mercosul\B\B (5).png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=r'D:\dataset_mercosul\B_data', save_prefix='b1',
                          save_format='jpeg'):
    i += 1
    if i > 20:
        break
