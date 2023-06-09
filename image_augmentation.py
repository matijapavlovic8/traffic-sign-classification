from keras.preprocessing.image import ImageDataGenerator


def augment():
    data_gen = ImageDataGenerator(width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.2,
                                  shear_range=0.1,
                                  rotation_range=10)
    return data_gen
