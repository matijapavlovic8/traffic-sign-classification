from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


def learning_model(classes_len):

    model = Sequential()
    model.add((Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu')))
    model.add((Conv2D(60, (5, 5), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add((Conv2D(30, (2, 2), activation='relu')))
    model.add((Conv2D(30, (2, 2), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_len, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
