from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import csv_reader
import image_loader
import preprocessor
from learning_model import learning_model
from image_augmentation import augment

path = "dataset/myData/myData"
labels = 'labels_file/labels.csv'
batch_size = 50
steps_per_epoch_val = 1000
epochs_val = 10
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

loaded = image_loader.load(path)
images = loaded['images']
class_no = loaded['class_no']
classes_len = loaded['classes_len']

X_train, X_test, y_train, y_test = train_test_split(images, class_no, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

data = csv_reader.read(labels)

X_train = np.array(list(map(preprocessor.preprocess, X_train)))
X_validation = np.array(list(map(preprocessor.preprocess, X_validation)))
X_test = np.array(list(map(preprocessor.preprocess, X_test)))


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

data_gen = augment()
data_gen.fit(X_train)
batches = data_gen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, classes_len)
y_validation = to_categorical(y_validation, classes_len)
y_test = to_categorical(y_test, classes_len)

model = learning_model(classes_len)
print(model.summary())
history = model.fit(cycle(data_gen.flow(X_train, y_train, batch_size=batch_size)),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation), shuffle=1)


score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
