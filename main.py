import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import pickle
import random
import csv_reader
import image_loader
import preprocessor
from learning_model import learning_model
from image_augmentation import augment

path = "datatest"  # folder with all the class folders
labels = 'labels_file/labels.csv'  # file with all names of classes
batch_size = 50
steps_per_epoch_val = 2000
epochs_val = 10
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

loaded = image_loader.load(path)
images = loaded['images']
class_no = loaded['class_no']
classes_len = loaded['classes_len']

############################### Split Data
X_train, X_test, y_train, y_test = train_test_split(images, class_no, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID

############################### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("Data Shapes")
print("Train", end="")
print(X_train.shape, y_train.shape)
print("Validation", end="")
print(X_validation.shape, y_validation.shape)
print("Test", end="");
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[
    0]), "The number of images in not equal to the number of lables in training set"
assert (X_validation.shape[0] == y_validation.shape[
    0]), "The number of images in not equal to the number of lables in validation set"
assert (X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert (X_train.shape[1:] == (imageDimesions)), " The dimesions of the Training images are wrong "
assert (X_validation.shape[1:] == (imageDimesions)), " The dimesionas of the Validation images are wrong "
assert (X_test.shape[1:] == (imageDimesions)), " The dimesions of the Test images are wrong"

############################### READ CSV FILE
data = csv_reader.read(labels)

############################### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
# print(num_of_samples)
# plt.figure(figsize=(12, 4))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("Distribution of the training dataset")
# plt.xlabel("Class number")
# plt.ylabel("Number of images")
# plt.show()


############################### PREPROCESSING THE IMAGES

# def grayscale(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img
#
#
# def equalize(img):
#     img = cv2.equalizeHist(img)
#     return img
#
#
# def preprocessing(img):
#     img = grayscale(img)  # CONVERT TO GRAYSCALE
#     img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
#     img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
#     return img


X_train = np.array(list(map(preprocessor.preprocess, X_train)))  # TO ITERATE AND PREPROCESS ALL IMAGES
X_validation = np.array(list(map(preprocessor.preprocess, X_validation)))
X_test = np.array(list(map(preprocessor.preprocess, X_test)))
cv2.imshow("GrayScale Images",
           X_train[random.randint(0, len(X_train) - 1)])  # TO CHECK IF THE TRAINING IS DONE PROPERLY

############################### ADD A DEPTH OF 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


# dataGen = ImageDataGenerator(width_shift_range=0.1,
#                              # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
#                              height_shift_range=0.1,
#                              zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
#                              shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
#                              rotation_range=10)  # DEGREES

data_gen = augment()
data_gen.fit(X_train)
batches = data_gen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# TO SHOW AUGMENTED IMAGE SAMPLES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, classes_len)
y_validation = to_categorical(y_validation, classes_len)
y_test = to_categorical(y_test, classes_len)


############################### TRAIN
model = learning_model(classes_len)
print(model.summary())
history = model.fit_generator(data_gen.flow(X_train, y_train, batch_size=batch_size),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# STORE THE MODEL AS A PICKLE OBJECT
pickle_out = open("model_trained.p", "wb")  # wb = WRITE BYTE
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)