import numpy as np
import cv2
import pickle

from preprocessor import preprocess

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.90
font = cv2.FONT_HERSHEY_DUPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

class_names = {
    0: 'Speed Limit 20 km/h',
    1: 'Speed Limit 30 km/h',
    2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h',
    4: 'Speed Limit 70 km/h',
    5: 'Speed Limit 80 km/h',
    6: 'End of Speed Limit 80 km/h',
    7: 'Speed Limit 100 km/h',
    8: 'Speed Limit 120 km/h',
    9: 'No passing',
    10: 'No passing for vechiles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 tons'
}


while 1:
    success, img_original = cap.read()

    img = np.asarray(img_original)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    class_index = np.argmax(predictions, axis=1)
    probability = np.amax(predictions)
    if probability < threshold:
        cv2.putText(img_original, "No sign detected currently!", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img_original, "Detected " + class_names.get(class_index.item()) + " sign.", (20, 35), font, 0.75,
                    (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(img_original, "Probability: " + str(round(probability * 100, 2)) + "%", (20, 75), font, 0.75, (0, 0, 255), 2,
                cv2.LINE_AA)
        

    cv2.imshow("Traffic sign classification", img_original)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
