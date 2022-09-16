import cv2
import dlib
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import euclidean_distances
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf

model = load_model("model2", compile=False)

labels_class = ['N', 'O', 'R']

cap = cv2.VideoCapture(1)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (128, 128))
    resized_img = resized_img/255.0
    # image to shape of (128,128,3)
    resized_img = np.expand_dims(resized_img, axis=0)
    # predicting the image
    predictions = model.predict(resized_img)
    print(labels_class[np.argmax(predictions)])
    print((predictions))
    cv2.imshow("test", test_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
