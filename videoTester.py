import cv2
import dlib
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import euclidean_distances
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats

model1 = load_model("model1", compile=False)
model2 = load_model("model2", compile=False)
model3 = load_model("model3", compile=False)
model4 = load_model("model4", compile=False)
model5 = load_model("model5", compile=False)

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
    predictions1 = model1.predict(resized_img)
    predictions2 = model2.predict(resized_img)
    predictions3 = model3.predict(resized_img)
    predictions4 = model4.predict(resized_img)
    predictions5 = model5.predict(resized_img)

    finalpred1 = (predictions1+predictions2 +
                  predictions3+predictions4+predictions5)/5
    predict1 = np.argmax(finalpred1, axis=1)
    print(predict1)

    finalpred2 = np.array(
        [predictions1, predictions2, predictions3, predictions4, predictions5])
    pr2 = stats.mode(finalpred2, axis=0)
    print(pr2)
    p = np.array(pr2[0])
    print(p)

    finalpred3 = (predictions1*0.1651)+(predictions2*0.2256) + \
        (predictions3*0.1865)+(predictions4*0.2089)+(predictions5*0.2138)
    print(finalpred3)
    predict3 = np.argmax(finalpred3, axis=1)
    print(predict3)
    print(predict3.shape)

    print(labels_class[np.argmax(predict1)])
    print(labels_class[np.argmax(pr2)])
    print(labels_class[np.argmax(predict3)])
    cv2.imshow("test", test_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
