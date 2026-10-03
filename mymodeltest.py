import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


model=load_model("carbonara.h5")
img=image.load_img("test/single-red-yellow-apple-with-stem-white-background_9975-124412.jpg", target_size=(150, 150))
img_array=image.img_to_array(img)
img_array=img_array/255
img_array=np.expand_dims(img_array, axis=0)

pred=model.predict(img_array)[0]
classes=os.listdir("dataset/")
# print(classes)
# print(pred)
if pred[0]>0.5:
    print("на картинке груша")
else:
    print("на картинке яблоко")