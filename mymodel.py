import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


DATASET_PATH="dataset/"
num_class=len(os.listdir(DATASET_PATH)) 
print(num_class)

#ПОДГОТОВКА ИЗОБРАЖЕНИЙ
 
datagen=ImageDataGenerator(
    rescale=1/255, validation_split=0.1, rotation_range=20, zoom_range=0.2, horizontal_flip=True
)


train_data=datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(150,150),
    batch_size=32, #сколько картинок нейросеть возьмет за один раз для обучения(порции, которыми учится нейросеть)
    class_mode="binary", #class_mode показывает сколлько классов мы определяем, если 2 - binary, если больше 2 - categorical
    subset="training" #для чего будут использоваться данные
)

val_data=datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(150,150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

#СОЗДАНИЕ МОДЕЛИ
model=Sequential([
    #слои
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)), #ищет важные признаки картинки
    MaxPooling2D(2, 2),  #уменьшает размер картинки в 2 раза
    Conv2D(64, (3, 3), activation="relu"), #ищет важные признаки картинки
    MaxPooling2D(2, 2), 
    Flatten(), #выстраивает все признаки в ряд
    Dense(128, activation="relu"),  
    Dense(1, activation="sigmoid") 
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) #компилируем модель
model.fit(train_data, validation_data=val_data, epochs=10) #обучение модели
model.save("carbonara.h5")