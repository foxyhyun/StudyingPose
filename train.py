import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

def unzip_data(zip_filename, target_dir):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

# 'data' 폴더에 압축 해제
unzip_data('data/phone.zip', 'data/phone')
unzip_data('data/sleeping.zip', 'data/sleeping')
unzip_data('data/stepOut.zip', 'data/stepOut')
unzip_data('data/studying.zip', 'data/studying')

# Image Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20%를 검증 데이터로 사용
)

train_generator = datagen.flow_from_directory(
    'data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 원본 이미지 불러오기
img = load_img('data/studying/img1.jpg')  # 해당 경로에 원하는 이미지를 넣으세요
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# 증강 방법별 ImageDataGenerator 생성
augmentations = {
    'Original': ImageDataGenerator(rescale=1./255),
    'Rotation': ImageDataGenerator(rescale=1./255, rotation_range=40),
    'Width Shift': ImageDataGenerator(rescale=1./255, width_shift_range=0.2),
    'Height Shift': ImageDataGenerator(rescale=1./255, height_shift_range=0.2),
    'Horizontal Flip': ImageDataGenerator(rescale=1./255, horizontal_flip=True),
}

# 이미지 시각화
fig, axes = plt.subplots(1, len(augmentations), figsize=(20, 20))

for ax, (name, aug) in zip(axes, augmentations.items()):
    aug_img = next(aug.flow(x, batch_size=1))
    ax.imshow(aug_img[0])
    ax.set_title(name)
    ax.axis('off')

plt.show()

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(4, activation='softmax'))  # 4개의 클래스

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator)
# )
