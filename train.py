import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.optimizers import SGD

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
print(train_generator.num_classes)
# MobileNet 모델을 불러옵니다. include_top=False로 설정하여 분류 레이어를 포함하지 않습니다.
base_model = MobileNet(weights='imagenet', include_top=False)

# 새로운 분류 레이어를 추가합니다.
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 특성 맵을 평균화합니다.
x = Dense(1024, activation='relu')(x)  # 완전 연결 레이어를 추가합니다.
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # 출력 레이어를 추가합니다.

# 새로운 모델을 정의합니다.
model = Model(inputs=base_model.input, outputs=predictions)

# 미리 훈련된 가중치를 사용한 MobileNet 모델의 일부를 동결합니다.
for layer in base_model.layers:
    layer.trainable = False

# 모델을 컴파일합니다.
model.compile(
    optimizer= SGD(lr=0.001, momentum=0.9), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

history = model.fit(train_generator, validation_data=validation_generator, epochs=100)

# 훈련과 검증 손실을 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 모델을 저장합니다.
model.save('mobile_net_model.h5')

# # 저장한 모델 파일의 경로 설정
# model_path = 'mobile_net_model.h5'

# # 모델 로드
# loaded_model = tf.keras.models.load_model(model_path)


# 테스트 데이터 제너레이터 생성
test_datagen = ImageDataGenerator(rescale=1./255)  # 데이터 스케일링만 수행합니다.

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # 다중 클래스 분류일 경우 'categorical'로 설정
    shuffle=False  # 테스트 데이터는 셔플하지 않습니다.
)

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_generator)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# 테스트 데이터에 대한 예측 수행
predictions = model.predict(test_generator)

# 예측 결과 확인
print("Predictions:")
print(predictions)

