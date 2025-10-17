import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation



# 1) 데이터 로드
IMAGE_DIR = 'C:/Users/KDT_35/PycharmProjects/Oracle_Bio_Project/Mini2/Chest-X-Ray/image'
MASK_DIR = 'C:/Users/KDT_35/PycharmProjects/Oracle_Bio_Project/Mini2/Chest-X-Ray/mask'
image_paths = sorted(glob(os.path.join(IMAGE_DIR, '*.png')))
mask_paths = sorted(glob(os.path.join(MASK_DIR, '*.png')))

imgs = []
masks = []
for img_path, mask_path in zip(image_paths, mask_paths):
    img = img_to_array(load_img(img_path, color_mode='grayscale', target_size=(128,128))) / 255.0
    mask = img_to_array(load_img(mask_path, color_mode='grayscale', target_size=(128,128))) / 255.0
    imgs.append(img)
    masks.append(mask)

X = np.array(imgs)   # shape: (N,128,128,1)
y = np.array(masks)  # shape: (N,128,128,1)


# 2) Sample 데이터 시각화
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
for i in range(3):
    axes[i,0].imshow(X[i].squeeze(), cmap='gray')
    axes[i,0].set_title('Chest X-ray')
    axes[i,0].axis('off')

    axes[i,1].imshow(y[i].squeeze(), cmap='gray')
    axes[i,1].set_title('Lung Mask')
    axes[i,1].axis('off')

plt.tight_layout()
plt.show()

# 3) train/validation 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)



print("전체 데이터")
print(f"X.shape: {X.shape}")        # 전체 이미지 수, 128, 128, 1
print(f"y.shape: {y.shape}")        # 전체 마스크 수, 128, 128, 1

print("\n학습용 데이터")
print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")

print("\n검증용 데이터")
print(f"X_val.shape: {X_val.shape}")
print(f"y_val.shape: {y_val.shape}")


# 4) U-Net 모델 고도화
# - BatchNormalization (학습 안정화, 과도한 활성화 방지), Dropout (과적합 방지, 일반화 성능 향상)
# 활성함수 분리형 (conv -> BN -> ReLu) - 성능향상에 일반적으로 더 좋음

smooth = 1.0
def unet(input_size=(128,128,1)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(64, 3, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D()(c1)
    p1 = Dropout(0.1)(p1)

    c2 = Conv2D(128, 3, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(128, 3, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D()(c2)
    p2 = Dropout(0.1)(p2)

    # Bottleneck
    c5 = Conv2D(512, 3, padding='same')(p2)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(512, 3, padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    # Decoder
    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    m6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, padding='same')(m6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(128, 3, padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Dropout(0.1)(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    m7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, padding='same')(m7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(64, 3, padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Dropout(0.1)(c7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    model = Model(inputs, outputs)
    return model

model = unet()

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])

# 5) 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=20,
    shuffle=True
)

# 6) 학습 결과 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coef'], label='Train Dice')
plt.plot(history.history['val_dice_coef'], label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.legend()

plt.tight_layout()
plt.show()

# 7) 예측 결과 시각화
import random

num_samples = 5
indices = random.sample(range(len(X_val)), num_samples)
preds = model.predict(X_val[indices])

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
for i, idx in enumerate(indices):
    axes[i,0].imshow(X_val[idx].squeeze(), cmap='gray')
    axes[i,0].set_title('Input')
    axes[i,0].axis('off')

    axes[i,1].imshow(y_val[idx].squeeze(), cmap='gray')
    axes[i,1].set_title('Ground Truth')
    axes[i,1].axis('off')

    axes[i,2].imshow(preds[i].squeeze(), cmap='gray')
    axes[i,2].set_title('Prediction')
    axes[i,2].axis('off')

plt.tight_layout()
plt.show()


# 8) 평가지표 계산
# 전체 검증 세트 예측
y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(np.uint8)

# Binary segmentation 평가용으로 flatten
y_true_flat = y_val.flatten()
y_pred_flat = y_val_pred.flatten()

# 지표 계산
dice = f1_score(y_true_flat, y_pred_flat)
iou = jaccard_score(y_true_flat, y_pred_flat)
precision = precision_score(y_true_flat, y_pred_flat)
recall = recall_score(y_true_flat, y_pred_flat)
acc = accuracy_score(y_true_flat, y_pred_flat)

# 결과 출력
print("📊 Validation 평가 지표")
print(f"Dice Coefficient (F1 Score): {dice:.4f}")
print(f"IoU (Jaccard Index):        {iou:.4f}")
print(f"Precision:                  {precision:.4f}")
print(f"Recall:                     {recall:.4f}")
print(f"Accuracy:                   {acc:.4f}")

