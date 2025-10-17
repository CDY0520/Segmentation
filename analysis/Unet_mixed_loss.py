import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import seaborn as sns

# 설정
IMAGE_DIR = 'C:/Users/KDT_35/PycharmProjects/Oracle_Bio_Project/Mini2/Chest-X-Ray/Chest-X-Ray/image'
MASK_DIR = 'C:/Users/KDT_35/PycharmProjects/Oracle_Bio_Project/Mini2/Chest-X-Ray/Chest-X-Ray/mask'
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 20

# 1. 데이터 로드
image_paths = sorted(glob(os.path.join(IMAGE_DIR, '*.png')))
mask_paths = sorted(glob(os.path.join(MASK_DIR, '*.png')))

imgs, masks = [], []
for img_path, mask_path in zip(image_paths, mask_paths):
    img = img_to_array(load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)) / 255.
    mask = img_to_array(load_img(mask_path, color_mode='grayscale', target_size=IMG_SIZE)) / 255.
    imgs.append(img)
    masks.append(mask)

X = np.array(imgs)
y = np.array(masks)

# 2. 데이터 분할: train, val, test (0.7 / 0.15 / 0.15)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=SEED)  # 0.15 / 0.85


# 전체 데이터 shape
print("✅ 전체 데이터셋")
print("X shape:", X.shape)       # 예: (전체 샘플 수, 128, 128, 1)
print("y shape:", y.shape)

# 학습/검증/테스트 분할 후 shape
print("\n📦 분할된 데이터셋")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)


# 3. 모델 정의
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def unet(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c5 = Conv2D(512, 3, activation='relu', padding='same')(p2)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    m6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(m6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    m7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(m7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    return Model(inputs, outputs)


model = unet()

## 손실함수 mix 정의 ##
# 🔁 추가: Dice Loss 정의
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# 🔁 수정: Mixed loss 사용
def mixed_loss(y_true, y_pred):
    return 0.5 * K.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

model.compile(optimizer=Adam(1e-4), loss=mixed_loss, metrics=[dice_coef])


# 4. 학습
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    shuffle=True
)

# 5. 손실 곡선 시각화
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 6. 예측 및 평가
preds_test = model.predict(X_test)
y_pred_bin = (preds_test > 0.5).astype(np.uint8)

# 지표 계산
y_true_f = y_test.flatten()
y_pred_f = y_pred_bin.flatten()

TP = np.sum((y_true_f == 1) & (y_pred_f == 1))
TN = np.sum((y_true_f == 0) & (y_pred_f == 0))
FP = np.sum((y_true_f == 0) & (y_pred_f == 1))
FN = np.sum((y_true_f == 1) & (y_pred_f == 0))

acc = accuracy_score(y_true_f, y_pred_f)
precision = precision_score(y_true_f, y_pred_f)
recall = recall_score(y_true_f, y_pred_f)
f1 = f1_score(y_true_f, y_pred_f)
iou = jaccard_score(y_true_f, y_pred_f)
specificity = TN / (TN + FP)

print(f"Accuracy:   {acc:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"Specificity:{specificity:.4f}")
print(f"F1 Score:   {f1:.4f}")
print(f"IoU:        {iou:.4f}")

# 혼동행렬 시각화
cm = confusion_matrix(y_true_f, y_pred_f)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 예측 시각화
import random

sample_idxs = random.sample(range(len(X_test)), 5)
preds = model.predict(X_test[sample_idxs])

fig, axes = plt.subplots(5, 3, figsize=(12, 15))
for i, idx in enumerate(sample_idxs):
    axes[i, 0].imshow(X_test[idx].squeeze(), cmap='gray')
    axes[i, 0].set_title('Input')
    axes[i, 1].imshow(y_test[idx].squeeze(), cmap='gray')
    axes[i, 1].set_title('Ground Truth')
    axes[i, 2].imshow(preds[i].squeeze(), cmap='gray')
    axes[i, 2].set_title('Prediction')
    for j in range(3):
        axes[i, j].axis('off')
plt.tight_layout()
plt.show()
