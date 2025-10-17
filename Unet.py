import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from PIL import ImageFile
import os
import random
import numpy as np
import tensorflow as tf

# Seed 값 고정
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# --- 데이터 로드: .npy 파일 ---
X_train = np.load('chest_X_train.npy')
X_val   = np.load('chest_X_val.npy')
X_test  = np.load('chest_X_test.npy')

y_train = np.load('chest_y_train.npy')
y_val   = np.load('chest_y_val.npy')
y_test  = np.load('chest_y_test.npy')

print("✅ Numpy 파일 로드 완료:")
print(f"chest_X_train: {X_train.shape}, chest_y_train: {y_train.shape}")
print(f"chest_X_val  : {X_val.shape}, chest_y_val  : {y_val.shape}")
print(f"chest_X_test : {X_test.shape}, chest_y_test : {y_test.shape}")


# 4) U-Net 모델 정의
smooth = 1.0
def unet(input_size=(128,128,1)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    # Bottleneck
    c5 = Conv2D(512, 3, activation='relu', padding='same')(p2)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    m6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(m6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    m7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(m7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

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

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# 테스트셋 예측
y_pred = model.predict(X_test)
y_pred_thresh = (y_pred > 0.5).astype(np.uint8)  # threshold 적용

# 평탄화
y_true_flat = y_test.flatten()
y_pred_flat = y_pred_thresh.flatten()

# Dice (F1 score)
dice = f1_score(y_true_flat, y_pred_flat)

# IoU (Jaccard)
iou = jaccard_score(y_true_flat, y_pred_flat)

# Accuracy
acc = accuracy_score(y_true_flat, y_pred_flat)

# Precision, Recall
prec = precision_score(y_true_flat, y_pred_flat)
rec = recall_score(y_true_flat, y_pred_flat)

# Specificity (TN / (TN+FP))
cm = confusion_matrix(y_true_flat, y_pred_flat)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp)

# 결과 출력
print(f"✅ Test Score:")
print(f" - Dice (F1 score): {dice:.4f}")
print(f" - IoU           : {iou:.4f}")
print(f" - Accuracy      : {acc:.4f}")
print(f" - Precision     : {prec:.4f}")
print(f" - Recall        : {rec:.4f}")
print(f" - Specificity   : {spec:.4f}")
print(f" - Confusion Matrix:\n{cm}")

import seaborn as sns
import matplotlib.pyplot as plt

labels = ["Negative", "Positive"]
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


import pandas as pd
# ✅ 개별 지표 저장용 리스트
dice_list = []
iou_list = []

# ✅ 각 샘플마다 Dice, IoU 계산
for i in range(len(y_test)):
    y_true_i = y_test[i].squeeze().flatten()
    y_pred_i = y_pred_thresh[i].squeeze().flatten()

    dice_i = f1_score(y_true_i, y_pred_i, zero_division=0)
    iou_i = jaccard_score(y_true_i, y_pred_i, zero_division=0)

    dice_list.append(dice_i)
    iou_list.append(iou_i)

# ✅ DataFrame 저장
metric_df = pd.DataFrame({
    "Dice": dice_list,
    "IoU": iou_list
})

# ✅ 모델 이름 지정 후 CSV 저장
model_name = "unet"  # <- 모델 이름만 바꿔주면 됨
metric_df.to_csv(f"{model_name}_individual_metrics.csv", index=False)

print(f"📁 Saved: {model_name}_individual_metrics.csv")
