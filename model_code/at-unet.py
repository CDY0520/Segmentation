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

import os
import numpy as np
from PIL import ImageFile
from tensorflow.keras.utils import img_to_array, load_img

# "image file is truncated" 오류 무시 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


# 4) attention U-Net 모델 정의

from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import multiply, Lambda

smooth = 1.0
def attention_gate(x, g, inter_shape):
    # x: skip connection from encoder (e.g., c1, c2)
    # g: gating signal from deeper decoder (e.g., c6, c7)
    # inter_shape: intermediate channel depth

    # 1. g를 1x1 Conv로 처리 (phi_g)
    phi_g = Conv2D(inter_shape, 1, strides=1, padding='same')(g)
    phi_g = BatchNormalization()(phi_g)

    # 2. x를 1x1 Conv로 처리 (theta_x)
    theta_x = Conv2D(inter_shape, 1, strides=1, padding='same')(x)
    theta_x = BatchNormalization()(theta_x)

    # 3. phi_g를 theta_x와 같은 공간 해상도로 업샘플링
    # g의 해상도가 x보다 작으므로, g를 x의 해상도로 UpSampling해야 함
    # 이 부분에서 정확한 업샘플링 비율을 계산해야 합니다.
    # U-Net 구조상 (x의 height / g의 height) 비율로 UpSampling
    upsample_factor_h = x.shape[1] // phi_g.shape[1] if phi_g.shape[1] is not None else 1
    upsample_factor_w = x.shape[2] // phi_g.shape[2] if phi_g.shape[2] is not None else 1

    # Conv2DTranspose를 사용하여 UpSampling
    # kernel_size와 strides를 upsample_factor로 설정하여 업샘플링 효과
    upsampled_phi_g = Conv2DTranspose(inter_shape, kernel_size=(upsample_factor_h, upsample_factor_w),
                                      strides=(upsample_factor_h, upsample_factor_w), padding='same')(phi_g)

    # 4. 두 피처맵을 합치고 ReLU 활성화
    add_xg = tf.keras.layers.add([theta_x, upsampled_phi_g])
    act_xg = Activation('relu')(add_xg)

    # 5. 최종 어텐션 계수 생성 (psi)
    psi = Conv2D(1, 1, strides=1, padding='same')(act_xg)
    psi = BatchNormalization()(psi)
    sigmoid_xg = Activation('sigmoid')(psi) # 어텐션 맵 (0~1 값)

    # 6. 원본 x에 어텐션 계수 적용 (채널 수를 맞춰야 함)
    # sigmoid_xg (1채널)를 x의 채널 수만큼 복제하여 곱함
    # x의 채널 수는 x.shape[3] 이며, 이는 런타임에 결정되므로 tf.shape를 사용
    attention_map = Lambda(lambda z: z[0] * tf.tile(z[1], [1, 1, 1, tf.shape(z[0])[3]]))(
        [x, sigmoid_xg]
    )
    return attention_map

def attention_unet(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    # Level 1
    c1 = Conv2D(64, 3, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(64, 3, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D()(c1) # 64x64

    # Level 2
    c2 = Conv2D(128, 3, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(128, 3, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D()(c2) # 32x32

    # Level 3
    c3 = Conv2D(256, 3, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(256, 3, padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D()(c3) # 16x16

    # Level 4
    c4 = Conv2D(512, 3, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(512, 3, padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D()(c4) # 8x8 (이전 코드에는 없던 p4 추가)

    # Bottleneck
    c5 = Conv2D(1024, 3, padding='same')(p4) # p4에서 오도록 변경, 채널 수도 늘림
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(1024, 3, padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5) # 8x8

    # Decoder
    # Decoder Level 4 (upsample from c5 to c4_size)
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5) # 16x16
    g4 = Conv2D(512, 1, padding='same')(c5) # Gating signal for c4
    att4 = attention_gate(c4, g4, 512) # c4 (encoder skip)와 g4 (decoder context)
    m6 = concatenate([u6, att4])
    c6 = Conv2D(512, 3, padding='same')(m6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(512, 3, padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6) # 16x16

    # Decoder Level 3 (upsample from c6 to c3_size)
    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6) # 32x32
    g3 = Conv2D(256, 1, padding='same')(c6) # Gating signal for c3
    att3 = attention_gate(c3, g3, 256) # c3 (encoder skip)와 g3 (decoder context)
    m7 = concatenate([u7, att3])
    c7 = Conv2D(256, 3, padding='same')(m7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(256, 3, padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7) # 32x32

    # Decoder Level 2 (upsample from c7 to c2_size)
    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7) # 64x64
    g2 = Conv2D(128, 1, padding='same')(c7) # Gating signal for c2
    att2 = attention_gate(c2, g2, 128) # c2 (encoder skip)와 g2 (decoder context)
    m8 = concatenate([u8, att2])
    c8 = Conv2D(128, 3, padding='same')(m8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(128, 3, padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8) # 64x64

    # Decoder Level 1 (upsample from c8 to c1_size)
    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8) # 128x128
    g1 = Conv2D(64, 1, padding='same')(c8) # Gating signal for c1
    att1 = attention_gate(c1, g1, 64) # c1 (encoder skip)와 g1 (decoder context)
    m9 = concatenate([u9, att1])
    c9 = Conv2D(64, 3, padding='same')(m9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(64, 3, padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9) # 128x128

    outputs = Conv2D(1, 1, activation='sigmoid')(c9) # 최종 출력 (이진 분할)

    model = Model(inputs, outputs)
    return model

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


model = attention_unet()

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])


# 5) 모델 학습
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=20,
    shuffle=True,
    callbacks=[early_stopping]
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
model_name = "attention_unet"  # <- 모델 이름만 바꿔주면 됨
metric_df.to_csv(f"{model_name}_individual_metrics.csv", index=False)

print(f"📁 Saved: {model_name}_individual_metrics.csv")
