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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import multiply, Lambda
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

smooth = 1.0

# --- Attention Gate Definition ---
def attention_gate(input_feature, gating_signal, inter_channels):
    """
    Attention Gate for U-Net.
    Args:
        input_feature (tf.Tensor): Feature map from the encoder (x).
        gating_signal (tf.Tensor): Gating signal from the deeper decoder (g).
        inter_channels (int): Number of intermediate channels for convolution.
    Returns:
        tf.Tensor: Attended feature map.
    """
    # 1x1 convolution on input_feature (x)
    theta_x = Conv2D(inter_channels, 1, strides=1, padding='same', use_bias=False)(input_feature)
    theta_x = BatchNormalization()(theta_x)

    # 1x1 convolution on gating_signal (g)
    # gating_signal은 이미 upsampling 되어 넘어왔으므로, 여기서는 1x1 conv만 적용
    phi_g = Conv2D(inter_channels, 1, strides=1, padding='same', use_bias=False)(gating_signal)
    phi_g = BatchNormalization()(phi_g)

    # Summing up theta_x and phi_g (이 둘의 공간 해상도는 같아야 함)
    # attention_gate를 호출할 때 gating_signal의 공간 해상도를 input_feature와 맞춰서 전달해야 합니다.
    # U-Net++ 구조상 gating_signal (decoder upsampled output)은 이미 input_feature (encoder skip connection)와 같은 해상도를 가집니다.
    add_xg = Add()([theta_x, phi_g]) # 수정: g_up 대신 phi_g 사용
    act_xg = Activation('relu')(add_xg)

    # Psi layer: 1x1 convolution followed by sigmoid to get attention coefficients
    psi = Conv2D(1, 1, strides=1, padding='same', use_bias=False)(act_xg)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi) # Attention coefficients (0 to 1)

    # Apply attention coefficients to the input_feature
    attended_feature = Multiply()([input_feature, psi])

    return attended_feature

# --- Convolution Block for U-Net/U-Net++ ---
def conv_block(input_tensor, num_filters):
    """
    Standard U-Net/U-Net++ convolution block (Conv -> BN -> ReLU -> Conv -> BN -> ReLU).
    """
    x = Conv2D(num_filters, 3, padding='same', use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# --- Attention U-Net++ Model Definition ---
def attention_unet_plus_plus_model(input_size=(128, 128, 1), num_classes=1):
    """
    Implements Attention U-Net++ for semantic segmentation.
    Args:
        input_size (tuple): Input image dimensions (height, width, channels).
        num_classes (int): Number of output classes (1 for binary, >1 for multi-class).
    Returns:
        tf.keras.Model: Attention U-Net++ model.
    """
    inputs = Input(input_size)

    # Encoder Path (X_0,j)
    x00 = conv_block(inputs, 32) # Level 0,0
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)

    x10 = conv_block(p0, 64) # Level 1,0
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)

    x20 = conv_block(p1, 128) # Level 2,0
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)

    x30 = conv_block(p2, 256) # Level 3,0
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)

    x40 = conv_block(p3, 512) # Level 4,0 (Bottleneck)

    # Nested Paths with Attention Gates (X_i,j where i+j=level)

    # Level 3 (from X_4,0 to X_3,1)
    # Decoder block for X_3,1
    up_x40_to_x31 = Conv2DTranspose(256, 2, strides=2, padding='same')(x40)
    # Apply attention on skip connection x30 using gating signal from up_x40_to_x31
    attn_x30 = attention_gate(x30, up_x40_to_x31, 256) # Use x30 as input_feature, up_x40_to_x31 as gating_signal
    x31 = concatenate([attn_x30, up_x40_to_x31])
    x31 = conv_block(x31, 256)

    # Level 2
    # X_2,1
    up_x30_to_x21 = Conv2DTranspose(128, 2, strides=2, padding='same')(x30) # Gating signal
    attn_x20_for_x21 = attention_gate(x20, up_x30_to_x21, 128)
    x21 = concatenate([attn_x20_for_x21, up_x30_to_x21]) # Modified to concatenate directly from up_x30_to_x21
    x21 = conv_block(x21, 128)

    # X_2,2 (nested from x21 and x31)
    up_x31_to_x22 = Conv2DTranspose(128, 2, strides=2, padding='same')(x31) # Gating signal
    attn_x21_for_x22 = attention_gate(x21, up_x31_to_x22, 128)
    x22 = concatenate([attn_x21_for_x22, up_x31_to_x22])
    x22 = conv_block(x22, 128)

    # Level 1
    # X_1,1
    up_x20_to_x11 = Conv2DTranspose(64, 2, strides=2, padding='same')(x20) # Gating signal
    attn_x10_for_x11 = attention_gate(x10, up_x20_to_x11, 64)
    x11 = concatenate([attn_x10_for_x11, up_x20_to_x11])
    x11 = conv_block(x11, 64)

    # X_1,2 (nested from x11 and x21)
    up_x21_to_x12 = Conv2DTranspose(64, 2, strides=2, padding='same')(x21) # Gating signal
    attn_x11_for_x12 = attention_gate(x11, up_x21_to_x12, 64)
    x12 = concatenate([attn_x11_for_x12, up_x21_to_x12])
    x12 = conv_block(x12, 64)

    # X_1,3 (nested from x12 and x22)
    up_x22_to_x13 = Conv2DTranspose(64, 2, strides=2, padding='same')(x22) # Gating signal
    attn_x12_for_x13 = attention_gate(x12, up_x22_to_x13, 64)
    x13 = concatenate([attn_x12_for_x13, up_x22_to_x13])
    x13 = conv_block(x13, 64)


    # Level 0
    # X_0,1
    up_x10_to_x01 = Conv2DTranspose(32, 2, strides=2, padding='same')(x10) # Gating signal
    attn_x00_for_x01 = attention_gate(x00, up_x10_to_x01, 32)
    x01 = concatenate([attn_x00_for_x01, up_x10_to_x01])
    x01 = conv_block(x01, 32)

    # X_0,2
    up_x11_to_x02 = Conv2DTranspose(32, 2, strides=2, padding='same')(x11) # Gating signal
    attn_x01_for_x02 = attention_gate(x01, up_x11_to_x02, 32)
    x02 = concatenate([attn_x01_for_x02, up_x11_to_x02])
    x02 = conv_block(x02, 32)

    # X_0,3
    up_x12_to_x03 = Conv2DTranspose(32, 2, strides=2, padding='same')(x12) # Gating signal
    attn_x02_for_x03 = attention_gate(x02, up_x12_to_x03, 32)
    x03 = concatenate([attn_x02_for_x03, up_x12_to_x03])
    x03 = conv_block(x03, 32)

    # X_0,4
    up_x13_to_x04 = Conv2DTranspose(32, 2, strides=2, padding='same')(x13) # Gating signal
    attn_x03_for_x04 = attention_gate(x03, up_x13_to_x04, 32)
    x04 = concatenate([attn_x03_for_x04, up_x13_to_x04])
    x04 = conv_block(x04, 32)


    # Output layer: uses the output from the deepest skip connection to the highest resolution
    # In U-Net++, output can be from X_0,1, X_0,2, X_0,3, or X_0,4.
    # For a deep supervision-like approach, you could have multiple outputs.
    # For single output, generally the deepest (highest 'j' value) is used.
    if num_classes == 1: # Binary segmentation
        outputs = Conv2D(1, 1, activation='sigmoid')(x04)
    else: # Multi-class segmentation
        outputs = Conv2D(num_classes, 1, activation='softmax')(x04)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Dice Coefficient for Metrics ---
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

model = attention_unet_plus_plus_model(input_size=(128, 128, 1), num_classes=1)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])

K.clear_session()
model = attention_unet_plus_plus_model(input_size=(128, 128, 1), num_classes=1)
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

model_name = "attention_unet_++"  # <- 모델 이름만 바꿔주면 됨
metric_df.to_csv(f"{model_name}_individual_metrics.csv", index=False)

print(f"📁 Saved: {model_name}_individual_metrics.csv")
