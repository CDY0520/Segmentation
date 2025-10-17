import os
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3584)]
        )
        print("Success")
    except RuntimeError as e:
        print("Fail", e)

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#### 1) 데이터 로드
X = np.load('X_images.npy')
y = np.load('y_masks.npy')

#### 3) train/validation/test 분할 ->Train: 70%, Val: 15%, Test: 15%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, shuffle=True)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

#### 4) U-Net++ 모델 정의
smooth = 1.0
def unet_plus_plus(input_size=(128, 128, 1), deep_supervision=False):
    def conv_block(x, filters):
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x

    inputs = Input(input_size)

    # Encoder
    x00 = conv_block(inputs, 64)
    x10 = conv_block(MaxPooling2D()(x00), 128)
    x20 = conv_block(MaxPooling2D()(x10), 256)
    x30 = conv_block(MaxPooling2D()(x20), 512)
    x40 = conv_block(MaxPooling2D()(x30), 512)

    # Decoder
    x01 = conv_block(concatenate([UpSampling2D()(x10), x00], axis=-1), 64)
    x11 = conv_block(concatenate([UpSampling2D()(x20), x10], axis=-1), 128)
    x21 = conv_block(concatenate([UpSampling2D()(x30), x20], axis=-1), 256)
    x31 = conv_block(concatenate([UpSampling2D()(x40), x30], axis=-1), 512)

    x02 = conv_block(concatenate([UpSampling2D()(x11), x00, x01], axis=-1), 64)
    x12 = conv_block(concatenate([UpSampling2D()(x21), x10, x11], axis=-1), 128)
    x22 = conv_block(concatenate([UpSampling2D()(x31), x20, x21], axis=-1), 256)

    x03 = conv_block(concatenate([UpSampling2D()(x12), x00, x01, x02], axis=-1), 64)
    x13 = conv_block(concatenate([UpSampling2D()(x22), x10, x11, x12], axis=-1), 128)

    x04 = conv_block(concatenate([UpSampling2D()(x13), x00, x01, x02, x03], axis=-1), 64)

    if deep_supervision:
        output1 = Conv2D(1, 1, activation='sigmoid', name='out1')(x01)
        output2 = Conv2D(1, 1, activation='sigmoid', name='out2')(x02)
        output3 = Conv2D(1, 1, activation='sigmoid', name='out3')(x03)
        output4 = Conv2D(1, 1, activation='sigmoid', name='out4')(x04)
        model = Model(inputs=inputs, outputs=[output1, output2, output3, output4])
    else:
        output = Conv2D(1, 1, activation='sigmoid', name='out')(x04)
        model = Model(inputs=inputs, outputs=output)

    return model

model = unet_plus_plus(input_size=(128, 128, 1), deep_supervision=True)

def get_dice_metric():
    def dice(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice.__name__ = 'dice'
    return dice

model.compile(
    optimizer=Adam(1e-4),
    loss=['binary_crossentropy'] * 4,
    metrics={
        'out1': get_dice_metric(),
        'out2': get_dice_metric(),
        'out3': get_dice_metric(),
        'out4': get_dice_metric(),
    }
)

# deep supervision에 맞게 label 확장
y_train_list = [y_train] * 4
y_val_list = [y_val] * 4

#### 5) 모델 학습
# EarlyStopping 정의
early_stop = EarlyStopping(
    monitor='val_loss',            # 또는 'val_out4_dice' (Dice 기준)
    patience=10,                   # 개선 없을 때 몇 epoch 기다릴지
    restore_best_weights=True,     # 가장 성능 좋았던 가중치로 복원
    verbose=1
)
history = model.fit(
    X_train, y_train_list,
    validation_data=(X_val, y_val_list),
    batch_size=8, # 수정
    epochs=20, # 수정
    shuffle=True,
    callbacks=[early_stop]
)

#### 6) 학습 결과 시각화 (Loss, Dice)
# 손실 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy')
plt.legend()

# Dice 계수 그래프 (out4 기준)
plt.subplot(1, 2, 2)
plt.plot(history.history['out4_dice'], label='Train Dice')
plt.plot(history.history['val_out4_dice'], label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.legend()

plt.tight_layout()
plt.show()

# --- 모델 평가 단계 시작 ---

# 1. Validation 셋에 대한 예측 확률 얻기 (최적 Threshold 탐색용)
y_val_pred_all = model.predict(X_val)
y_val_pred_prob = y_val_pred_all[-1] # out4 출력 사용

# 2. 최적 Threshold (Youden's J) 탐색 (Validation 셋 기준)
fpr_val, tpr_val, thresholds_val = roc_curve(y_val.flatten(), y_val_pred_prob.flatten())
youden_index_val = np.argmax(tpr_val - fpr_val)
optimal_threshold = thresholds_val[youden_index_val]
print(f"\n⭐ 최적 Threshold (Validation Youden's J): {optimal_threshold:.4f}")

# 3. Validation 셋 평가지표 계산 (선택된 최적 Threshold 적용)
y_val_pred_binary = (y_val_pred_prob > optimal_threshold).astype(np.uint8)
y_val_true_flat = y_val.flatten()
y_val_pred_flat = y_val_pred_binary.flatten()

val_dice = f1_score(y_val_true_flat, y_val_pred_flat)
val_iou = jaccard_score(y_val_true_flat, y_val_pred_flat)
val_precision = precision_score(y_val_true_flat, y_val_pred_flat)
val_recall = recall_score(y_val_true_flat, y_val_pred_flat)
val_acc = accuracy_score(y_val_true_flat, y_val_pred_flat)
cm_val = confusion_matrix(y_val_true_flat, y_val_pred_flat)
tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
val_specificity = tn_val / (tn_val + fp_val)

print("\n📊 Validation 평가 지표 (최적 Threshold 적용)")
print(f"Dice Coefficient (F1 Score): {val_dice:.4f}")
print(f"IoU (Jaccard Index):        {val_iou:.4f}")
print(f"Precision:                  {val_precision:.4f}")
print(f"Recall:                     {val_recall:.4f}")
print(f"Specificity:                {val_specificity:.4f}")
print(f"Accuracy:                   {val_acc:.4f}")

# 4. Test 셋에 대한 예측 확률 얻기
y_test_pred_all = model.predict(X_test)
y_test_pred_prob = y_test_pred_all[-1] # out4 출력 사용

# 5. Test 셋 평가지표 계산 (Validation 셋에서 찾은 최적 Threshold 적용)
y_test_pred_binary = (y_test_pred_prob > optimal_threshold).astype(np.uint8)
y_test_true_flat = y_test.flatten()
y_test_pred_flat = y_test_pred_binary.flatten()

cm_test = confusion_matrix(y_test_true_flat, y_test_pred_flat)
tn, fp, fn, tp = cm_test.ravel()
dice = f1_score(y_test_true_flat, y_test_pred_flat)
iou = jaccard_score(y_test_true_flat, y_test_pred_flat)
precision = precision_score(y_test_true_flat, y_test_pred_flat)
recall = recall_score(y_test_true_flat, y_test_pred_flat)
acc = accuracy_score(y_test_true_flat, y_test_pred_flat)
specificity = tn / (tn + fp)


print("\n🧪 Test 평가 지표 (Validation Threshold 적용)")
print(f"Dice Coefficient (F1 Score): {dice:.4f}")
print(f"IoU (Jaccard Index):        {iou:.4f}")
print(f"Precision:                  {precision:.4f}")
print(f"Recall:                     {recall:.4f}")
print(f"Specificity:                {specificity:.4f}")
print(f"Accuracy:                   {acc:.4f}")


# 6. Test 셋 ROC 커브 및 AUC 시각화 (Test 셋 자체의 ROC)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test_true_flat, y_test_pred_prob.flatten())
roc_auc_test = auc(fpr_test, tpr_test)

# Test 셋 ROC 시각화
plt.figure(figsize=(6, 5))
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# Validation 셋에서 찾은 최적 threshold에 해당하는 Test 셋의 FPR, TPR 지점 찾기 (선택 사항)
# 이 지점을 표시하려면 Test 셋의 예측 확률에 대한 FPR/TPR 배열에서 해당 threshold와 가장 가까운 지점을 찾아야 함
# 또는 단순히 optimal_threshold를 적용한 후의 실제 TPR, FPR을 표시
test_tpr_at_opt_thresh = recall # recall이 TPR과 같음
test_fpr_at_opt_thresh = fp / (fp + tn) # 1 - specificity

plt.scatter(test_fpr_at_opt_thresh, test_tpr_at_opt_thresh, color='red', s=100, marker='X', label=f'Optimal Threshold from Validation (at Test)') # 엑스마크로 표시
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Test 셋 혼동 행렬 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                              display_labels=['Background (0)', 'Foreground (1)']) # 레이블 명확화
plt.figure(figsize=(5, 4))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

#### 8) 예측 결과 시각화 (Test 셋)
num_samples = 5
indices = random.sample(range(len(X_test)), num_samples)

preds_all_test_samples = model.predict(X_test[indices])
preds_test_samples = preds_all_test_samples[-1]  # out4만 사용
preds_binary_test_samples = (preds_test_samples > optimal_threshold).astype(np.uint8)

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
for i, idx in enumerate(indices):
    axes[i,0].imshow(X_test[idx].squeeze(), cmap='gray')
    axes[i,0].set_title('Input')
    axes[i,0].axis('off')

    axes[i,1].imshow(y_test[idx].squeeze(), cmap='gray')
    axes[i,1].set_title('Ground Truth')
    axes[i,1].axis('off')

    axes[i,2].imshow(preds_binary_test_samples[i].squeeze(), cmap='gray')
    axes[i,2].set_title(f'Prediction')
    axes[i,2].axis('off')

plt.tight_layout()
plt.show()
a=1
