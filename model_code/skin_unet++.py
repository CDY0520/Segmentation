### skin 데이터에 U-net++ 모델 적용 ###


# 라이브러리 목록
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping

import cv2
from PIL import Image



### 1. 각 데이터 파일을 불러와 변수에 할당
base_dir = "Skin"

X_train = np.load(os.path.join(base_dir, "X_train_data.npy"))
X_val = np.load(os.path.join(base_dir, "X_val_data.npy"))
X_test = np.load(os.path.join(base_dir, "X_test_data.npy"))
y_train = np.load(os.path.join(base_dir, "y_train_data.npy"))
y_val = np.load(os.path.join(base_dir, "y_val_data.npy"))
y_test = np.load(os.path.join(base_dir, "y_test_data.npy"))

# 불러온 데이터의 shape 확인
print(f"불러온 X_train shape: {X_train.shape}")
print(f"불러온 y_train shape: {y_train.shape}")
print(f"불러온 X_val shape: {X_val.shape}")
print(f"불러온 y_val shape: {y_val.shape}")
print(f"불러온 X_test shape: {X_test.shape}")
print(f"불러온 y_test shape: {y_test.shape}")

print("\n모든 데이터가 성공적으로 불러와졌습니다.")


#### 2. U-Net++ 모델 정의
smooth = 1.0


# get_dice_metric 함수 정의
# 1) 병변 속성 계산 함수 추가
def calculate_lesion_properties(mask_array_2d, pixel_to_mm_ratio=None):
    """
    이진 마스크에서 병변의 픽셀 수와 면적(옵션)을 계산합니다.

    Args:
        mask_array_2d (np.array): (H, W) 형태의 이진 마스크 (0 또는 1).
        pixel_to_mm_ratio (float, optional): 1mm 당 픽셀 수.
                                             이 값이 주어지면 실제 면적을 mm^2 단위로 계산.
                                             None이면 픽셀 면적만 반환.

    Returns:
        dict: 픽셀 수, 픽셀 면적, (옵션) 실제 면적, 경계 상자 정보 (픽셀 및 mm)
    """
    # 마스크가 비어있는 경우 (병변 없음)
    if np.sum(mask_array_2d) == 0:
        return {
            'pixel_count': 0,
            'pixel_area': 0,
            'actual_area_mm2': 0,
            'bbox': None,
            'bbox_width_pixels': 0,
            'bbox_height_pixels': 0,
            'bbox_width_mm': 0,
            'bbox_height_mm': 0
        }

    # 병변 픽셀 수
    pixel_count = np.sum(mask_array_2d)
    pixel_area = pixel_count  # 픽셀 면적은 픽셀 수와 동일 (1픽셀=1단위 면적)

    actual_area_mm2 = 0
    bbox = None
    bbox_width_pixels = 0
    bbox_height_pixels = 0
    bbox_width_mm = 0
    bbox_height_mm = 0

    # 경계 상자 및 실제 면적 계산 (OpenCV 사용)
    # findContours 함수는 이진화된 이미지를 필요로 하므로, mask_array_2d를 uint8로 변환
    contours, _ = cv2.findContours(mask_array_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 가장 큰 윤곽선 (주 병변으로 가정)
        main_contour = max(contours, key=cv2.contourArea)

        # 경계 상자 (x, y, width, height)
        x, y, w, h = cv2.boundingRect(main_contour)
        bbox = {'x': x, 'y': y, 'width': w, 'height': h}
        bbox_width_pixels = w
        bbox_height_pixels = h

        if pixel_to_mm_ratio:
            # 1픽셀 당 실제 mm = 1 / pixel_to_mm_ratio
            pixel_mm_scale = 1 / pixel_to_mm_ratio
            actual_area_mm2 = pixel_area * (pixel_mm_scale ** 2)  # (픽셀 수) * (1픽셀 면적 mm^2)
            bbox_width_mm = bbox_width_pixels * pixel_mm_scale
            bbox_height_mm = bbox_height_pixels * pixel_mm_scale

    return {
        'pixel_count': pixel_count,
        'pixel_area': pixel_area,
        'actual_area_mm2': actual_area_mm2,
        'bbox': bbox,
        'bbox_width_pixels': bbox_width_pixels,
        'bbox_height_pixels': bbox_height_pixels,
        'bbox_width_mm': bbox_width_mm,
        'bbox_height_mm': bbox_height_mm
    }


# 2) 마스크 오버레이 함수 추가
def overlay_mask(original_image, mask, alpha=0.3, color=(255, 0, 0)):
    """
    원본 이미지 위에 마스크를 오버레이하여 시각화합니다.
    Args:
        original_image (PIL.Image.Image): 원본 이미지 (PIL Image 객체).
        mask (np.array): 이진 마스크 (H, W) 형태.
        alpha (float): 마스크 투명도 (0.0 투명, 1.0 불투명).
        color (tuple): 마스크 색상 (R, G, B).
    Returns:
        np.array: 오버레이된 이미지 (NumPy 배열).
    """
    # 원본 이미지를 컬러 이미지로 변환 (마스크 오버레이를 위해)
    original_image_np = np.array(original_image.convert('RGB'))

    # 마스크를 3채널 컬러 마스크로 변환 (지정된 색상)
    colored_mask = np.zeros_like(original_image_np)
    # 마스크가 1인 픽셀에만 색상 적용
    colored_mask[mask.squeeze() == 1] = color

    # 이미지와 마스크를 블렌딩
    # cv2.addWeighted를 사용하기 위해 타입 변환 (uint8)
    overlay = cv2.addWeighted(original_image_np, 1 - alpha, colored_mask, alpha, 0)
    return overlay

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



#### 3. 모델 학습

# EarlyStopping 정의
early_stop = EarlyStopping(
    monitor='val_loss',            # or 'val_out4_dice' for Dice 기준
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


#### 4. 학습 결과 시각화`
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



# 5. 모델 평가 단계

# 1) Validation 셋에 대한 예측 확률 얻기 (최적 Threshold 탐색용)
y_val_pred_all = model.predict(X_val)
y_val_pred_prob = y_val_pred_all[-1] # out4 출력 사용

# 2) 최적 Threshold (Youden's J) 탐색 (Validation 셋 기준)
fpr_val, tpr_val, thresholds_val = roc_curve(y_val.flatten(), y_val_pred_prob.flatten())
youden_index_val = np.argmax(tpr_val - fpr_val)
optimal_threshold = thresholds_val[youden_index_val]
print(f"\n⭐ 최적 Threshold (Validation Youden's J): {optimal_threshold:.4f}")

# 3) Validation 셋 평가지표 계산 (선택된 최적 Threshold 적용)
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

# 4) Test 셋에 대한 예측 확률 얻기
y_test_pred_all = model.predict(X_test)
y_test_pred_prob = y_test_pred_all[-1] # out4 출력 사용

# 5) Test 셋 평가지표 계산 (Validation 셋에서 찾은 최적 Threshold 적용)
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


# 6) Test 셋 ROC 커브 및 AUC 시각화 (Test 셋 자체의 ROC)
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
plt.title('Receiver Operating Characteristic (ROC) on Test Set')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) Test 셋 혼동 행렬 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                              display_labels=['Background (0)', 'Foreground (1)']) # 레이블 명확화
plt.figure(figsize=(5, 4))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix on Test Set')
plt.tight_layout()
plt.show()



#### 6. 예측 결과 시각화 (Test 셋) 및 속성 출력
num_samples = 5
indices = random.sample(range(len(X_test)), num_samples)

preds_all_test_samples = model.predict(X_test[indices])
preds_test_samples = preds_all_test_samples[-1]  # out4만 사용 (deep supervision의 마지막 출력)
preds_binary_test_samples = (preds_test_samples > optimal_threshold).astype(np.uint8)

# 서브플롯의 열 개수를 4개로 변경 (원본, Ground Truth, 예측 마스크, 오버레이)
fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3 * num_samples))

# 가상의 픽셀-mm 비율 설정 (이 값은 실제 데이터셋의 해상도에 맞게 조정해야 합니다!)
# ISIC 데이터셋의 경우 메타데이터에 이 정보가 있을 수 있습니다.
# 예를 들어, 128x128 픽셀 이미지가 실제 12.8mm x 12.8mm를 나타낸다면, 1mm당 10픽셀입니다.
assumed_pixel_to_mm_ratio = 10.0

for i, idx in enumerate(indices):
    input_image = X_test[idx].squeeze()
    ground_truth_mask = y_test[idx].squeeze()
    predicted_mask = preds_binary_test_samples[i].squeeze()

    # 오버레이 이미지 생성
    # PIL.Image.fromarray는 0-255 uint8 스케일을 기대하므로 변환
    overlayed_img_np = overlay_mask(Image.fromarray((input_image * 255).astype(np.uint8)).convert('L'), predicted_mask)

    # 예측 마스크의 속성 계산
    pred_props = calculate_lesion_properties(predicted_mask, pixel_to_mm_ratio=assumed_pixel_to_mm_ratio)

    # 그라운드 트루스 마스크의 속성 계산 (참고용)
    gt_props = calculate_lesion_properties(ground_truth_mask, pixel_to_mm_ratio=assumed_pixel_to_mm_ratio)

    # --- 시각화 ---
    axes[i, 0].imshow(input_image, cmap='gray')
    axes[i, 0].set_title('Input Image')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(ground_truth_mask, cmap='gray')
    # Ground Truth 마스크 제목에 픽셀 수 추가
    axes[i, 1].set_title(f'Ground Truth\nPixels: {gt_props["pixel_count"]}')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(predicted_mask, cmap='gray')
    # 예측 마스크 제목에 픽셀 수와 (있다면) 실제 면적 정보 추가
    pred_title = f'Prediction\nPixels: {pred_props["pixel_count"]}'
    if pred_props['actual_area_mm2'] > 0:
        pred_title += f'\nArea: {pred_props["actual_area_mm2"]:.2f} mm²'  # 소수점 두 자리까지

    axes[i, 2].set_title(pred_title)
    axes[i, 2].axis('off')

    axes[i, 3].imshow(overlayed_img_np)
    # 오버레이 이미지 제목에 계산된 정보 추가
    overlay_title = f"Overlay\n" \
                    f"Pred Pixels: {pred_props['pixel_count']}\n" \
                    f"Pred BBox (px): {pred_props['bbox_width_pixels']}x{pred_props['bbox_height_pixels']}"
    if assumed_pixel_to_mm_ratio:
        overlay_title += f"\nPred Area (mm²): {pred_props['actual_area_mm2']:.2f}\n" \
                         f"Pred BBox (mm): {pred_props['bbox_width_mm']:.2f}x{pred_props['bbox_height_mm']:.2f}"

    axes[i, 3].set_title(overlay_title)
    axes[i, 3].axis('off')

plt.tight_layout()
plt.show()



### 7. 손실함수 리스트 출력해서 dataframe으로 저장
dice_list = []
iou_list = []

for i in range(len(y_test)):
    y_true_i = y_test[i].squeeze().flatten()
    # y_pred_thresh 대신 y_test_pred_binary 사용
    y_pred_i = y_test_pred_binary[i].squeeze().flatten()

    dice_i = f1_score(y_true_i, y_pred_i, zero_division=0)
    iou_i = jaccard_score(y_true_i, y_pred_i, zero_division=0)

    dice_list.append(dice_i)
    iou_list.append(iou_i)

import pandas as pd

metric_df = pd.DataFrame({
    "Dice": dice_list,
    "IoU": iou_list
})

print("\n각 테스트 샘플에 대한 Dice 및 IoU 점수 DataFrame:")
print(metric_df.head()) # 상위 5개 행 출력

# 데이터프레임을 CSV 파일로 저장
csv_file_path = "skin_unet++_dataframe.csv"
metric_df.to_csv(csv_file_path, index=False)

print(f"\nDataFrame이 '{csv_file_path}' 파일로 성공적으로 저장되었습니다.")