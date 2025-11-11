# Segmentation X-ray Segmentation Project

엑스레이 영상 기반 폐 영역 분할 모델 개발 및 일반화 실험

---

# 프로젝트 개요

목적
병원별 영상 해상도·촬영조건이 달라 일반화된 분할 성능 확보가 어려운 의료영상(X-ray) 에서,
다양한 딥러닝 분할 모델(CNN·Transformer 기반)을 비교하고
가장 일반화 성능이 우수한 구조를 탐색하는 것이 목표이다.

연구 배경
딥러닝 기반 분할 기술 발전으로 X-ray 영역 분석이 활발해짐
Transformer 구조 도입으로 global feature 학습 가능
그러나 X-ray는 CT 대비 정보량이 적고 경계가 불명확하여
모델이 과적합되거나 데이터 편향에 민감 → 일반화 성능 확보 어려움

---

# 연구 흐름

기본 데이터(Chest X-ray) 로 모델 학습
U-Net 계열 여러 모델 성능 비교 (Dice Score 기반)
최적 모델 선정 (UNet++ 우수 성능 확인)
UNet++ 모델을 타 데이터셋에 적용하여 일반화 성능 평가

아래 데이터 간 성능 변화 및 구조적 차이 분석
Heart X-ray (심장 분할)
COVID lesion (폐 병변 부위)
Skin lesion (피부 병변 부위)

---

# 폴더 구조

```
Segmentation/
│
├── analysis/                     # 성능 고도화 및 일반화 실험 분석 코드
│   ├── Unet_batchnorm_dropout.py
│   ├── Unet_mixed_loss.py
│   ├── loss_dice_comparison_chest.py
│   ├── loss_dice_comparison_heart.py
│   ├── loss_dice_comparison_covid.py
│   └── loss_dice_comparison_skin.py
│
├── data/                         # 학습 및 일반화 실험용 데이터셋
│   ├── Chest X-ray_split.zip     # 기본 학습 데이터
│   ├── Heart X-ray_split.zip     # 심장 일반화 데이터
│   ├── covid_split.zip           # 폐 병변 일반화 데이터
│   ├── Skin_split.zip            # 피부 병변 일반화 데이터
│   └── plot_training_metrics.py  # 데이터 분할 코드
│
├── model_code/                   # 사용된 모델 아키텍처 코드
│   ├── base_Unet.py
│   ├── at-unet.py
│   ├── at-unet++.py
│   ├── chest-x-ray_unet++.py
│   ├── heart_unet++.py
│   ├── covid_unet++.py
│   ├── skin_unet++.py
│   ├── DeepLabV3+.py
│   ├── segformer.py
│   └── transunet.py
│
└── README.md
```

---

# 주요 실험 내용

폐 분할 (Chest X-ray)
Transformer 기반 모델은 CNN보다 평균 성능 낮음
소규모 이미지(128×128) 에서는 local feature 학습 중심의 CNN 구조가 더 유리
CNN은 patch 단위 손실 최소화에 효율적 → UNet++이 최적

심장 분할 (Heart X-ray)
Transformer 기반 모델(global feature 강점)은 심장처럼 경계가 명확한 구조에서 과적합 위험
데이터 수가 적을 때 CNN 기반 구조가 더 일반화 안정적
Transformer 모델은 충분한 데이터가 없을 경우 overfitting 경향

피부 병변 분할 (Skin)
데이터 차이가 크고 표면 구조 다양 → 전반적으로 Dice 80%대 상한
모든 모델 간 성능 차 미미, 데이터 다양성·크기 부족이 주원인
작은 데이터셋에서는 모델 복잡도보다 데이터 품질이 주요 변수

COVID 병변 분할 (CT X-ray)
경계 불명확, 샘플 적음 → Dice·IoU 모두 낮음
SwinUNet, TransUNet 등 Transformer 계열은 학습 불안정
CNN 기반 구조가 더 안정적이며 일관된 성능 유지
비교적 넓은 병변 영역에서 Dice 분산 높음, 모델 간 편차 존재

---

# 최종 결론

UNet++ 이 전체 실험에서 가장 일관적이고 높은 Dice score를 보였음
Transformer 계열 모델은 데이터가 충분하고 경계가 명확할 때 효과적
의료 영상의 특성상, 데이터 크기·품질·기관별 촬영 차이를 고려한 hybrid 구조(CNN+Attention) 가 향후 방향성으로 제시됨
