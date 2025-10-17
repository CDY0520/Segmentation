# Segmentation X-ray Segmentation Project

엑스레이 영상 기반 폐 영역 분할 모델 개발 및 일반화 실험

---

# 프로젝트 개요

목적
의료 영상에서 폐 영역을 정밀하게 분할하는 딥러닝 모델을 개발하고,
여러 질환 및 장기 데이터로 일반화 성능을 검증하는 것이 목표이다.

본 프로젝트에서는 U-Net 계열 모델(UNet, UNet++, DeepLabV3+, SegFormer 등) 을 비교하여
가장 높은 성능을 보이는 모델을 선정한 후,
해당 모델의 다른 데이터셋(심장, COVID-19, 피부 병변 등) 에 대한
일반화 가능성(Generalization) 을 평가하였다.

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

---

# 주요 실험 내용
모델 비교:	UNet, UNet++, DeepLabV3+, SegFormer, TransUNet 등
성능 지표:	Dice Similarity Coefficient (DSC)
최고 성능 모델:	UNet++ — Chest X-ray 데이터에서 가장 높은 Dice 점수 기록
일반화 실험:	UNet++을 Heart, COVID, Skin 데이터에 적용 후 Dice Score 분석
결과 해석:	기본 unet 모델만으로도 충분히 다른 데이터에 성능 유지 가능함을 확인
