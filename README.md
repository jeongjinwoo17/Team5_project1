# 💊 알약 객체 탐지 프로젝트 (YOLOv12s)

여러 모델을 활용하여 알약 이미지 객체 탐지를 진행하고 
가장 성능이 좋은 모델을 대표 모델로 선정하여 Github에 
공개하였습니다.

## 📌 프로젝트 소개

- 알약 이미지에서 알약의 위치와 종류를 자동으로 탐지합니다.
- YOLOv12s 모델 기반 학습 및 추론
- 데이터 증강(Augmentation)을 통한 소수 클래스 불균형 해소
- 제출용 CSV 자동 생성 기능 포함

## 👥 팀원

| 이름 | 사용한 모델 |
|------|------|
| 이수호 | YOLOv12s |
| 이아인 | YOLOv8, YOLOv12 |
| 이용빈 | YOLOv2 |
| 조영한 | Faster R-CNN, ResNet18, EfficiiientNet-B0, EasyOCR |
| 정진우 | RT-DETR |

## 🛠 사용 기술

- Python 3.10+
- YOLOv12s (Ultralytics)
- PyTorch
- Albumentations
- OpenCV
- Pandas / NumPy / Matplotlib

## 📁 폴더 구조

```
project1/
├── src/
│   └── train.py              # 전처리 + 증강 + 학습 + 평가 + CSV 생성
├── data/
│   └── (데이터셋 - 아래 링크 참고)
├── runs/                     # 학습 결과 자동 생성 (git 미포함)
├── requirements.txt          # 필요 라이브러리
├── .gitignore
└── README.md
```

## 📦 데이터셋

용량 문제로 GitHub에 직접 포함하지 않습니다.

👉 **Kaggle 대회 페이지**에서 Join Competition 후 `kagglehub`로 자동 다운로드됩니다.

코드 실행 시 아래 순서로 자동 처리됩니다:

```
1. Kaggle API 키 설정 (최초 1회)
2. python src/train.py 실행
3. kagglehub가 자동으로 데이터 다운로드 및 경로 설정
```

### Kaggle API 키 설정 방법

1. [kaggle.com](https://www.kaggle.com) 로그인
2. 우측 상단 프로필 → **Settings → API → Create New Token**
3. 다운로드된 `kaggle.json`을 아래 경로에 저장

| OS | 경로 |
|----|------|
| Windows | `C:\Users\본인이름\.kaggle\kaggle.json` |
| Mac/Linux | `~/.kaggle/kaggle.json` |

## ⚙️ 환경 설정

```bash
# 1. 저장소 클론
git clone https://github.com/jeongjinwoo17/Team5_project1.git
cd Team5_project1

# 2. 라이브러리 설치
pip install -r requirements.txt
```

## 🚀 실행 방법

```bash
# 학습 실행 (데이터 전처리 → 증강 → 학습 → 평가 → CSV 생성 순서로 자동 진행)
python src/train.py
```

## 📊 모델 학습 설정

| 파라미터 | 값 |
|----------|-----|
| 모델 | YOLOv12s |
| Epochs | 200 |
| Patience | 20 |
| Image Size | 1280 |
| Batch Size | 4 |
| Train/Val 비율 | 80:20 |
| Mosaic | 1.0 |
| Mixup | 0.15 |

## 🔍 데이터 증강 종류

- Horizontal Flip
- Vertical Flip + Rotate
- Gaussian Noise
- Brightness/Contrast
- Color Jitter (HSV)
- Blur/Sharpen
- Geometric (ShiftScaleRotate + Perspective)

> 소수 클래스(15장 미만)는 8배 추가 증강 적용

## 📝 출력 결과

- `runs/detect/train/weights/best.pt` — 최고 성능 모델 가중치
- `submission.csv` — 제출용 예측 결과 파일
