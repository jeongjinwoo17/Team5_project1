# 💊 경구 알약 탐지 프로그램

여러 모델을 활용하여 알약 이미지 객체 탐지를 진행하고 
가장 성능이 좋은 모델을 대표 모델로 선정하여 Github에 
공개하였습니다.
데이터는 kagglehub으로 자동 다운로드되며, 로컬 PC에서 바로 실행 가능합니다.

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

## 📁 프로젝트 구조

```
├── train.py                            # 메인 실행 파일 (전체 파이프라인 호출)
├── README.md
├── requirements.txt
├── submission.csv                      # 실행 후 자동 생성
├── images/                             # 결과 이미지
│   ├── evaluation_metric_graph.png     # Train Loss / Validation Metrics 그래프
│   ├── test_data_confidence.png        # Prediction Confidence Distribution
│   ├── test_image_1.png               # 테스트 이미지 결과 (1~5번)
│   ├── test_image_2.png               # 테스트 이미지 결과 (6~10번)
│   ├── test_image_3.png               # 테스트 이미지 결과 (11~15번)
│   └── test_image_4.png               # 테스트 이미지 결과 (16~20번)
└── src/
    ├── config.py                       # 경로 및 하이퍼파라미터 설정
    ├── data_loader.py                  # Step 1 - kagglehub 데이터 다운로드
    ├── dataset.py                      # Step 2 - 클래스 맵핑 + Train/Val 분리
    ├── augmentation.py                 # Step 3 - 오프라인 데이터 증강
    ├── trainer.py                      # Step 4·5 - data.yaml 생성 + YOLO 학습
    ├── visualizer.py                   # Step 6·7 - Loss·mAP 시각화 + 테스트 예측
    ├── evaluator.py                    # Train 자체 채점 (IoU 기반 TP/FP/FN)
    └── submission.py                   # Step 8 - 제출용 CSV 생성
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

## ⚙️ 설치

```bash
pip install kagglehub ultralytics albumentations opencv-python torchmetrics tqdm pandas matplotlib
```

> kagglehub 사용을 위해 Kaggle API 키가 필요합니다.  
> [kaggle.com → Settings → API → Create New Token](https://www.kaggle.com/settings) 에서 `kaggle.json` 다운로드 후  
> Windows: `C:\Users\사용자이름\.kaggle\kaggle.json` 에 위치시키세요.

## 🚀 실행

```bash
python train.py
```

## 📋 파이프라인 순서

| 파일 | Step | 내용 |
|------|------|------|
| `data_loader.py`  | 1 | kagglehub 데이터 자동 다운로드 |
| `dataset.py`      | 2 | 클래스 맵핑 생성 + Train/Val 분리 (80:20) |
| `augmentation.py` | 3 | 오프라인 데이터 증강 |
| `trainer.py`      | 4·5 | data.yaml 생성 + YOLO12s 학습 |
| `visualizer.py`   | 6·7 | Loss·mAP 시각화 + 테스트 예측 |
| `evaluator.py`    | 검증 | Train 채점 (IoU 기반 TP/FP/FN) |
| `submission.py`   | 8 | 제출용 submission.csv 생성 |

## ⚡ 주요 하이퍼파라미터

`src/config.py` 에서 수정하면 전체 파이프라인에 반영됩니다.

| 항목 | 기본값 |
|------|--------|
| 모델 | YOLO12s |
| epochs | 200 |
| patience | 20 |
| imgsz | 1280 |
| batch | 4 |
| mosaic / mixup | 1.0 / 0.15 |
| degrees | 40.0 |
| 소수 클래스 기준 | 6장 이하 → 8배 증강 |
| 일반 클래스 | 5배 증강 |
