# -*- coding: utf-8 -*-
"""
[config.py] 경로 및 하이퍼파라미터 설정
여기만 수정하면 전체 파이프라인에 반영됩니다.
"""

# ── 경로 ──────────────────────────────────────────
BASE_DIR        = './YOLO_Dataset_Split'   # YOLO 데이터셋 저장 폴더
OUTPUT_DIR      = './runs'                  # 학습 결과 저장 폴더
SUBMISSION_PATH = './submission.csv'        # 제출용 CSV 경로
MODEL_WEIGHTS   = 'yolo12s.pt'             # 사전학습 가중치

# ── 학습 하이퍼파라미터 ───────────────────────────
EPOCHS          = 200
PATIENCE        = 20
IMGSZ           = 1280
BATCH           = 4

# ── 데이터 분리 ───────────────────────────────────
TRAIN_VAL_RATIO = 0.8    # Train 비율 (나머지는 Val)
RANDOM_SEED     = 42

# ── 데이터 증강 ───────────────────────────────────
MINOR_THRESHOLD  = 6     # 이 장수 이하면 소수 클래스
MINOR_AUG_COUNT  = 8     # 소수 클래스 증강 횟수
NORMAL_AUG_COUNT = 5     # 일반 클래스 증강 횟수
