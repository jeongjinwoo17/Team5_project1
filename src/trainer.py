# -*- coding: utf-8 -*-
"""
[trainer.py] Step 4 & 5 - data.yaml 생성 + YOLO12s 모델 학습
"""

import os
from pathlib import Path
from ultralytics import YOLO
from src.config import BASE_DIR, OUTPUT_DIR, MODEL_WEIGHTS, EPOCHS, PATIENCE, IMGSZ, BATCH


def create_yaml(class_mapping: dict, class_names: list) -> str:
    print("\n" + "=" * 55)
    print("[Step 4] data.yaml 생성")
    print("=" * 55)

    # 로컬 환경에서 YOLO가 경로를 못 찾는 오류를 막기 위해 절대경로로 저장
    yaml_path = os.path.join(BASE_DIR, 'data.yaml')
    aug_path  = str(Path(os.path.join(BASE_DIR, 'images/train_aug')).resolve())
    val_path  = str(Path(os.path.join(BASE_DIR, 'images/val')).resolve())

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"train: {aug_path}\n")
        f.write(f"val:   {val_path}\n\n")
        f.write(f"nc: {len(class_mapping)}\n")
        f.write(f"names: {class_names}\n")

    print(f"data.yaml 생성 완료: {yaml_path}")
    return yaml_path


def train_model(yaml_path: str):
    print("\n" + "=" * 55)
    print("[Step 5] YOLO12s 학습 시작...")
    print("=" * 55)

    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        mosaic=1.0,
        mixup=0.15,
        degrees=40.0,
        project=OUTPUT_DIR,
        name='train',
        plots=True
    )
