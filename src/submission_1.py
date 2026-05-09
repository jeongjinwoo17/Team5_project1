# -*- coding: utf-8 -*-
"""
[submission.py] Step 8 - 제출용 CSV 파일 생성
- YOLO ID → 원본 category_id 역방향 변환
- 이미지 번호 오름차순 정렬 후 예측
- annotation_id: 1부터 순번 부여
"""

import os
import re
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from src.config import OUTPUT_DIR, SUBMISSION_PATH, IMGSZ


def _best_model_path() -> str:
    return os.path.join(OUTPUT_DIR, 'train', 'weights', 'best.pt')


def get_image_id(filepath: str) -> int:
    """파일명에서 마지막 숫자를 image_id로 추출"""
    numbers = re.findall(r'\d+', os.path.basename(filepath))
    return int(numbers[-1]) if numbers else 0


def generate_submission(test_image_files: list, category_dict: dict):
    print("\n" + "=" * 55)
    print("[Step 8] 제출용 CSV 파일 생성")
    print("=" * 55)

    # YOLO ID → 원본 category_id 역방향 맵핑
    sorted_categories = sorted(category_dict.items())
    yolo_to_original  = {yolo_id: orig_cat_id for yolo_id, (orig_cat_id, _) in enumerate(sorted_categories)}
    print(f"역방향 장부 완성! (예: YOLO 0번 → 원본 {yolo_to_original[0]} 번)")

    model          = YOLO(_best_model_path())
    test_img_files = sorted(test_image_files, key=get_image_id)
    print(f"📁 총 {len(test_img_files)}장을 번호 순서대로 정렬하여 예측합니다!")

    submission_data = []
    annotation_id   = 1

    for img_path in tqdm(test_img_files, desc="테스트 이미지 예측 중"):
        image_id = get_image_id(img_path)
        results  = model.predict(source=img_path, imgsz=IMGSZ, conf=0.1, verbose=False)

        for box in results[0].boxes:
            yolo_pred_id    = int(box.cls.cpu().numpy()[0])
            original_cat_id = yolo_to_original[yolo_pred_id]
            score           = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]

            submission_data.append({
                'annotation_id': annotation_id,
                'image_id':      image_id,
                'category_id':   original_cat_id,
                'bbox_x':        round(x1, 2),
                'bbox_y':        round(y1, 2),
                'bbox_w':        round(x2 - x1, 2),
                'bbox_h':        round(y2 - y1, 2),
                'score':         round(score, 3)
            })
            annotation_id += 1

    df = pd.DataFrame(submission_data)
    df.to_csv(SUBMISSION_PATH, index=False)

    print("\n" + "=" * 55)
    print(f"제출용 파일 생성 완료: {SUBMISSION_PATH}")
    print(f"총 {len(test_img_files)}장에서 {len(df)}개의 알약 탐지")
    print("=" * 55)
    print(df.head(10).to_string(index=False))
