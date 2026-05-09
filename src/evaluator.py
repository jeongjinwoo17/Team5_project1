# -*- coding: utf-8 -*-
"""
[evaluator.py] Train 자체 검증 - IoU 기반 TP / FP / FN 계산
best.pt 모델로 train 이미지 최대 100장을 랜덤 채점합니다.
"""

import os
import random
from ultralytics import YOLO
from src.config import BASE_DIR, OUTPUT_DIR, IMGSZ


def _best_model_path() -> str:
    return os.path.join(OUTPUT_DIR, 'train', 'weights', 'best.pt')


def calculate_iou(box1: list, box2: list) -> float:
    """box = [x_min, y_min, x_max, y_max] (정규화 좌표)"""
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / float(area1 + area2 - inter)


def evaluate_train_set(sample_n: int = 100, iou_thres: float = 0.5, conf_thres: float = 0.25):
    print("\n" + "=" * 55)
    print("[Train 검증] IoU 기반 TP / FP / FN 채점")
    print("=" * 55)

    model            = YOLO(_best_model_path())
    train_images_dir = os.path.join(BASE_DIR, 'images/train')
    train_labels_dir = os.path.join(BASE_DIR, 'labels/train')

    train_img_files = [f for f in os.listdir(train_images_dir) if f.endswith('.png')]
    sample_imgs     = random.sample(train_img_files, min(sample_n, len(train_img_files)))
    print(f"모델 채점을 시작합니다! (총 {len(sample_imgs)}장 검사 중...)\n")

    total_tp, total_fp, total_fn = 0, 0, 0

    for img_name in sample_imgs:
        img_path   = os.path.join(train_images_dir, img_name)
        label_path = os.path.join(train_labels_dir, img_name.replace('.png', '.txt'))

        # 정답 로드
        ground_truths = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts    = line.strip().split()
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    ground_truths.append({
                        'class':   class_id,
                        'box':     [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                        'matched': False
                    })

        # 모델 예측
        results       = model.predict(source=img_path, imgsz=IMGSZ, conf=conf_thres, verbose=False)
        predictions   = results[0].boxes
        preds_matched = [False] * len(predictions)

        for p_idx, pred in enumerate(predictions):
            pred_cls        = int(pred.cls.cpu().numpy()[0])
            px1, py1, px2, py2 = pred.xyxyn.cpu().numpy()[0]
            pred_box        = [px1, py1, px2, py2]
            best_iou, best_gt_idx = 0, -1

            for gt_idx, gt in enumerate(ground_truths):
                if not gt['matched']:
                    iou = calculate_iou(pred_box, gt['box'])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx

            if best_iou >= iou_thres and best_gt_idx != -1 and ground_truths[best_gt_idx]['class'] == pred_cls:
                ground_truths[best_gt_idx]['matched'] = True
                preds_matched[p_idx] = True
                total_tp += 1

        total_fp += preds_matched.count(False)
        total_fn += sum(1 for gt in ground_truths if not gt['matched'])

    # 결과 출력
    print("=" * 50)
    print(f"Train {len(sample_imgs)}장 확인 결과")
    print("=" * 50)
    print(f"완벽하게 맞춘 약 (True Positive) : {total_tp} 개")
    print(f"잘못 예측한 약   (False Positive): {total_fp} 개")
    print(f"아예 놓쳐버린 약 (False Negative): {total_fn} 개")
    print("=" * 50)
    total_objects = total_tp + total_fn
    if total_objects > 0:
        print(f"정답률(Recall): {total_tp / total_objects * 100:.1f}% (전체 약 중에서 맞춘 비율)")
