# -*- coding: utf-8 -*-
"""
[dataset.py] Step 2 - 클래스 맵핑 생성 + YOLO 라벨 변환 + Train/Val 분리 (80:20)
"""

import os
import json
import shutil
import random
from tqdm import tqdm
from src.config import BASE_DIR, TRAIN_VAL_RATIO, RANDOM_SEED


def prepare_dataset(train_image_files: list, json_files: list):
    print("\n" + "=" * 55)
    print("[Step 2] 클래스 맵핑 생성 및 Train/Val 분리 (80:20)")
    print("=" * 55)

    # ── 전체 JSON에서 고유 카테고리 수집 → 고정 맵핑 생성 ──
    print("전체 JSON 파일에서 클래스(Category) 정보를 수집하여 고정 맵핑을 생성합니다...")
    category_dict = {}
    for j_path in json_files:
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for cat in data.get('categories', []):
                category_dict[cat['id']] = cat['name']

    sorted_categories = sorted(category_dict.items())
    class_mapping, class_names = {}, []
    for yolo_id, (orig_cat_id, cat_name) in enumerate(sorted_categories):
        class_mapping[orig_cat_id] = yolo_id
        class_names.append(cat_name)

    print(f"클래스 맵핑 완료 (총 {len(class_names)}개 클래스)")
    for orig_id, yolo_id in class_mapping.items():
        print(f"  - 원본 ID {orig_id} ➡️ YOLO ID {yolo_id} ({category_dict[orig_id]})")

    # ── YOLO 데이터셋 폴더 생성 ──
    for split in ['train', 'val']:
        os.makedirs(os.path.join(BASE_DIR, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, f'labels/{split}'), exist_ok=True)

    # ── JSON 파싱 → YOLO 형식 어노테이션 변환 ──
    img_name_to_path     = {os.path.basename(p): p for p in train_image_files}
    image_to_annotations = {}

    for j_path in tqdm(json_files, desc="JSON 분석 및 변환 중"):
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_info     = data['images'][0]
        img_filename = img_info['file_name']
        base_name    = os.path.splitext(img_filename)[0]

        if img_filename not in img_name_to_path:
            continue

        if base_name not in image_to_annotations:
            image_to_annotations[base_name] = {
                'actual_img_path': img_name_to_path[img_filename],
                'width':           img_info['width'],
                'height':          img_info['height'],
                'annotations':     []
            }

        for ann in data.get('annotations', []):
            yolo_class_id        = class_mapping[ann['category_id']]
            x_min, y_min, bw, bh = ann['bbox']
            x_center = (x_min + bw / 2.0) / img_info['width']
            y_center = (y_min + bh / 2.0) / img_info['height']
            norm_w   = bw / img_info['width']
            norm_h   = bh / img_info['height']
            image_to_annotations[base_name]['annotations'].append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

    # ── 80:20 분리 및 파일 저장 ──
    items = list(image_to_annotations.items())
    random.seed(RANDOM_SEED)
    random.shuffle(items)
    split_idx              = int(len(items) * TRAIN_VAL_RATIO)
    train_items, val_items = items[:split_idx], items[split_idx:]

    def _save_split(data_items, split_name):
        for base_name, info in data_items:
            shutil.copy(
                info['actual_img_path'],
                os.path.join(BASE_DIR, f'images/{split_name}', os.path.basename(info['actual_img_path']))
            )
            with open(os.path.join(BASE_DIR, f'labels/{split_name}', base_name + '.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(info['annotations']))

    _save_split(train_items, 'train')
    _save_split(val_items,   'val')
    print(f"Train: {len(train_items)}개 | Val: {len(val_items)}개 저장 완료")

    return class_mapping, class_names, category_dict
