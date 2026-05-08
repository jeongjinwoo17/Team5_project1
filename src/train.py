# -*- coding: utf-8 -*-
"""
알약 객체 탐지 - YOLOv12s 학습 코드
kagglehub 자동 다운로드 + 로컬 PC 실행 버전

[사전 준비]
1. pip install -r requirements.txt
2. Kaggle API 키 설정:
   - https://www.kaggle.com → Settings → API → Create New Token
   - 다운로드된 kaggle.json을 아래 경로에 저장:
     Windows: C:\\Users\\본인이름\\.kaggle\\kaggle.json
     Mac/Linux: ~/.kaggle/kaggle.json
3. Kaggle 대회 페이지에서 "Join Competition" 후 규칙 동의 필수

[실행 방법]
    python src/train.py
"""

import os
import re
import json
import shutil
import random
import glob
import platform
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import albumentations as A
import kagglehub
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from ultralytics import YOLO

# ==================================================
# [설정] 경로 및 하이퍼파라미터
# ==================================================
KAGGLE_COMPETITION = 'ai10-level1-project'   # Kaggle 대회 슬러그
BASE_OUTPUT_DIR    = './YOLO_Dataset_Split'  # YOLO 데이터셋 저장 경로
SUBMISSION_PATH    = './submission.csv'      # 제출 CSV 저장 경로

# 학습 하이퍼파라미터
EPOCHS    = 200
PATIENCE  = 20
IMGSZ     = 1280
BATCH     = 4
MINOR_THR = 15   # 소수 클래스 기준 (장 수)
# ==================================================


# --------------------------------------------------
# [Step 0] 폰트 설정 (OS 자동 감지)
# --------------------------------------------------
def setup_font():
    system = platform.system()
    if system == 'Windows':
        candidates = ['malgun.ttf', 'gulim.ttc']
        dirs = [r'C:\Windows\Fonts']
    elif system == 'Darwin':
        candidates = ['AppleGothic.ttf', 'NanumGothic.ttf']
        dirs = ['/Library/Fonts', '/System/Library/Fonts']
    else:
        candidates = ['NanumBarunGothic.ttf', 'NanumGothic.ttf']
        dirs = ['/usr/share/fonts/truetype/nanum', '/usr/share/fonts']

    for d in dirs:
        for name in candidates:
            fp = os.path.join(d, name)
            if os.path.exists(fp):
                plt.rc('font', family=fm.FontProperties(fname=fp).get_name())
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 폰트 설정 완료: {fp}")
                return
    print("⚠️ 한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
    plt.rcParams['axes.unicode_minus'] = False


# --------------------------------------------------
# [Step 1] kagglehub로 데이터 자동 다운로드
# --------------------------------------------------
def download_data():
    print("📥 Kaggle 데이터 다운로드 중... (최초 1회만 다운로드됩니다)")
    data_path = kagglehub.competition_download(KAGGLE_COMPETITION)
    print(f"✅ 다운로드 완료! 데이터 경로: {data_path}")
    return data_path


def load_data_paths(data_path):
    train_image_files = glob.glob(os.path.join(data_path, '**/train_images/**/*.png'), recursive=True)
    test_image_files  = glob.glob(os.path.join(data_path, '**/test_images/**/*.png'),  recursive=True)
    json_files        = glob.glob(os.path.join(data_path, '**/train_annotations/**/*.json'), recursive=True)

    # png 없으면 jpg 탐색
    if not train_image_files:
        train_image_files = glob.glob(os.path.join(data_path, '**/train_images/**/*.jpg'), recursive=True)
    if not test_image_files:
        test_image_files  = glob.glob(os.path.join(data_path, '**/test_images/**/*.jpg'), recursive=True)

    print(f"Train 이미지: {len(train_image_files)}개 | "
          f"Test 이미지: {len(test_image_files)}개 | "
          f"JSON 정답지: {len(json_files)}개")

    if not train_image_files or not json_files:
        print("\n⚠️  파일을 찾지 못했습니다. 실제 폴더 구조를 확인하세요:")
        for root, dirs, files in os.walk(data_path):
            level = root.replace(data_path, '').count(os.sep)
            if level < 3:
                print(f"{'  ' * level}{os.path.basename(root)}/")
                for f in files[:5]:
                    print(f"{'  ' * (level+1)}{f}")

    return train_image_files, test_image_files, json_files


# --------------------------------------------------
# [Step 2] 클래스 맵핑 + Train/Val 분리 (80:20)
# --------------------------------------------------
def prepare_dataset(train_image_files, json_files):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, f'labels/{split}'), exist_ok=True)

    img_name_to_path = {os.path.basename(p): p for p in train_image_files}

    print("🔍 전체 JSON에서 클래스(Category) 정보 수집 중...")
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

    print(f"✅ 클래스 맵핑 완료! (총 {len(class_names)}개 클래스)")
    for orig_id, yolo_id in class_mapping.items():
        print(f"  - 원본 ID {orig_id} → YOLO ID {yolo_id} ({category_dict[orig_id]})")

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
                'width':  img_info['width'],
                'height': img_info['height'],
                'annotations': []
            }

        for ann in data.get('annotations', []):
            yolo_class_id = class_mapping[ann['category_id']]
            x_min, y_min, box_w, box_h = ann['bbox']
            x_center = (x_min + box_w / 2.0) / img_info['width']
            y_center = (y_min + box_h / 2.0) / img_info['height']
            norm_w   = box_w / img_info['width']
            norm_h   = box_h / img_info['height']
            image_to_annotations[base_name]['annotations'].append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

    items = list(image_to_annotations.items())
    random.seed(42)
    random.shuffle(items)
    split_idx = int(len(items) * 0.8)
    train_items, val_items = items[:split_idx], items[split_idx:]

    def save_split(data_items, split_name):
        for base_name, info in data_items:
            shutil.copy(
                info['actual_img_path'],
                os.path.join(BASE_OUTPUT_DIR, f'images/{split_name}', os.path.basename(info['actual_img_path']))
            )
            with open(os.path.join(BASE_OUTPUT_DIR, f'labels/{split_name}', base_name + '.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(info['annotations']))

    save_split(train_items, 'train')
    save_split(val_items,   'val')
    print(f"✅ Train: {len(train_items)}장 / Val: {len(val_items)}장 분리 완료")

    return class_mapping, class_names, category_dict


# --------------------------------------------------
# [Step 3] 데이터 증강
# --------------------------------------------------
def get_transforms():
    bp = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
    return {
        'hflip':        A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bp),
        'vrot':         A.Compose([A.VerticalFlip(p=1.0), A.Rotate(limit=15, p=0.8)], bbox_params=bp),
        'noise':        A.Compose([A.GaussNoise(var_limit=(10, 50), p=1.0), A.HorizontalFlip(p=0.5)], bbox_params=bp),
        'lighting':     A.Compose([A.RandomBrightnessContrast(0.3, 0.3, p=1.0), A.RandomGamma((80, 120), p=0.5)], bbox_params=bp),
        'color_jitter': A.Compose([A.HueSaturationValue(10, 20, 20, p=1.0)], bbox_params=bp),
        'blur_sharpen': A.Compose([A.OneOf([A.GaussianBlur((3,7), p=0.5), A.MotionBlur(7, p=0.5)], p=1.0)], bbox_params=bp),
        'geometric':    A.Compose([A.ShiftScaleRotate(0.05, 0.1, 10, p=1.0), A.Perspective((0.02, 0.05), p=0.5)], bbox_params=bp),
    }

def read_yolo_label(label_path):
    class_ids, bboxes = [], []
    if not label_path.exists(): return class_ids, bboxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            class_ids.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:5]])
    return class_ids, bboxes

def write_yolo_label(label_path, class_ids, bboxes):
    lines = [
        f"{cid} {max(0.0,min(1.0,b[0])):.6f} {max(0.0,min(1.0,b[1])):.6f} "
        f"{max(0.0,min(1.0,b[2])):.6f} {max(0.0,min(1.0,b[3])):.6f}"
        for cid, b in zip(class_ids, bboxes)
    ]
    label_path.write_text('\n'.join(lines))

def augment_one(img_path, lbl_path, out_img_dir, out_lbl_dir, transforms, suffix_list):
    img = cv2.imread(str(img_path))
    if img is None: return 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    class_ids, bboxes = read_yolo_label(lbl_path)
    count = 0
    for aug_name in suffix_list:
        try:
            res     = transforms[aug_name](image=img_rgb, bboxes=bboxes, class_labels=class_ids)
            out_img = out_img_dir / f"{img_path.stem}_{aug_name}{img_path.suffix}"
            out_lbl = out_lbl_dir / f"{img_path.stem}_{aug_name}.txt"
            cv2.imwrite(str(out_img), cv2.cvtColor(res['image'], cv2.COLOR_RGB2BGR))
            write_yolo_label(out_lbl, res['class_labels'], res['bboxes'])
            count += 1
        except:
            continue
    return count

def count_classes(lbl_dir):
    counter = Counter()
    for lbl in Path(lbl_dir).glob('*.txt'):
        seen = {int(line.split()[0]) for line in lbl.read_text().strip().splitlines() if line.split()}
        for cid in seen: counter[cid] += 1
    return counter

def run_augmentation():
    print("\n [데이터 증강] Train 데이터 증강 시작...")
    train_img_dir = Path(os.path.join(BASE_OUTPUT_DIR, 'images/train'))
    train_lbl_dir = Path(os.path.join(BASE_OUTPUT_DIR, 'labels/train'))
    aug_img_dir   = Path(os.path.join(BASE_OUTPUT_DIR, 'images/train_aug'))
    aug_lbl_dir   = Path(os.path.join(BASE_OUTPUT_DIR, 'labels/train_aug'))
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_lbl_dir.mkdir(parents=True, exist_ok=True)

    transforms    = get_transforms()
    aug_names     = list(transforms.keys())
    class_counts  = count_classes(train_lbl_dir)
    minor_classes = {cid for cid, cnt in class_counts.items() if cnt < MINOR_THR}
    print(f" 소수 클래스 ({MINOR_THR}장 미만): {len(minor_classes)}개 발견")

    img_paths = sorted(train_img_dir.glob('*.png')) + sorted(train_img_dir.glob('*.jpg'))
    total_aug = 0

    for img_path in tqdm(img_paths, desc="데이터 증강 중"):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        shutil.copy(img_path, aug_img_dir / img_path.name)
        if lbl_path.exists(): shutil.copy(lbl_path, aug_lbl_dir / lbl_path.name)

        class_ids, _ = read_yolo_label(lbl_path)
        is_minor = bool(set(class_ids) & minor_classes)

        total_aug += augment_one(img_path, lbl_path, aug_img_dir, aug_lbl_dir, transforms, aug_names)
        if is_minor:
            for repeat in range(7):
                total_aug += augment_one(img_path, lbl_path, aug_img_dir, aug_lbl_dir, transforms,
                                         [aug_names[repeat % len(aug_names)]])

    print(f" 증강 완료! 원본 {len(img_paths)}장 → 총 {len(img_paths) + total_aug}장")


# --------------------------------------------------
# [Step 4] data.yaml 생성
# --------------------------------------------------
def create_yaml(class_mapping, class_names):
    yaml_path      = os.path.join(BASE_OUTPUT_DIR, 'data.yaml')
    aug_train_path = os.path.abspath(os.path.join(BASE_OUTPUT_DIR, 'images/train_aug'))
    val_path       = os.path.abspath(os.path.join(BASE_OUTPUT_DIR, 'images/val'))
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"train: {aug_train_path}\n")
        f.write(f"val: {val_path}\n\n")
        f.write(f"nc: {len(class_mapping)}\n")
        f.write(f"names: {class_names}\n")
    print(f"✅ data.yaml 생성 완료: {yaml_path}")
    return yaml_path


# --------------------------------------------------
# [Step 5] 모델 학습
# --------------------------------------------------
def train_model(yaml_path):
    print("\n YOLO12s 학습 시작...")
    model = YOLO('yolo12s.pt')
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        mosaic=1.0,
        mixup=0.15,
        degrees=10.0,
        hsv_s=0.2,
        plots=True
    )


# --------------------------------------------------
# [Step 6] 학습 결과 시각화
# --------------------------------------------------
def visualize_results():
    csv_path = 'runs/detect/train/results.csv'
    if not os.path.exists(csv_path):
        print("⚠️ results.csv 없음. 학습 완료 후 다시 실행하세요.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss',     color='blue')
    ax1.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss',     color='green')
    ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL(L1) Loss', color='orange')
    ax1.set_title('Train Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(df['epoch'], df['metrics/mAP50(B)'],    label='mAP@50',       color='red',    linewidth=2)
    ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95',    color='blue',   linewidth=2)
    ax2.plot(df['epoch'], df['val/cls_loss'],         label='Val Cls Loss', color='purple', linestyle='--')
    ax2.set_title('Validation Metrics'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('runs/detect/train/loss_map_plot.png', dpi=150)
    plt.show()


# --------------------------------------------------
# [Step 7] Train 샘플 평가 (IoU 기반)
# --------------------------------------------------
def calculate_iou(box1, box2):
    x_left  = max(box1[0], box2[0]); y_top    = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2]); y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    return inter / float((box1[2]-box1[0])*(box1[3]-box1[1]) +
                         (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)

def evaluate_model():
    best_model       = YOLO('runs/detect/train/weights/best.pt')
    train_images_dir = os.path.join(BASE_OUTPUT_DIR, 'images/train')
    train_labels_dir = os.path.join(BASE_OUTPUT_DIR, 'labels/train')
    train_img_files  = [f for f in os.listdir(train_images_dir) if f.endswith(('.png', '.jpg'))]
    sample_imgs      = random.sample(train_img_files, min(100, len(train_img_files)))

    print(f"\n 모델 채점 시작 (총 {len(sample_imgs)}장)")
    total_tp = total_fp = total_fn = 0

    for img_name in sample_imgs:
        img_path   = os.path.join(train_images_dir, img_name)
        label_path = os.path.join(train_labels_dir, os.path.splitext(img_name)[0] + '.txt')

        ground_truths = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts  = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    ground_truths.append({
                        'class': cls_id,
                        'box': [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                        'matched': False
                    })

        predictions   = best_model.predict(source=img_path, imgsz=IMGSZ, conf=0.25, verbose=False)[0].boxes
        preds_matched = [False] * len(predictions)

        for p_idx, pred in enumerate(predictions):
            pred_cls            = int(pred.cls.cpu().numpy()[0])
            px1, py1, px2, py2 = pred.xyxyn.cpu().numpy()[0]
            pred_box            = [px1, py1, px2, py2]
            best_iou = 0; best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if not gt['matched']:
                    iou = calculate_iou(pred_box, gt['box'])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx

            if best_iou >= 0.5 and best_gt_idx != -1 and ground_truths[best_gt_idx]['class'] == pred_cls:
                ground_truths[best_gt_idx]['matched'] = True
                preds_matched[p_idx] = True
                total_tp += 1

        total_fp += preds_matched.count(False)
        total_fn += sum(1 for gt in ground_truths if not gt['matched'])

    print("=" * 50)
    print(f"True Positive  : {total_tp}")
    print(f"False Positive : {total_fp}")
    print(f"False Negative : {total_fn}")
    if (total_tp + total_fn) > 0:
        print(f"Recall         : {total_tp / (total_tp + total_fn) * 100:.1f}%")
    print("=" * 50)


# --------------------------------------------------
# [Step 8] 제출용 CSV 생성
# --------------------------------------------------
def generate_submission(test_image_files, json_files):
    print("\n 제출용 CSV 파일 생성 중...")

    category_dict = {}
    for j_path in json_files:
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for cat in data.get('categories', []):
                category_dict[cat['id']] = cat['name']

    sorted_categories = sorted(category_dict.items())
    yolo_to_original  = {yolo_id: orig_id for yolo_id, (orig_id, _) in enumerate(sorted_categories)}
    print(f"역방향 맵핑 완성! (예: YOLO 0번 → 원본 {yolo_to_original[0]}번)")

    model = YOLO('runs/detect/train/weights/best.pt')

    def get_image_id(filepath):
        numbers = re.findall(r'\d+', os.path.basename(filepath))
        return int(numbers[-1]) if numbers else 0

    test_img_files  = sorted(test_image_files, key=get_image_id)
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

    df_submission = pd.DataFrame(submission_data)
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n✅ 제출 파일 생성 완료: {SUBMISSION_PATH}")
    print(f"총 {len(test_img_files)}장에서 {len(df_submission)}개 알약 탐지")
    print(df_submission.head(10).to_string())


# ==================================================
# Main
# ==================================================
if __name__ == '__main__':
    setup_font()

    # Step 1: kagglehub로 데이터 다운로드
    data_path = download_data()

    # Step 1-1: 파일 경로 수집
    train_image_files, test_image_files, json_files = load_data_paths(data_path)

    # Step 2: 클래스 맵핑 + Train/Val 분리
    class_mapping, class_names, category_dict = prepare_dataset(train_image_files, json_files)

    # Step 3: 데이터 증강
    run_augmentation()

    # Step 4: data.yaml 생성
    yaml_path = create_yaml(class_mapping, class_names)

    # Step 5: 모델 학습
    train_model(yaml_path)

    # Step 6: 결과 시각화
    visualize_results()

    # Step 7: 평가
    evaluate_model()

    # Step 8: 제출 CSV 생성
    generate_submission(test_image_files, json_files)
