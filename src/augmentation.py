# -*- coding: utf-8 -*-
"""
[augmentation.py] Step 3 - 오프라인 데이터 증강
- 소수 클래스 (MINOR_THRESHOLD 이하): MINOR_AUG_COUNT 배 증강
- 일반 클래스: NORMAL_AUG_COUNT 배 증강
- 증강 기법: hflip / vrot / noise / lighting / color_jitter / geometric

[수정 포인트]
augment_one()이 증강 기법 이름 대신 횟수(aug_count)를 받아
aug_names를 순환하며 _aug{i}_{name} 패턴으로 파일명을 생성 → 덮어쓰기 방지
"""

import os
import shutil
import cv2
import albumentations as A
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from src.config import BASE_DIR, MINOR_THRESHOLD, MINOR_AUG_COUNT, NORMAL_AUG_COUNT


# ── 증강 파이프라인 ────────────────────────────────
def get_transforms() -> dict:
    bp = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
    return {
        'hflip':        A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bp),
        'vrot':         A.Compose([A.VerticalFlip(p=1.0), A.Rotate(limit=15, p=0.8)], bbox_params=bp),
        'noise':        A.Compose([A.GaussNoise(var_limit=(10, 50), p=1.0), A.HorizontalFlip(p=0.5)], bbox_params=bp),
        'lighting':     A.Compose([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                                   A.RandomGamma(gamma_limit=(80, 120), p=0.5)], bbox_params=bp),
        'color_jitter': A.Compose([A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20, p=1.0)], bbox_params=bp),
        'geometric':    A.Compose([A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=1.0),
                                   A.Perspective(scale=(0.02, 0.05), p=0.5)], bbox_params=bp),
    }


# ── YOLO 라벨 읽기 / 쓰기 ─────────────────────────
def read_yolo_label(label_path: Path):
    class_ids, bboxes = [], []
    if not label_path.exists():
        return class_ids, bboxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            class_ids.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:5]])
    return class_ids, bboxes


def write_yolo_label(label_path: Path, class_ids: list, bboxes: list):
    lines = [
        f"{cid} {max(0.0,min(1.0,b[0])):.6f} {max(0.0,min(1.0,b[1])):.6f} "
        f"{max(0.0,min(1.0,b[2])):.6f} {max(0.0,min(1.0,b[3])):.6f}"
        for cid, b in zip(class_ids, bboxes)
    ]
    label_path.write_text('\n'.join(lines))


# ── 이미지 1장 증강 ───────────────────────────────
def augment_one(img_path: Path, lbl_path: Path,
                out_img_dir: Path, out_lbl_dir: Path,
                transforms: dict, aug_count: int) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    class_ids, bboxes = read_yolo_label(lbl_path)
    aug_names = list(transforms.keys())
    count     = 0

    for i in range(aug_count):
        aug_name = aug_names[i % len(aug_names)]   # 순환 선택
        try:
            res     = transforms[aug_name](image=img_rgb, bboxes=bboxes, class_labels=class_ids)
            out_img = out_img_dir / f"{img_path.stem}_aug{i}_{aug_name}{img_path.suffix}"
            out_lbl = out_lbl_dir / f"{img_path.stem}_aug{i}_{aug_name}.txt"
            cv2.imwrite(str(out_img), cv2.cvtColor(res['image'], cv2.COLOR_RGB2BGR))
            write_yolo_label(out_lbl, res['class_labels'], res['bboxes'])
            count += 1
        except Exception:
            continue
    return count


def _count_classes(lbl_dir: str) -> Counter:
    counter = Counter()
    for lbl in Path(lbl_dir).glob('*.txt'):
        seen = {int(line.split()[0]) for line in lbl.read_text().strip().splitlines() if line.split()}
        for cid in seen:
            counter[cid] += 1
    return counter


# ── 전체 증강 실행 ────────────────────────────────
def run_augmentation():
    print("\n" + "=" * 55)
    print("[Step 3] 데이터 증강 시작...")
    print("=" * 55)

    train_img_dir = Path(os.path.join(BASE_DIR, 'images/train'))
    train_lbl_dir = Path(os.path.join(BASE_DIR, 'labels/train'))
    aug_img_dir   = Path(os.path.join(BASE_DIR, 'images/train_aug'))
    aug_lbl_dir   = Path(os.path.join(BASE_DIR, 'labels/train_aug'))
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_lbl_dir.mkdir(parents=True, exist_ok=True)

    transforms    = get_transforms()
    class_counts  = _count_classes(str(train_lbl_dir))
    minor_classes = {cid for cid, cnt in class_counts.items() if cnt <= MINOR_THRESHOLD}

    print(f"\n📉 소수 클래스 ({MINOR_THRESHOLD}장 이하): {len(minor_classes)}개 발견 ({MINOR_AUG_COUNT}배 집중 증강)")
    print(f"🟢 일반 클래스: {NORMAL_AUG_COUNT}배 증강 적용")

    img_paths = sorted(train_img_dir.glob('*.png')) + sorted(train_img_dir.glob('*.jpg'))
    total_aug = 0

    for img_path in tqdm(img_paths, desc="데이터 증강 중"):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"

        # 원본 복사
        shutil.copy(img_path, aug_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, aug_lbl_dir / lbl_path.name)

        class_ids, _ = read_yolo_label(lbl_path)
        is_minor     = bool(set(class_ids) & minor_classes)
        aug_count    = MINOR_AUG_COUNT if is_minor else NORMAL_AUG_COUNT
        total_aug   += augment_one(img_path, lbl_path, aug_img_dir, aug_lbl_dir, transforms, aug_count)

    print(f"\n원본 {len(img_paths)}장 → 총 {len(img_paths) + total_aug}장으로 증가했습니다.")
