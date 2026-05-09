# -*- coding: utf-8 -*-
"""
[visualizer.py] Step 6 & 7 - 학습 결과 시각화 + 테스트 이미지 예측
"""

import os
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from src.config import OUTPUT_DIR, IMGSZ


def _best_model_path() -> str:
    return os.path.join(OUTPUT_DIR, 'train', 'weights', 'best.pt')


# ── Step 6: Loss / mAP 그래프 ────────────────────
def plot_results():
    print("\n" + "=" * 55)
    print("[Step 6] 학습 결과 시각화")
    print("=" * 55)

    csv_path = os.path.join(OUTPUT_DIR, 'train', 'results.csv')
    if not os.path.exists(csv_path):
        print(f"[경고] 결과 CSV 없음: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss',     color='blue')
    ax1.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss',     color='green')
    ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL(L1) Loss', color='orange')
    ax1.set_title('Train Loss Tracking (Augmented)')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['epoch'], df['metrics/mAP50(B)'],    label='mAP@50',       color='red',    linewidth=2)
    ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95',    color='blue',   linewidth=2)
    ax2.plot(df['epoch'], df['val/cls_loss'],         label='Val Cls Loss', color='purple', linestyle='--')
    ax2.set_title('Validation Metrics (Augmented)')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'train', 'result_plot.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"그래프 저장 완료: {save_path}")


# ── Step 7: 테스트 이미지 예측 + Confidence 분포 ─
def evaluate_test(test_image_files: list, sample_n: int = 20):
    print("\n" + "=" * 55)
    print("[Step 7] 테스트 이미지 예측 시작...")
    print("=" * 55)

    best_model   = YOLO(_best_model_path())
    sample_imgs  = random.sample(test_image_files, min(sample_n, len(test_image_files)))
    test_results = best_model.predict(source=sample_imgs, save=True, imgsz=IMGSZ, conf=0.25)

    all_confidences = []
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()

    for i, r in enumerate(test_results):
        if r.boxes:
            all_confidences.extend(r.boxes.conf.cpu().numpy().tolist())
        img = cv2.cvtColor(cv2.imread(os.path.join(r.save_dir, os.path.basename(r.path))), cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Test Image {i+1}", fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'train', 'test_gallery.png'), dpi=100)
    plt.show()

    if all_confidences:
        plt.figure(figsize=(10, 5))
        plt.hist(all_confidences, bins=20, color='skyblue', edgecolor='black')
        plt.title('Prediction Confidence Distribution', fontsize=15)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.5)
        plt.savefig(os.path.join(OUTPUT_DIR, 'train', 'confidence_dist.png'), dpi=150)
        plt.show()
        print(f"평균 확신도(Confidence): {sum(all_confidences)/len(all_confidences):.3f}")
