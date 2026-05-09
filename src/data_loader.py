# -*- coding: utf-8 -*-
"""
[data_loader.py] Step 1 - kagglehub 데이터 다운로드 및 파일 목록 생성
"""

import os
import glob
import kagglehub


def load_data():
    print("=" * 55)
    print("[Step 1] kagglehub에서 데이터 다운로드 중...")
    print("=" * 55)

    path = kagglehub.competition_download('ai10-level1-project')
    print("Path to competition files:", path)

    train_image_files = glob.glob(os.path.join(path, '**/train_images/**/*.png'), recursive=True)
    test_image_files  = glob.glob(os.path.join(path, '**/test_images/**/*.png'),  recursive=True)
    json_files        = glob.glob(os.path.join(path, '**/train_annotations/**/*.json'), recursive=True)

    # 서브폴더 구조가 다를 경우 루트에서 재탐색
    if not train_image_files:
        all_pngs          = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
        train_image_files = [f for f in all_pngs if 'train' in f]
        test_image_files  = [f for f in all_pngs if 'test'  in f]
        json_files        = glob.glob(os.path.join(path, '**/*.json'), recursive=True)

    print(f"Train 이미지: {len(train_image_files)}개 | "
          f"Test 이미지: {len(test_image_files)}개 | "
          f"JSON 정답지: {len(json_files)}개")

    return path, train_image_files, test_image_files, json_files
