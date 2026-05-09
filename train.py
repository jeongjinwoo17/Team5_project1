# -*- coding: utf-8 -*-
"""
경구 알약 탐지 프로그램 - 메인 실행 파일

실행 방법:
    pip install kagglehub ultralytics albumentations opencv-python torchmetrics tqdm pandas matplotlib
    python train.py
"""

from src.data_loader  import load_data
from src.dataset      import prepare_dataset
from src.augmentation import run_augmentation
from src.trainer      import create_yaml, train_model
from src.visualizer   import plot_results, evaluate_test
from src.evaluator    import evaluate_train_set
from src.submission   import generate_submission


if __name__ == '__main__':
    # Step 1: 데이터 다운로드
    _, train_image_files, test_image_files, json_files = load_data()

    # Step 2: 클래스 맵핑 + Train/Val 분리
    class_mapping, class_names, category_dict = prepare_dataset(train_image_files, json_files)

    # Step 3: 데이터 증강
    run_augmentation()

    # Step 4: data.yaml 생성
    yaml_path = create_yaml(class_mapping, class_names)

    # Step 5: 모델 학습
    train_model(yaml_path)

    # Step 6: Loss / mAP 시각화
    plot_results()

    # Step 7: 테스트 예측 + Confidence 시각화
    evaluate_test(test_image_files)

    # Train 자체 검증
    evaluate_train_set()

    # Step 8: 제출용 CSV 생성
    generate_submission(test_image_files, category_dict)
