#!/usr/bin/env python3
"""
evaluate_umn_frames_dataset.py

Proper dataset-level evaluation of UMN stored as frame folders:

UMN-test/
   ├─ 0/scene-1/*.jpg
   ├─ 1/scene-2/*.jpg

Each scene folder is treated as one video.
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

import test_video as tv  # reuse your feature + model code

IMAGE_HEIGHT = tv.IMAGE_HEIGHT
IMAGE_WIDTH = tv.IMAGE_WIDTH
SEQUENCE_LENGTH = tv.SEQUENCE_LENGTH
CATEGORIES = tv.CATEGORIES

# 4-class → 2-class mapping
NORMAL_CLASSES = ['normal', 'moderate']
ABNORMAL_CLASSES = ['dense', 'risky']


def load_scene_frames(scene_path):
    """Load frames in sorted order."""
    frame_files = sorted([
        f for f in os.listdir(scene_path)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    frames = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(scene_path, fname))
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frames.append(img)

    return frames


def compute_optical_flow_from_frames(frames):
    """Generate Farneback optical flow from frame list."""
    flows = []
    prev_gray = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0
        )

        flows.append(flow)
        prev_gray = gray

    return flows


def build_sequences(flow_frames, original_frames):
    """Create sliding-window sequences."""
    flow_sequences = []
    orig_sequences = []

    for start in range(0, len(flow_frames) - SEQUENCE_LENGTH + 1):
        flow_seq = flow_frames[start:start + SEQUENCE_LENGTH]
        orig_seq = original_frames[start:start + SEQUENCE_LENGTH]
        flow_sequences.append(np.array(flow_seq))
        orig_sequences.append(np.array(orig_seq))

    return flow_sequences, orig_sequences


def evaluate_scene(model, scene_path):
    frames = load_scene_frames(scene_path)

    if len(frames) < SEQUENCE_LENGTH + 1:
        return None

    flow_frames = compute_optical_flow_from_frames(frames)
    flow_seqs, orig_seqs = build_sequences(flow_frames, frames)

    if not flow_seqs:
        return None

    # Scalar features
    acc = tv.calculate_flow_acceleration(flow_seqs)
    div = tv.calculate_flow_divergence(flow_seqs)
    sc  = tv.calculate_scene_changes(orig_seqs)
    ent = tv.calculate_motion_entropy(flow_seqs)

    scalar = np.stack([acc, div, sc, ent], axis=2)

    X_flow = np.array(flow_seqs, dtype=np.float32)
    X_scalar = np.array(scalar, dtype=np.float32)

    preds = model.predict([X_flow, X_scalar], verbose=0)
    avg_pred = np.mean(preds, axis=0)

    return avg_pred


def evaluate_dataset(model, dataset_path):
    y_true = []
    y_pred = []
    y_score = []

    name_to_idx = {n: i for i, n in enumerate(CATEGORIES)}
    normal_idxs = [name_to_idx[n] for n in NORMAL_CLASSES]
    abnormal_idxs = [name_to_idx[n] for n in ABNORMAL_CLASSES]

    for label_folder in ['0', '1']:
        class_path = os.path.join(dataset_path, label_folder)
        if not os.path.isdir(class_path):
            continue

        true_label = 0 if label_folder == '0' else 1

        for scene in os.listdir(class_path):
            scene_path = os.path.join(class_path, scene)
            if not os.path.isdir(scene_path):
                continue

            print(f"Processing {scene_path}")

            avg_pred = evaluate_scene(model, scene_path)
            if avg_pred is None:
                print("  Skipped (too few frames)")
                continue

            prob_abnormal = np.sum([avg_pred[i] for i in abnormal_idxs])
            pred_label = 1 if prob_abnormal >= 0.5 else 0

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_score.append(prob_abnormal)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred)

    print("\n===== UMN FRAME-BASED EVALUATION =====")
    print(f"Scenes evaluated: {len(y_true)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['normal','abnormal']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--dataset-path', required=True,
                        help='Path to UMN-test folder')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, compile=False)
    evaluate_dataset(model, args.dataset_path)


if __name__ == "__main__":
    main()