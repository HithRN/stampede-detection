import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# Import functions from your main training file
from stampede_detection_fixed import (
    IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH,
    calculate_flow_acceleration,
    calculate_flow_divergence,
    calculate_scene_changes,
    calculate_motion_entropy
)

# ---------- CONFIG ----------
MODEL_PATH = "enhanced_stampede_detection_final.h5"
UMN_PATH = "UMN-test"  # root folder containing 0/ and 1/
# ----------------------------


def natural_sort(files):
    import re
    return sorted(files, key=lambda x: [int(t) if t.isdigit() else t.lower()
                                        for t in re.split('(\d+)', x)])


def compute_flow_sequence_from_frames(frame_list):

    downsample_factor = 0.2
    flows = []
    originals = []

    prev_gray = None

    for frame in frame_list:

        resized_original = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        originals.append(resized_original)

        small = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT))

        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Convert flow → HSV → BGR (same as training)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Reconstruct 2-channel flow exactly like training loader
        reconstructed_flow = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 2), dtype=np.float32)
        reconstructed_flow[..., 0] = bgr[..., 2] / 255.0 * 2 - 1
        reconstructed_flow[..., 1] = bgr[..., 1] / 255.0 * 2 - 1

        flows.append(reconstructed_flow)
        prev_gray = gray

    return flows, originals


def build_sequences(flows, originals):

    flow_sequences = []
    orig_sequences = []

    step = SEQUENCE_LENGTH // 2  # 50% overlap

    for start in range(0, len(flows) - SEQUENCE_LENGTH + 1, step):

        flow_seq = flows[start:start + SEQUENCE_LENGTH]
        orig_seq = originals[start:start + SEQUENCE_LENGTH]

        if len(flow_seq) == SEQUENCE_LENGTH:
            flow_sequences.append(np.array(flow_seq))
            orig_sequences.append(np.array(orig_seq))

    return np.array(flow_sequences), np.array(orig_sequences)


def evaluate_umn():

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    y_true = []
    y_pred = []
    y_scores = []

    for label_folder in ['0', '1']:

        class_path = os.path.join(UMN_PATH, label_folder)
        if not os.path.isdir(class_path):
            continue

        true_label = 0 if label_folder == '0' else 1

        for scene in os.listdir(class_path):

            scene_path = os.path.join(class_path, scene)
            if not os.path.isdir(scene_path):
                continue

            print(f"Processing {scene_path}")

            frame_files = natural_sort(
                [f for f in os.listdir(scene_path)
                 if f.lower().endswith(('.jpg', '.png'))]
            )

            frames = []
            for f in frame_files:
                img = cv2.imread(os.path.join(scene_path, f))
                if img is not None:
                    frames.append(img)

            if len(frames) < SEQUENCE_LENGTH + 1:
                continue

            flows, originals = compute_flow_sequence_from_frames(frames)
            X_flow, X_orig = build_sequences(flows, originals)

            if len(X_flow) == 0:
                continue

            # Scalar features
            acc = calculate_flow_acceleration(X_flow)
            div = calculate_flow_divergence(X_flow)
            sc  = calculate_scene_changes(X_orig)
            ent = calculate_motion_entropy(X_flow)

            X_scalar = np.stack([acc, div, sc, ent], axis=2)

            preds = model.predict([X_flow, X_scalar], verbose=0)
            avg_pred = np.mean(preds, axis=0)

            # 4-class → 2-class mapping
            prob_abnormal = avg_pred[2] + avg_pred[3]  # dense + risky
            pred_label = 1 if prob_abnormal >= 0.5 else 0

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(prob_abnormal)

    # ---------- METRICS ----------
    print("\n===== UMN EVALUATION RESULTS =====")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['normal','abnormal']))


if __name__ == "__main__":
    evaluate_umn()