"""
test_video.py
-------------
Standalone test-only script for Stampede Detection.

Usage:
    python test_video.py --model-path path/to/model.h5 --video-path path/to/video.mp4

All logic is taken directly from stampede_detection_fixed.py.
No training, no dataset loading — just inference on a single video.
"""

import os
import cv2
import numpy as np
import argparse
import time
import datetime

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Flatten,
    Dense, Dropout, LSTM, TimeDistributed, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

# ============================================================
# CONSTANTS  (must match what was used during training)
# ============================================================
IMAGE_HEIGHT   = 224
IMAGE_WIDTH    = 224
SEQUENCE_LENGTH = 16
NUM_CLASSES    = 4
CATEGORIES     = ["normal", "moderate", "dense", "risky"]


# ============================================================
# MODEL ARCHITECTURE
# (exact copy from stampede_detection_fixed.py)
# ============================================================
def create_enhanced_cnn_lstm_model():
    flow_input_shape   = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 2)
    scalar_input_shape = (SEQUENCE_LENGTH, 4)

    flow_input = Input(shape=flow_input_shape, name='optical_flow_input')

    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(flow_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Flatten())(x)

    scalar_input = Input(shape=scalar_input_shape, name='scalar_features_input')

    combined = Concatenate(axis=2)([x, scalar_input])

    lstm1    = LSTM(256, return_sequences=True)(combined)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2    = LSTM(128)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)

    dense1 = Dense(64, activation='relu')(dropout2)
    bn     = BatchNormalization()(dense1)
    output = Dense(NUM_CLASSES, activation='softmax')(bn)

    model = Model(inputs=[flow_input, scalar_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# SCALAR FEATURE HELPERS
# (exact copies from stampede_detection_fixed.py)
# ============================================================
def calculate_flow_acceleration(flow_sequences):
    accelerations = []
    for sequence in flow_sequences:
        seq_acceleration = []
        for i in range(1, len(sequence)):
            accel           = sequence[i] - sequence[i - 1]
            accel_magnitude = np.sqrt(np.sum(accel ** 2, axis=2))
            mean_accel      = np.mean(accel_magnitude)
            seq_acceleration.append(mean_accel)
        seq_acceleration.insert(0, 0)
        accelerations.append(np.array(seq_acceleration))
    return np.array(accelerations)


def calculate_flow_divergence(flow_sequences):
    divergences = []
    for sequence in flow_sequences:
        seq_divergence = []
        for flow in sequence:
            dx             = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
            dy             = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)
            divergence     = dx + dy
            mean_divergence = np.mean(np.abs(divergence))
            seq_divergence.append(mean_divergence)
        divergences.append(np.array(seq_divergence))
    return np.array(divergences)


def calculate_scene_changes(original_frames):
    ssim_scores = []
    for sequence in original_frames:
        seq_ssim = []
        for i in range(1, len(sequence)):
            if len(sequence[i].shape) == 3:
                frame1 = cv2.cvtColor(sequence[i - 1], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(sequence[i],     cv2.COLOR_BGR2GRAY)
            else:
                frame1 = sequence[i - 1]
                frame2 = sequence[i]
            score       = ssim(frame1, frame2, data_range=frame2.max() - frame2.min())
            change_score = 1.0 - score
            seq_ssim.append(change_score)
        seq_ssim.insert(0, 0)
        ssim_scores.append(np.array(seq_ssim))
    return np.array(ssim_scores)


def calculate_motion_entropy(flow_sequences):
    entropies = []
    for sequence in flow_sequences:
        seq_entropy = []
        for flow in sequence:
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hist, _ = np.histogram(
                mag, bins=32,
                range=(0, np.max(mag) if np.max(mag) > 0 else 1)
            )
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            else:
                hist = np.ones_like(hist) / len(hist)
            flow_entropy = entropy(hist, base=2)
            if not np.isfinite(flow_entropy):
                flow_entropy = 0.0
            seq_entropy.append(flow_entropy)
        entropies.append(np.array(seq_entropy))
    return np.array(entropies)


# ============================================================
# OPTICAL FLOW EXTRACTION FROM VIDEO
# (exact copy from stampede_detection_fixed.py)
# ============================================================
def generate_optical_flow_and_features_from_video(
        video_path, output_dir,
        resize_dim=(IMAGE_WIDTH, IMAGE_HEIGHT),
        max_frames=200, frame_skip=5):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps            = cap.get(cv2.CAP_PROP_FPS)
    target_duration = 4
    target_frames  = min(max_frames, int(target_duration * fps))

    if total_frames <= target_frames:
        frame_indices = list(range(0, total_frames, frame_skip))
    else:
        frame_indices = [int(i * total_frames / target_frames) for i in range(target_frames)]

    print(f"Video: {total_frames} frames at {fps:.1f} FPS. Using {len(frame_indices)} frames for analysis.")

    downsample_factor = 0.2
    flow_frames      = []
    original_frames  = []
    processed_count  = 0
    prev_gray        = None

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at index {frame_idx}")
            continue

        resized_frame = cv2.resize(frame, resize_dim)
        original_frames.append(resized_frame)

        frame_small = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
        gray        = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray        = cv2.resize(gray, resize_dim)

        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Save a few flow visualisations for debugging
        if processed_count % 10 == 0:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv      = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(output_dir, f"frame_{processed_count:04d}.jpg"), rgb)

        flow_frames.append(flow)
        prev_gray = gray
        processed_count += 1

        if processed_count % 5 == 0:
            print(f"  Processed {processed_count}/{len(frame_indices)} frames")

    cap.release()
    print(f"Generated {len(flow_frames)} optical flow frames")

    if not original_frames and flow_frames:
        original_frames = [
            np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        ] * len(flow_frames)

    return flow_frames, original_frames


# ============================================================
# MAIN PREDICTION FUNCTION
# (exact copy of predict_with_enhanced_model from
#  stampede_detection_fixed.py — nothing changed)
# ============================================================
def predict_with_enhanced_model(model, video_path,
                                temp_dir="temp_optical_flow",
                                timeout_seconds=300):
    """
    Predict the risk level for a video using the enhanced model
    with performance monitoring.

    Returns
    -------
    category    : str   — predicted class label
    confidence  : float — softmax confidence for that class
    perf_metrics: dict  — timing / throughput statistics
    """

    perf_metrics = {
        'device'                 : 'GPU' if len(tf.config.list_physical_devices('GPU')) > 0 else 'CPU',
        'batch_size'             : 8,
        'total_frames_processed' : 0,
        'total_sequences'        : 0,
        'frame_processing_times' : [],
        'sequence_inference_times': [],
        'total_inference_time'   : 0.0,
        'avg_time_per_frame'     : 0.0,
        'avg_time_per_sequence'  : 0.0,
        'fps'                    : 0.0
    }

    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE MONITORING")
    print(f"{'='*70}")
    print(f"Device    : {perf_metrics['device']}")
    print(f"Batch Size: {perf_metrics['batch_size']}")
    print(f"{'='*70}\n")

    # ---- warm-up / sanity check ----
    def test_model_inference(model):
        print("Testing model inference speed...")
        dummy_flow   = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 2), dtype=np.float32)
        dummy_scalar = np.zeros((1, SEQUENCE_LENGTH, 4), dtype=np.float32)
        t0 = time.time()
        dummy_pred = model.predict([dummy_flow, dummy_scalar], verbose=0)
        t1 = time.time()
        print(f"  Warm-up inference: {t1 - t0:.3f}s  |  "
              f"pred shape={dummy_pred.shape}  "
              f"min={dummy_pred.min():.3f}  max={dummy_pred.max():.3f}")
        return t1 - t0

    test_model_inference(model)

    start_time = time.time()

    def check_timeout():
        if time.time() - start_time > timeout_seconds:
            print(f"Processing timed out after {timeout_seconds} seconds!")
            return True
        return False

    # ---- extract optical flow ----
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Generating optical flow for: {video_path}")

    t0 = time.time()
    flow_frames, original_frames = generate_optical_flow_and_features_from_video(
        video_path, temp_dir, max_frames=200, frame_skip=5
    )
    t1 = time.time()
    perf_metrics['total_frames_processed'] = len(flow_frames)
    perf_metrics['frame_processing_times'].append(t1 - t0)

    if check_timeout():
        return "Timeout (processing took too long)", 0.0, perf_metrics

    if len(flow_frames) < SEQUENCE_LENGTH:
        print(f"Warning: only {len(flow_frames)} frames — need at least {SEQUENCE_LENGTH}.")
        return "Unknown (not enough frames)", 0.0, perf_metrics

    # ---- build sequences ----
    print(f"Creating sequences from {len(flow_frames)} frames...")

    flow_sequences = []
    orig_sequences = []
    max_sequences  = 20

    if len(flow_frames) > SEQUENCE_LENGTH:
        step_size    = max(1, (len(flow_frames) - SEQUENCE_LENGTH) // max_sequences)
        start_indices = list(range(0, len(flow_frames) - SEQUENCE_LENGTH, step_size))[:max_sequences]

        for i in start_indices:
            flow_seq = flow_frames[i: i + SEQUENCE_LENGTH]
            orig_seq = original_frames[i: i + SEQUENCE_LENGTH] if i < len(original_frames) else []

            flow_sequences.append(np.array(flow_seq))
            orig_sequences.append(np.array(orig_seq) if orig_seq else None)

            if check_timeout():
                return "Timeout (processing took too long)", 0.0, perf_metrics

    if not flow_sequences:
        return "Unknown (could not create sequences)", 0.0, perf_metrics

    perf_metrics['total_sequences'] = len(flow_sequences)

    # ---- compute scalar features ----
    print("Calculating additional features for prediction...")

    flow_acceleration = calculate_flow_acceleration(flow_sequences)
    flow_divergence   = calculate_flow_divergence(flow_sequences)
    scene_changes     = (
        calculate_scene_changes(orig_sequences)
        if all(seq is not None for seq in orig_sequences)
        else np.zeros((len(flow_sequences), SEQUENCE_LENGTH))
    )
    motion_entropy = calculate_motion_entropy(flow_sequences)

    scalar_features = np.stack(
        [flow_acceleration, flow_divergence, scene_changes, motion_entropy],
        axis=2
    )   # shape: (num_sequences, SEQUENCE_LENGTH, 4)

    X_flow_pred   = np.array(flow_sequences)
    X_scalar_pred = scalar_features

    # ---- batched inference ----
    batch_size       = perf_metrics['batch_size']
    all_predictions  = []

    print(f"Making predictions in batches of {batch_size}...")
    total_infer_start = time.time()

    for i in range(0, len(X_flow_pred), batch_size):
        batch_flow   = X_flow_pred[i: i + batch_size]
        batch_scalar = X_scalar_pred[i: i + batch_size]

        t0           = time.time()
        batch_preds  = model.predict([batch_flow, batch_scalar], verbose=0)
        t1           = time.time()

        perf_metrics['sequence_inference_times'].append(t1 - t0)
        all_predictions.append(batch_preds)

        n_batches = (len(X_flow_pred) + batch_size - 1) // batch_size
        print(f"  Batch {i // batch_size + 1}/{n_batches}  ({t1 - t0:.3f}s)")

    total_infer_end = time.time()
    perf_metrics['total_inference_time'] = total_infer_end - total_infer_start

    # ---- aggregate predictions ----
    predictions    = np.vstack(all_predictions) if len(all_predictions) > 1 else all_predictions[0]
    avg_prediction = np.mean(predictions, axis=0)
    class_idx      = np.argmax(avg_prediction)
    category       = CATEGORIES[class_idx]
    confidence     = float(avg_prediction[class_idx])

    # ---- timing summary ----
    if perf_metrics['total_sequences'] > 0:
        perf_metrics['avg_time_per_sequence'] = (
            perf_metrics['total_inference_time'] / perf_metrics['total_sequences']
        )
    if perf_metrics['total_frames_processed'] > 0:
        total_elapsed = time.time() - start_time
        perf_metrics['avg_time_per_frame'] = total_elapsed / perf_metrics['total_frames_processed']
        perf_metrics['fps']               = perf_metrics['total_frames_processed'] / total_elapsed

    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Device                   : {perf_metrics['device']}")
    print(f"Total Frames Processed   : {perf_metrics['total_frames_processed']}")
    print(f"Total Sequences Created  : {perf_metrics['total_sequences']}")
    print(f"Total Inference Time     : {perf_metrics['total_inference_time']:.4f}s")
    print(f"Avg Time / Sequence      : {perf_metrics['avg_time_per_sequence']:.4f}s")
    print(f"Avg Time / Frame         : {perf_metrics['avg_time_per_frame']:.4f}s")
    print(f"FPS                      : {perf_metrics['fps']:.2f}")
    print(f"{'='*70}\n")

    # ---- save performance log ----
    os.makedirs('outputs', exist_ok=True)
    log_filename = os.path.join(
        'outputs',
        f"inference_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(log_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("INFERENCE PERFORMANCE LOG\n")
        f.write("="*70 + "\n")
        f.write(f"Video Path          : {video_path}\n")
        f.write(f"Timestamp           : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nPrediction Results:\n")
        f.write(f"  Category          : {category}\n")
        f.write(f"  Confidence        : {confidence * 100:.2f}%\n")
        f.write(f"\nPer-class probabilities:\n")
        for i, cat in enumerate(CATEGORIES):
            f.write(f"  {cat:<12}: {avg_prediction[i] * 100:.2f}%\n")
        f.write(f"\nDevice              : {perf_metrics['device']}\n")
        f.write(f"Total Inference Time: {perf_metrics['total_inference_time']:.4f}s\n")
        f.write(f"Avg Time / Sequence : {perf_metrics['avg_time_per_sequence']:.4f}s\n")
        f.write(f"Avg Time / Frame    : {perf_metrics['avg_time_per_frame']:.4f}s\n")
        f.write(f"FPS                 : {perf_metrics['fps']:.2f}\n")
        f.write("="*70 + "\n")

    print(f"Performance log saved to: {log_filename}")

    # ---- per-class breakdown ----
    print("\nDetailed prediction results:")
    for i, cat in enumerate(CATEGORIES):
        print(f"  {cat:<12}: {avg_prediction[i] * 100:.2f}%")

    return category, confidence, perf_metrics


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stampede Detection — Test-Only Script"
    )
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to the trained .h5 model file'
    )
    parser.add_argument(
        '--video-path', type=str, required=True,
        help='Path to the video file to test (.mp4, .avi, etc.)'
    )
    parser.add_argument(
        '--temp-dir', type=str, default='temp_optical_flow_test',
        help='Temporary directory for optical flow frames (default: temp_optical_flow_test)'
    )
    parser.add_argument(
        '--timeout', type=int, default=300,
        help='Timeout in seconds for inference (default: 300)'
    )
    args = parser.parse_args()

    # ---- GPU setup ----
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s) — memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

    # ---- validate inputs ----
    if not os.path.isfile(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        return
    if not os.path.isfile(args.video_path):
        print(f"ERROR: Video file not found: {args.video_path}")
        return

    # ---- load model ----
    print(f"\nLoading model from: {args.model_path}")
    model = None
    try:
        model = tf.keras.models.load_model(args.model_path, compile=False)
        print("Model loaded successfully via load_model().")
    except Exception as e:
        print(f"load_model() failed: {e}")
        print("Attempting to recreate architecture and load weights...")
        try:
            model = create_enhanced_cnn_lstm_model()
            model.load_weights(args.model_path)
            print("Weights loaded into recreated architecture.")
        except Exception as e2:
            print(f"ERROR: Could not load model or weights: {e2}")
            return

    # ---- run prediction ----
    print(f"\nRunning prediction on: {args.video_path}")
    category, confidence, perf_metrics = predict_with_enhanced_model(
        model,
        args.video_path,
        temp_dir=args.temp_dir,
        timeout_seconds=args.timeout
    )

    # ---- final result ----
    print("\n" + "=" * 50)
    print(f"  Result     : {category.upper()}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
