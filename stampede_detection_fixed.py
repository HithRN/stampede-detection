
#New Code Fully working

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, LSTM, \
    TimeDistributed, Input, Concatenate, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import time
import datetime

# ============= ADDITIONAL IMPORTS FOR EVALUATION =============
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc
)
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import label_binarize


# ============= COMPREHENSIVE EVALUATION PIPELINE =============
def evaluate_model_comprehensive(model, X_flow_val, X_scalar_val, y_val, original_frames_val=None, config=None):
    """
    Comprehensive evaluation: metrics, confusion matrix, classification report, and ROC-AUC
    """

    import numpy as np
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime

    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    
    print("\n" + "="*70)
    print("           COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    if config is None:
        class_names = ["normal", "moderate", "dense", "risky"]
    else:
        class_names = config.get('data', {}).get('class_names', ["normal", "moderate", "dense", "risky"])
    
    # Convert true labels to one-hot
    y_val_onehot = to_categorical(y_val, num_classes=len(class_names))
    #y_test_onehot = to_categorical(y_test, num_classes=len(class_names))
    
    # Get prediction probabilities
    y_pred_proba = model.predict([X_flow_val, X_scalar_val], verbose=1)
    
    # SAFE: Handle NaN/Inf in prediction probabilities
    if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
        print("WARNING: NaN/Inf detected in prediction probabilities. Applying safe fixes...")
        # Replace NaN/Inf with 0 and clip to avoid overflow
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0, posinf=1e6, neginf=-1e6)
        # Ensure each sample sums to 1 (numerical stability)
        prob_sum = np.sum(y_pred_proba, axis=1, keepdims=True)
        # Avoid division by zero
        prob_sum = np.clip(prob_sum, 1e-8, None)
        y_pred_proba = y_pred_proba / prob_sum
    
    # Convert to class predictions
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # === Classification Metrics ===
    print("\n" + "-"*70)
    print("           CLASSIFICATION METRICS (Validation Set)")
    print("-"*70)
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix (Validation):")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Validation Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix_validation.png')
    plt.close()
    
    # Classification Report
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Save report to file
    with open('outputs/figures/classification_report.txt', 'w') as f:
        f.write("Classification Report (Validation Set)\n")
        f.write("="*50 + "\n")
        f.write(classification_report(y_val, y_pred, target_names=class_names))
    
    # === ROC-AUC (One-vs-Rest) ===
    print("\n" + "-"*70)
    print("           ROC-AUC METRICS (One-vs-Rest)")
    print("-"*70)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_names)
    
    # Safe ROC-AUC computation
    for i in range(n_classes):
        # Extract scores for class i
        scores = y_pred_proba[:, i]
        
        # Ensure scores are finite and in [0,1] range
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            print(f"Warning: Invalid scores for class {i} ({class_names[i]}). Clipping...")
            scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        scores = np.clip(scores, 0.0, 1.0)
        
        # Compute ROC curve and AUC
        try:
            fpr[i], tpr[i], _ = roc_curve(y_val_onehot[:, i], scores)
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(f"Class {i} ({class_names[i]}): AUC = {roc_auc[i]:.4f}")
        except Exception as e:
            print(f"Failed to compute ROC for class {i} ({class_names[i]}): {e}")
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 1])
            roc_auc[i] = 0.5  # fallback
    
    # ROC-AUC Plot (One-vs-Rest)
    plt.figure(figsize=(10, 8))
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    
    for i, color in enumerate(colors[:n_classes]):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Validation Set - One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/roc_auc_multiclass.png')
    plt.close()
    
    # === Test Set Evaluation ===
    print("\n" + "-"*70)
    print("           TEST SET METRICS")
    print("-"*70)
    
    # Test set predictions
    '''
    if X_flow_test is not None and X_scalar_test is not None:
        y_test_pred_proba = model.predict([X_flow_test, X_scalar_test], verbose=1)
        y_test_pred = np.argmax(y_test_pred_proba, axis=1)

        # Handle NaN/Inf in test predictions
        if np.any(np.isnan(y_test_pred_proba)) or np.any(np.isinf(y_test_pred_proba)):
            y_test_pred_proba = np.nan_to_num(y_test_pred_proba, nan=0.0, posinf=1e6, neginf=-1e6)
            prob_sum = np.clip(np.sum(y_test_pred_proba, axis=1, keepdims=True), 1e-8, None)
            y_test_pred_proba = y_test_pred_proba / prob_sum
            y_test_pred = np.argmax(y_test_pred_proba, axis=1)
        
        # Test set metrics
        print("\nTest Set Confusion Matrix:")
        cm_test = confusion_matrix(y_test, y_test_pred)
        print(cm_test)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('outputs/figures/confusion_matrix_test.png')
        plt.close()
        
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=class_names))
        
        # Save test report
        with open('outputs/figures/classification_report_test.txt', 'w') as f:
            f.write("Classification Report (Test Set)\n")
            f.write("="*50 + "\n")
            f.write(classification_report(y_test, y_test_pred, target_names=class_names))
        
        # Save test probabilities and predictions for further analysis
        np.save('outputs/results/y_test_pred_proba.npy', y_test_pred_proba)
        np.save('outputs/results/y_test_pred.npy', y_test_pred)
        np.save('outputs/results/y_test_true.npy', y_test)
    '''

    # Save val probabilities and predictions
    np.save('outputs/results/y_val_pred_proba.npy', y_pred_proba)
    np.save('outputs/results/y_val_pred.npy', y_pred)
    np.save('outputs/results/y_val_true.npy', y_val)
    
    # === Evaluation Results Dictionary ===
    evaluation_results = {
        'confusion_matrix_val': cm,
        'classification_report_val': classification_report(y_val, y_pred, target_names=class_names, output_dict=True),
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'class_names': class_names,
        'validation_metrics': {
            'accuracy': np.mean(y_val == y_pred),
            'macro_precision': np.mean([cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0 for i in range(len(class_names))]),
            'macro_recall': np.mean([cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0 for i in range(len(class_names))])
        }
    }
    
    print(f"\nOverall validation accuracy: {evaluation_results['validation_metrics']['accuracy']:.4f}")
    print("\nROC-AUC calculation completed successfully.")
    
    return evaluation_results

def test_on_dataset(model_path, dataset_path, categories=None):
    """
    Load a saved model and run evaluation on a dataset folder structured as:
      dataset_path/
        normal/
        moderate/
        dense/
        risky/

    This function will NOT change the model or the evaluation pipeline; it simply
    reuses existing loader + evaluate_model_comprehensive to evaluate the dataset.
    """
    import tensorflow as tf
    import numpy as np
    import os

    if categories is None:
        categories = ["normal", "moderate", "dense", "risky"]

    print(f"[test_on_dataset] Loading model from: {model_path}")
    # Try loading whole model first; if it fails, try recreating model and loading weights
    model = None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[test_on_dataset] Model loaded with tf.keras.models.load_model()")
    except Exception as e:
        print(f"[test_on_dataset] load_model failed: {e}")
        print("[test_on_dataset] Attempting to recreate architecture and load weights...")
        try:
            model = create_enhanced_cnn_lstm_model()
            model.load_weights(model_path)
            print("[test_on_dataset] Weights loaded into recreated model.")
        except Exception as e2:
            raise RuntimeError(f"[test_on_dataset] Could not load model or weights: {e2}")

    # Load sequences from dataset using existing loader
    print(f"[test_on_dataset] Loading optical-flow sequences from: {dataset_path}")
    X_flow, X_scalar, y, original_frames, sequence_video_ids = load_optical_flow_data(dataset_path, categories)

    if X_flow is None or len(X_flow) == 0:
        print("[test_on_dataset] No sequences found in dataset. Exiting.")
        return None

    # If scalar features are missing, create zero placeholders matching expected shape:
    if X_scalar is None:
        seq_len_minus1 = X_flow.shape[1] if len(X_flow.shape) > 1 else (SEQUENCE_LENGTH - 1)
        print("[test_on_dataset] Scalar features missing; creating zero placeholder of shape "
              f"({X_flow.shape[0]}, {seq_len_minus1}, 4)")
        X_scalar = np.zeros((X_flow.shape[0], seq_len_minus1, 4), dtype=np.float32)

    # Ensure labels are numpy array
    y = np.array(y)

    print(f"[test_on_dataset] Running evaluation on {len(X_flow)} sequences.")
    evaluation_results = evaluate_model_comprehensive(model, X_flow, X_scalar, y, original_frames_val=original_frames,
                                                     config={"dataset_path": dataset_path})
    print("[test_on_dataset] Evaluation complete. Results saved to outputs/ (figures & results).")
    return evaluation_results


# Define constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
SEQUENCE_LENGTH = 16  # Number of frames to consider as one sequence
BATCH_SIZE = 8
EPOCHS = 1
NUM_CLASSES = 4  # normal, moderate, dense, risky


def calculate_flow_acceleration(flow_sequences):
    """
    Calculate frame-wise differences in optical flow (acceleration)
    """
    accelerations = []

    for sequence in flow_sequences:
        # Calculate frame-by-frame differences in optical flow
        seq_acceleration = []
        for i in range(1, len(sequence)):
            # Calculate difference between consecutive frames
            accel = sequence[i] - sequence[i - 1]
            # Compute magnitude of acceleration
            accel_magnitude = np.sqrt(np.sum(accel ** 2, axis=2))
            # Average magnitude across the frame
            mean_accel = np.mean(accel_magnitude)
            seq_acceleration.append(mean_accel)

        # Pad the sequence with a zero at the beginning to maintain sequence length
        seq_acceleration.insert(0, 0)
        accelerations.append(np.array(seq_acceleration))

    return np.array(accelerations)


def calculate_flow_divergence(flow_sequences):
    """
    Calculate spatial divergence of optical flow
    """
    divergences = []

    for sequence in flow_sequences:
        seq_divergence = []
        for flow in sequence:
            # Calculate spatial derivatives
            dx = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)

            # Divergence is the sum of partial derivatives
            divergence = dx + dy

            # Get mean divergence as a scalar feature
            mean_divergence = np.mean(np.abs(divergence))
            seq_divergence.append(mean_divergence)

        divergences.append(np.array(seq_divergence))

    return np.array(divergences)


def calculate_scene_changes(original_frames):
    """
    Detect scene changes using structural similarity (SSIM)
    Returns a sequence of SSIM scores between consecutive frames
    """
    ssim_scores = []

    for sequence in original_frames:
        seq_ssim = []
        for i in range(1, len(sequence)):
            # Convert to grayscale if not already
            if len(sequence[i].shape) == 3:
                frame1 = cv2.cvtColor(sequence[i - 1], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(sequence[i], cv2.COLOR_BGR2GRAY)
            else:
                frame1 = sequence[i - 1]
                frame2 = sequence[i]

            # Calculate SSIM between consecutive frames
            score = ssim(frame1, frame2, data_range=frame2.max() - frame2.min())

            # 1-score to get change metric (higher value means more change)
            change_score = 1.0 - score
            seq_ssim.append(change_score)

        # Pad with zero for first frame
        seq_ssim.insert(0, 0)
        ssim_scores.append(np.array(seq_ssim))

    return np.array(ssim_scores)


def calculate_motion_entropy(flow_sequences):
    """
    Calculate entropy of motion to measure chaos/randomness in the optical flow field
    """
    entropies = []

    for sequence in flow_sequences:
        seq_entropy = []
        for flow in sequence:
            # Calculate magnitude and angle of optical flow
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Calculate entropy of magnitudes
            # First, create histogram
            hist, _ = np.histogram(mag, bins=32, range=(0, np.max(mag) if np.max(mag) > 0 else 1))
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


def load_optical_flow_data(base_dir, categories=["normal", "moderate", "dense", "risky"]):
    """
    Load the optical flow data from directory structure with JPG images directly in category folders.
    Each sequence is tagged with a video-source ID (sequence_video_ids) so that the training /
    validation split can be performed at the VIDEO level, preventing temporal data leakage caused
    by overlapping sequences from the same source being distributed across both splits.
    """
    X = []  # Will contain sequences of optical flow
    original_frames = []  # Will contain original frames for SSIM calculation
    y = []  # Will contain labels
    sequence_video_ids = []  # VIDEO-LEVEL TAG: unique source ID for every sequence

    print(f"Looking for data in: {base_dir}")

    for category_idx, category in enumerate(categories):
        category_path = os.path.join(base_dir, category)
        print(f"Checking category path: {category_path}")
        if not os.path.exists(category_path):
            print(f"Warning: Path {category_path} does not exist")
            continue

        # Check if there are JPG files directly in this folder
        frames = sorted([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.tif'))])

        if frames:
            # If we have JPG files directly in the category folder, treat the whole
            # flat folder as one logical "video" source.
            print(f"Found {len(frames)} frames in {category} folder")
            # Unique source ID for this flat folder: category___flat
            video_id = f"{category}___flat"

            # Create sequences of SEQUENCE_LENGTH frames
            for i in range(0, max(1, len(frames) - SEQUENCE_LENGTH + 1), SEQUENCE_LENGTH // 2):  # 50% overlap
                sequence = []
                orig_sequence = []
                for j in range(i, i + SEQUENCE_LENGTH):
                    if j < len(frames):
                        frame_path = os.path.join(category_path, frames[j])

                        # Process the frame
                        img = cv2.imread(frame_path)
                        if img is None:
                            print(f"Warning: Could not read image {frame_path}")
                            continue

                        # Store original resized frame for SSIM calculation
                        orig_img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                        orig_sequence.append(orig_img)

                        # Extract optical flow from BGR image
                        flow = np.zeros((orig_img.shape[0], orig_img.shape[1], 2), dtype=np.float32)
                        flow[..., 0] = orig_img[..., 2] / 255.0 * 2 - 1  # x direction from R channel
                        flow[..., 1] = orig_img[..., 1] / 255.0 * 2 - 1  # y direction from G channel

                        sequence.append(flow)

                if len(sequence) == SEQUENCE_LENGTH:
                    X.append(np.array(sequence))
                    original_frames.append(np.array(orig_sequence))
                    y.append(category_idx)
                    sequence_video_ids.append(video_id)  # tag with source
        else:
            # Check for subdirectories
            subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
            print(f"Found {len(subfolders)} subfolders in {category} category")

            for subfolder in subfolders:
                subfolder_path = os.path.join(category_path, subfolder)
                frames = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.tif'))])
                print(f"Found {len(frames)} frames in subfolder {subfolder}")
                # Unique source ID per subfolder: category___subfoldername
                video_id = f"{category}___{subfolder}"

                # Create sequences with frames from this subfolder
                for i in range(0, max(1, len(frames) - SEQUENCE_LENGTH + 1), SEQUENCE_LENGTH // 2):
                    sequence = []
                    orig_sequence = []
                    for j in range(i, i + SEQUENCE_LENGTH):
                        if j < len(frames):
                            frame_path = os.path.join(subfolder_path, frames[j])

                            img = cv2.imread(frame_path)
                            if img is None:
                                print(f"Warning: Could not read image {frame_path}")
                                continue

                            # Store original resized frame for SSIM calculation
                            orig_img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            orig_sequence.append(orig_img)

                            flow = np.zeros((orig_img.shape[0], orig_img.shape[1], 2), dtype=np.float32)
                            flow[..., 0] = orig_img[..., 2] / 255.0 * 2 - 1
                            flow[..., 1] = orig_img[..., 1] / 255.0 * 2 - 1

                            sequence.append(flow)

                    if len(sequence) == SEQUENCE_LENGTH:
                        X.append(np.array(sequence))
                        original_frames.append(np.array(orig_sequence))
                        y.append(category_idx)
                        sequence_video_ids.append(video_id)  # tag with source

    X = np.array(X) if X else np.array([])
    original_frames = np.array(original_frames) if original_frames else np.array([])
    y = np.array(y) if y else np.array([])

    print(f"Loaded {len(X)} sequences with shape {X[0].shape if len(X) > 0 else 'N/A'}")
    print(f"Loaded {len(y)} labels")

    # Calculate additional features
    if len(X) > 0:
        print("Calculating additional features...")

        # Flow acceleration features
        flow_acceleration = calculate_flow_acceleration(X)
        print(f"Flow acceleration shape: {flow_acceleration.shape}")

        # Flow divergence features
        flow_divergence = calculate_flow_divergence(X)
        print(f"Flow divergence shape: {flow_divergence.shape}")

        # Scene change detection using SSIM
        scene_changes = calculate_scene_changes(original_frames)
        print(f"Scene changes shape: {scene_changes.shape}")

        # Motion entropy
        motion_entropy = calculate_motion_entropy(X)
        print(f"Motion entropy shape: {motion_entropy.shape}")

        # Combine all scalar features
        scalar_features = np.stack([
            flow_acceleration,
            flow_divergence,
            scene_changes,
            motion_entropy
        ], axis=2)  # Shape: [num_sequences, sequence_length, 4]

        # --- FINAL CLEANUP (keep your existing scalar_features computation above this) ---
        # ensure X and scalar_features exist (original code computed them)
        try:
            scalar_features = np.nan_to_num(scalar_features, nan=0.0, posinf=1e3, neginf=-1e3)
        except Exception:
            scalar_features = None

        # ensure X numeric stability
        try:
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception:
            pass

        # Ensure original_frames is a numpy array (if present) or an empty array with proper shape
        if len(original_frames) > 0:
            original_frames_arr = np.array(original_frames)
        else:
            # keep consistent shape if needed: (0, seq_len, H, W, C)
            original_frames_arr = np.zeros((0, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        # sequence_video_ids is populated throughout the loop above; it contains one
        # string ID per sequence and is always a plain list at this point.

        # Now return five outputs in all cases (caller must unpack five)
        # If scalar_features is None, return None in the second slot (as earlier)
        return X, scalar_features, np.array(y), original_frames_arr, sequence_video_ids

    return X, None, y



def create_enhanced_cnn_lstm_model():
    """
    Create an enhanced CNN-LSTM model that incorporates additional motion features
    """
    # Input shapes
    flow_input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 2)  # Optical flow input
    scalar_features_shape = (SEQUENCE_LENGTH, 4)  # 4 scalar features per frame

    # Optical flow input branch
    flow_input = Input(shape=flow_input_shape, name='optical_flow_input')

    # CNN feature extraction with TimeDistributed to process each frame
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(flow_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    # Flatten CNN output for each time step
    x = TimeDistributed(Flatten())(x)

    # Scalar features input branch
    scalar_input = Input(shape=scalar_features_shape, name='scalar_features_input')

    # Concatenate CNN features with scalar features
    combined = Concatenate(axis=2)([x, scalar_input])

    # LSTM to capture temporal patterns
    lstm1 = LSTM(256, return_sequences=True)(combined)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(128)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)

    # Final classification layers
    dense1 = Dense(64, activation='relu')(dropout2)
    bn = BatchNormalization()(dense1)
    output = Dense(NUM_CLASSES, activation='softmax')(bn)

    # Create model with multiple inputs
    model = Model(inputs=[flow_input, scalar_input], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def log_hyperparameters_to_json(model, batch_size, epochs, learning_rate,
                                  optimizer_name, scheduler_info=None,
                                  weight_init=None, log_file='hyperparameters.json'):
    """
    Log all training hyperparameters to a JSON file at training start.

    Args:
        model: Keras model instance
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate value
        optimizer_name: Name of optimizer (e.g., 'Adam', 'SGD')
        scheduler_info: Dictionary with scheduler details (optional)
        weight_init: Weight initialization method (optional)
        log_file: Path to save JSON log file
    """
    import json
    import datetime
    import numpy as np

    # Helper function to convert numpy/tf types to native Python types
    def convert_to_serializable(obj):
        """Convert numpy/tensorflow types to JSON serializable types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Extract optimizer details from model
    optimizer_config = model.optimizer.get_config()
    # Convert optimizer config to serializable format
    optimizer_config = convert_to_serializable(optimizer_config)

    # Get parameter counts safely
    total_params = int(model.count_params())
    trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    non_trainable_params = int(sum([tf.size(w).numpy() for w in model.non_trainable_weights]))

    # Prepare hyperparameters dictionary
    hyperparameters = {
        "training_configuration": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "epochs": int(epochs)
        },
        "optimizer": {
            "name": optimizer_name,
            "type": model.optimizer.__class__.__name__,
            "config": optimizer_config
        },
        "model_architecture": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "input_shapes": {
                "flow_input": str(model.input[0].shape),
                "scalar_input": str(model.input[1].shape)
            },
            "output_shape": str(model.output.shape)
        },
        "loss_function": str(model.loss) 
    }
        #"metrics": [m if isinstance(m, str) else m.__name__ for m in model.compiled_metrics._metrics]
        # OLD (line 640):
# "metrics": [m if isinstance(m, str) else m.__name__ for m in model.compiled_metrics._metrics]

# NEW (replacement):
        
    try:
        if hasattr(model, 'metrics_names'):
            metrics_list = model.metrics_names
        else:
            metrics_list = ['accuracy']
    except:
        metrics_list = ['accuracy']

    hyperparameters["metrics"] = metrics_list

    
    

    # Add scheduler information if provided
    if scheduler_info:
        hyperparameters["learning_rate_scheduler"] = convert_to_serializable(scheduler_info)
    else:
        hyperparameters["learning_rate_scheduler"] = "None"

    # Add weight initialization if provided
    if weight_init:
        hyperparameters["weight_initialization"] = str(weight_init)
    else:
        # Try to extract from model layers
        init_methods = []
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                init_methods.append({
                    "layer": layer.name,
                    "initializer": layer.kernel_initializer.__class__.__name__
                })
        hyperparameters["weight_initialization"] = init_methods if init_methods else "default (GlorotUniform)"

    # Save to JSON file
    with open(log_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    print(f"\n{'='*70}")
    print(f"Hyperparameters logged to: {log_file}")
    print(f"{'='*70}")

    # Print summary
    print("\nTraining Configuration Summary:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Scheduler: {hyperparameters['learning_rate_scheduler']}")
    print(f"  Total Parameters: {hyperparameters['model_architecture']['total_params']:,}")
    print(f"{'='*70}\n")

    return hyperparameters

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

def create_output_folders(base_dir='paper_figures'):
    """
    Create structured folder hierarchy for saving visualizations

    Returns:
        dict: Dictionary with paths to all output folders
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    folders = {
        'base': base_dir,
        'optical_flow': os.path.join(base_dir, 'optical_flow_visualizations'),
        'predictions': os.path.join(base_dir, 'prediction_overlays'),
        'correct': os.path.join(base_dir, 'correct_predictions'),
        'incorrect': os.path.join(base_dir, 'incorrect_predictions'),
        'comparison': os.path.join(base_dir, 'correct_vs_incorrect'),
        'timestamp': timestamp
    }

    # Create all directories
    for folder_name, folder_path in folders.items():
        if folder_name != 'timestamp':
            os.makedirs(folder_path, exist_ok=True)

    print(f"\n{'='*70}")
    print("OUTPUT FOLDERS CREATED")
    print(f"{'='*70}")
    print(f"Base directory: {base_dir}")
    for name, path in folders.items():
        if name not in ['base', 'timestamp']:
            print(f"  • {name}: {path}")
    print(f"{'='*70}\n")

    return folders


def visualize_optical_flow(flow, save_path=None, title="Optical Flow"):
    """
    Create a colorful HSV visualization of optical flow

    Args:
        flow: Optical flow array (H, W, 2)
        save_path: Path to save the visualization
        title: Title for the plot

    Returns:
        hsv_rgb: RGB image of the flow visualization
    """
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude

    # Convert to RGB for display/saving
    hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if save_path:
        cv2.imwrite(save_path, hsv_rgb)

    return hsv_rgb


def create_flow_grid_visualization(flow_sequence, save_path, max_frames=8):
    """
    Create a grid visualization of optical flow sequence

    Args:
        flow_sequence: Sequence of flow frames (T, H, W, 2)
        save_path: Path to save the visualization
        max_frames: Maximum number of frames to display
    """
    num_frames = min(len(flow_sequence), max_frames)
    cols = 4
    rows = (num_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for idx in range(num_frames):
        flow = flow_sequence[idx]
        flow_vis = visualize_optical_flow(flow)
        flow_vis_rgb = cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(flow_vis_rgb)
        axes[idx].set_title(f'Frame {idx+1}', fontsize=10)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved optical flow grid: {save_path}")


def create_prediction_overlay(original_frame, prediction_probs, true_label=None,
                              save_path=None, class_names=None):
    """
    Create an overlay showing prediction probabilities on the original frame
    """
    if class_names is None:
        class_names = ["normal", "moderate", "dense", "risky"]
    
    # Create a copy of the frame
    overlay_frame = original_frame.copy()
    h, w = overlay_frame.shape[:2]
    
    # FIXED: Check for NaN values in predictions
    if np.any(np.isnan(prediction_probs)):
        print(f"Warning: NaN detected in predictions: {prediction_probs}")
        # Replace NaN with zeros
        prediction_probs = np.nan_to_num(prediction_probs, nan=0.0)
        # Ensure probabilities sum to 1
        prob_sum = np.sum(prediction_probs)
        if prob_sum > 0:
            prediction_probs = prediction_probs / prob_sum
        else:
            prediction_probs = np.ones(len(class_names)) / len(class_names)
    
    # Get predicted class
    pred_class = np.argmax(prediction_probs)
    pred_confidence = prediction_probs[pred_class]
    
    # Define colors for each class (BGR format)
    class_colors = {
        0: (0, 255, 0),      # Normal - Green
        1: (0, 255, 255),    # Moderate - Yellow
        2: (0, 165, 255),    # Dense - Orange
        3: (0, 0, 255)       # Risky - Red
    }
    
    # Draw prediction box
    box_height = 150
    box_y_start = h - box_height - 10
    cv2.rectangle(overlay_frame, (10, box_y_start), (w - 10, h - 10), (0, 0, 0), -1)
    overlay = overlay_frame.copy()
    cv2.rectangle(overlay, (10, box_y_start), (w - 10, h - 10), (0, 0, 0), -1)
    overlay_frame = cv2.addWeighted(overlay, 0.6, overlay_frame, 0.4, 0)
    
    # Add prediction text
    y_offset = box_y_start + 30
    pred_text = f"Prediction: {class_names[pred_class].upper()}"
    conf_text = f"Confidence: {pred_confidence*100:.1f}%"
    
    cv2.putText(overlay_frame, pred_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, class_colors[pred_class], 2)
    cv2.putText(overlay_frame, conf_text, (20, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add probability bars
    bar_y = y_offset + 60
    bar_width = w - 100
    bar_height = 15
    
    for i, (class_name, prob) in enumerate(zip(class_names, prediction_probs)):
        # FIXED: Handle NaN in individual probabilities
        if np.isnan(prob) or np.isinf(prob):
            prob = 0.0
        
        # Clamp probability to [0, 1] range
        prob = np.clip(prob, 0.0, 1.0)
        
        # Class label
        cv2.putText(overlay_frame, class_name[:4].upper(), (20, bar_y + i*25 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Probability bar background
        cv2.rectangle(overlay_frame, (100, bar_y + i*25),
                     (100 + bar_width, bar_y + i*25 + bar_height), (50, 50, 50), -1)
        
        # Probability bar fill (safe conversion)
        fill_width = int(bar_width * prob)
        cv2.rectangle(overlay_frame, (100, bar_y + i*25),
                     (100 + fill_width, bar_y + i*25 + bar_height),
                     class_colors[i], -1)
        
        # Percentage text
        cv2.putText(overlay_frame, f"{prob*100:.1f}%",
                   (105 + bar_width, bar_y + i*25 + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add true label if provided
    if true_label is not None:
        true_text = f"True: {class_names[true_label].upper()}"
        color = (0, 255, 0) if true_label == pred_class else (0, 0, 255)
        cv2.putText(overlay_frame, true_text, (w - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if save_path:
        cv2.imwrite(save_path, overlay_frame)
    
    return overlay_frame



def save_prediction_visualizations(model, X_flow_val, X_scalar_val, y_val,
                                     original_frames_val, folders,
                                     num_samples=10, class_names=None):
    """
    Generate and save visualizations for correct and incorrect predictions

    Args:
        model: Trained model
        X_flow_val: Validation optical flow data
        X_scalar_val: Validation scalar features
        y_val: Validation labels (integer format)
        original_frames_val: Original frames corresponding to validation data
        folders: Dictionary of output folder paths
        num_samples: Number of samples to visualize per category
        class_names: List of class names
    """
    if class_names is None:
        class_names = ["normal", "moderate", "dense", "risky"]

    print(f"\n{'='*70}")
    print("GENERATING PREDICTION VISUALIZATIONS")
    print(f"{'='*70}")

    # Get predictions
    y_pred_probs = model.predict([X_flow_val, X_scalar_val], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Find correct and incorrect predictions
    correct_mask = (y_pred == y_val)
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    print(f"\nCorrect predictions: {len(correct_indices)}/{len(y_val)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}/{len(y_val)}")

    # Save optical flow visualizations
    print(f"\nSaving optical flow visualizations...")
    for idx in range(min(num_samples, len(X_flow_val))):
        flow_seq = X_flow_val[idx]
        save_path = os.path.join(folders['optical_flow'],
                                 f'flow_sequence_{idx:03d}_class_{class_names[y_val[idx]]}.png')
        create_flow_grid_visualization(flow_seq, save_path)

    # Save correct predictions
    print(f"\nSaving correct prediction overlays...")
    samples_per_class = num_samples // len(class_names)
    for class_id in range(len(class_names)):
        class_correct = correct_indices[y_val[correct_indices] == class_id]
        for i, idx in enumerate(class_correct[:samples_per_class]):
            if len(original_frames_val[idx]) > 0:
                # Use middle frame from sequence
                mid_frame = original_frames_val[idx][len(original_frames_val[idx])//2]
                save_path = os.path.join(folders['correct'],
                                         f'correct_{class_names[class_id]}_{i:02d}.png')
                create_prediction_overlay(mid_frame, y_pred_probs[idx], y_val[idx],
                                          save_path, class_names)

    # Save incorrect predictions
    print(f"\nSaving incorrect prediction overlays...")
    for i, idx in enumerate(incorrect_indices[:num_samples]):
        if len(original_frames_val[idx]) > 0:
            mid_frame = original_frames_val[idx][len(original_frames_val[idx])//2]
            save_path = os.path.join(folders['incorrect'],
                                     f'incorrect_{i:02d}_true_{class_names[y_val[idx]]}_pred_{class_names[y_pred[idx]]}.png')
            create_prediction_overlay(mid_frame, y_pred_probs[idx], y_val[idx],
                                      save_path, class_names)

    # Create comparison figure (correct vs incorrect)
    print(f"\nCreating correct vs incorrect comparison...")
    create_comparison_figure(correct_indices, incorrect_indices, y_val, y_pred, y_pred_probs,
                              original_frames_val, folders, class_names)

    print(f"\n{'='*70}")
    print("VISUALIZATION GENERATION COMPLETE")
    print(f"{'='*70}\n")


def create_comparison_figure(correct_indices, incorrect_indices, y_val, y_pred, y_pred_probs,
                               original_frames_val, folders, class_names, num_pairs=4):
    """
    Create side-by-side comparison of correct vs incorrect predictions
    """
    fig, axes = plt.subplots(num_pairs, 2, figsize=(12, 4*num_pairs))

    for i in range(num_pairs):
        # Correct prediction
        if i < len(correct_indices):
            correct_idx = correct_indices[i]
            if len(original_frames_val[correct_idx]) > 0:
                frame_correct = original_frames_val[correct_idx][len(original_frames_val[correct_idx])//2]
                frame_with_overlay = create_prediction_overlay(
                    frame_correct, y_pred_probs[correct_idx], y_val[correct_idx],
                    None, class_names
                )
                axes[i, 0].imshow(cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB))
                axes[i, 0].set_title(f'CORRECT: {class_names[y_val[correct_idx]]}',
                                     color='green', fontweight='bold')
                axes[i, 0].axis('off')

        # Incorrect prediction
        if i < len(incorrect_indices):
            incorrect_idx = incorrect_indices[i]
            if len(original_frames_val[incorrect_idx]) > 0:
                frame_incorrect = original_frames_val[incorrect_idx][len(original_frames_val[incorrect_idx])//2]
                frame_with_overlay = create_prediction_overlay(
                    frame_incorrect, y_pred_probs[incorrect_idx], y_val[incorrect_idx],
                    None, class_names
                )
                axes[i, 1].imshow(cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB))
                axes[i, 1].set_title(
                    f'INCORRECT: True={class_names[y_val[incorrect_idx]]}, Pred={class_names[y_pred[incorrect_idx]]}',
                    color='red', fontweight='bold'
                )
                axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(folders['comparison'], 'correct_vs_incorrect_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison figure: {save_path}")


def video_level_split(y, sequence_video_ids, val_size=0.2, random_state=42):
    """
    Perform a LEAK-FREE train / validation split at the VIDEO SOURCE level.

    Problem solved
    --------------
    Sequences are built with a 50 % sliding-window overlap, so adjacent sequences
    share 8 frames.  A random frame-level split (sklearn train_test_split) scatters
    those overlapping sequences across both partitions, leaking ground-truth temporal
    context from training into validation and artificially inflating every metric.

    Solution
    --------
    1. Collect every *unique* video source ID.
    2. Split the unique sources 80 / 20 per class (stratified by the modal label of
       each source, so class balance is preserved as closely as possible).
    3. Assign every sequence whose source ID ended up in the validation pool to the
       validation set; all others go to training.

    This guarantees that no frame from a validation video is ever seen during
    training, regardless of the overlap between consecutive sequences.

    Parameters
    ----------
    y                 : np.ndarray  -- integer class label for each sequence
    sequence_video_ids: list[str]   -- one source-ID string per sequence
                                       (populated by load_optical_flow_data)
    val_size          : float       -- fraction of VIDEO SOURCES to hold out (default 0.2)
    random_state      : int         -- reproducibility seed

    Returns
    -------
    train_idx : np.ndarray  -- sequence indices assigned to training
    val_idx   : np.ndarray  -- sequence indices assigned to validation
    """
    import math
    rng = np.random.default_rng(random_state)

    sequence_video_ids = np.asarray(sequence_video_ids)
    y = np.asarray(y)
    all_indices = np.arange(len(y))

    # Map each unique video source to its sequences and their majority label
    unique_sources = np.unique(sequence_video_ids)
    source_label = {}   # source_id -> class label (modal)
    for src in unique_sources:
        mask = sequence_video_ids == src
        labels_for_src = y[mask]
        # majority vote; ties broken by lowest label index
        counts = np.bincount(labels_for_src)
        source_label[src] = int(np.argmax(counts))

    # Group sources by class so we can stratify
    class_sources = {}
    for src, lbl in source_label.items():
        class_sources.setdefault(lbl, []).append(src)

    train_sources = set()
    val_sources   = set()

    for lbl, sources in class_sources.items():
        sources = list(sources)
        rng.shuffle(sources)
        n_val = max(1, math.ceil(len(sources) * val_size))
        # Guard: always keep at least one source in training
        if n_val >= len(sources):
            n_val = max(1, len(sources) - 1)
        val_sources.update(sources[:n_val])
        train_sources.update(sources[n_val:])

    train_idx = all_indices[np.isin(sequence_video_ids, list(train_sources))]
    val_idx   = all_indices[np.isin(sequence_video_ids, list(val_sources))]

    # Diagnostic summary
    print("\n" + "="*70)
    print("VIDEO-LEVEL SPLIT SUMMARY  (leak-free)")
    print("="*70)
    print(f"  Total unique video sources : {len(unique_sources)}")
    print(f"  Training sources           : {len(train_sources)}")
    print(f"  Validation sources         : {len(val_sources)}")
    print(f"  Training sequences         : {len(train_idx)}")
    print(f"  Validation sequences       : {len(val_idx)}")
    # Verify zero overlap between source sets
    overlap = train_sources & val_sources
    if overlap:
        print(f"  WARNING: {len(overlap)} sources appear in BOTH splits (should be 0): {overlap}")
    else:
        print("  Overlap between splits     : 0  (no temporal leakage)")
    # Per-class breakdown
    class_names_diag = ["normal", "moderate", "dense", "risky"]
    print("\n  Per-class sequence counts:")
    print(f"  {'Class':<12}  {'Train':>8}  {'Val':>8}")
    print(f"  {'-'*30}")
    for cls_idx, cls_name in enumerate(class_names_diag):
        n_tr  = int(np.sum(y[train_idx] == cls_idx))
        n_val_cls = int(np.sum(y[val_idx] == cls_idx))
        print(f"  {cls_name:<12}  {n_tr:>8}  {n_val_cls:>8}")
    print("="*70 + "\n")

    return train_idx, val_idx


# Modified train_enhanced_model to include visualization generation
def train_enhanced_model_with_visualizations(X_flow, X_scalar, y, original_frames=None, sequence_video_ids=None):
    """
    Enhanced training function that generates visualizations after training

    This wraps the existing train_enhanced_model function and adds visualization generation
    """
    import csv
    import os
    import json
    import datetime

    initial_epoch = 0

    #sanity check
    print("X_flow stats:", np.min(X_flow), np.max(X_flow), np.isnan(X_flow).any(), np.isinf(X_flow).any())
    print("X_scalar stats:", np.min(X_scalar), np.max(X_scalar), np.isnan(X_scalar).any(), np.isinf(X_scalar).any())


    # Convert labels to one-hot encoding
    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    # -------------------------------------------------------------------------
    # VIDEO-LEVEL SPLIT — replaces the previous random train_test_split.
    #
    # Why: sequences are built with 50 % overlap, so adjacent sequences share
    # 8 frames.  A random sequence-level split scatters those overlapping
    # sequences across both partitions (temporal leakage), inflating metrics.
    # Splitting by VIDEO SOURCE ensures no frame ever appears in both splits.
    # -------------------------------------------------------------------------
    if sequence_video_ids is not None and len(sequence_video_ids) == len(y):
        train_idx, val_idx = video_level_split(y, sequence_video_ids, val_size=0.2, random_state=42)
    else:
        # Fallback: if caller did not pass video IDs (e.g. legacy path),
        # warn and fall back to random split so the rest of the pipeline still works.
        print("WARNING: sequence_video_ids not provided or length mismatch. "
              "Falling back to random train_test_split — temporal leakage is possible.")
        from sklearn.model_selection import train_test_split as _tts
        all_idx = np.arange(len(y))
        train_idx, val_idx = _tts(all_idx, test_size=0.2, random_state=42, stratify=y)

    X_flow_train,   X_flow_val   = X_flow[train_idx],   X_flow[val_idx]
    X_scalar_train, X_scalar_val = X_scalar[train_idx], X_scalar[val_idx]
    y_train        = y_onehot[train_idx]
    y_val_onehot   = y_onehot[val_idx]
    y_train_labels = y[train_idx]
    y_val_labels   = y[val_idx]

    # Also split original frames if provided
    if original_frames is not None and len(original_frames) > 0:
        original_frames_val = original_frames[val_idx]
    else:
        original_frames_val = None

    # Create enhanced model
    model = create_enhanced_cnn_lstm_model()
    model.summary()

    # Log hyperparameters
    learning_rate = 0.001
    optimizer_name = "Adam"

    hyperparams = log_hyperparameters_to_json(
        model=model, batch_size=BATCH_SIZE, epochs=EPOCHS,
        learning_rate=learning_rate, optimizer_name=optimizer_name,
        log_file='hyperparameters.json'
    )

    # Setup callbacks
    checkpoint = ModelCheckpoint('enhanced_stampede_detection_best.h5',
                                  monitor='val_accuracy', save_best_only=True,
                                  mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                    restore_best_weights=True, verbose=1)

    # Training logs setup
    training_logs = {'epoch': [], 'train_loss': [], 'train_accuracy': [],
                     'val_loss': [], 'val_accuracy': []}

    with open('training_logs.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        writer.writeheader()

    # Train model
    history = model.fit(
        [X_flow_train, X_scalar_train], y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=([X_flow_val, X_scalar_val], y_val_onehot),
        callbacks=[checkpoint, early_stopping]
    )

    # Extract and save logs (same as before)
    for epoch in range(len(history.history['loss'])):
        training_logs['epoch'].append(epoch + 1)
        training_logs['train_loss'].append(float(history.history['loss'][epoch]))
        training_logs['train_accuracy'].append(float(history.history['accuracy'][epoch]))
        training_logs['val_loss'].append(float(history.history['val_loss'][epoch]))
        training_logs['val_accuracy'].append(float(history.history['val_accuracy'][epoch]))

    # Save training plots (same as before)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_logs['epoch'], training_logs['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(training_logs['epoch'], training_logs['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(training_logs['epoch'], training_logs['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(training_logs['epoch'], training_logs['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============= GENERATE VISUALIZATIONS =============
    folders = create_output_folders('paper_figures')

    if original_frames_val is not None:
        save_prediction_visualizations(
            model, X_flow_val, X_scalar_val, y_val_labels,
            original_frames_val, folders, num_samples=20
        )

    # Run comprehensive evaluation
    evaluation_results = evaluate_model_comprehensive(
    model,
    X_flow_val,
    X_scalar_val,
    y_val_labels,           # pass integer labels here
    original_frames_val=original_frames_val,  # optional if you later use them
    config=None             # or pass your config dict if you have one
)



    model.save('enhanced_stampede_detection_final.h5')

    return model, history, initial_epoch + EPOCHS



def generate_optical_flow_and_features_from_video(video_path, output_dir, resize_dim=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                  max_frames=200, frame_skip=5):
    """
    Generate optical flow and additional features from video
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate time intervals for frame sampling - aim for ~4 seconds of content
    target_duration = 4  # seconds
    target_frames = min(max_frames, int(target_duration * fps))

    # Calculate frame indices to capture
    if total_frames <= target_frames:
        frame_indices = list(range(0, total_frames, frame_skip))
    else:
        # Pick frames evenly distributed across the video
        frame_indices = [int(i * total_frames / target_frames) for i in range(target_frames)]

    print(f"Video: {total_frames} frames at {fps} FPS. Using {len(frame_indices)} frames for analysis.")

    # Even more aggressive downsampling
    downsample_factor = 0.2  # Further reduced from 0.25 to 0.2

    # Process frames
    flow_frames = []
    original_frames = []
    processed_count = 0
    prev_gray = None

    for frame_idx in frame_indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame at index {frame_idx}")
            continue

        # Save original frame for SSIM calculation
        resized_frame = cv2.resize(frame, resize_dim)
        original_frames.append(resized_frame)

        # Downsample for speed
        frame = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, resize_dim)

        # Skip optical flow calculation for the first frame
        if prev_gray is None:
            prev_gray = gray
            continue

        # Calculate optical flow with simplified parameters
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=2,  # Reduced from 3 to 2
            winsize=11,  # Reduced from 15 to 11
            iterations=2,  # Reduced from 3 to 2
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Only save visualization for debugging on selected frames
        if processed_count % 10 == 0:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            output_file = os.path.join(output_dir, f"frame_{processed_count:04d}.jpg")
            cv2.imwrite(output_file, rgb)

        # Store flow for prediction
        flow_frames.append(flow)
        prev_gray = gray
        processed_count += 1

        # Print progress
        if processed_count % 5 == 0:
            print(f"Processed {processed_count}/{len(frame_indices)} frames")

    cap.release()
    print(f"Generated {len(flow_frames)} optical flow frames")

    # Create dummy original frames array if empty
    if not original_frames and flow_frames:
        original_frames = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)] * len(flow_frames)

    return flow_frames, original_frames


def predict_with_enhanced_model(model, video_path, temp_dir="temp_optical_flow", timeout_seconds=300):
    """
    Predict the risk level for a video using the enhanced model with performance monitoring

    Returns:
        category: Predicted class
        confidence: Prediction confidence
        perf_metrics: Dictionary containing performance metrics
    """

    # Initialize performance tracking
    perf_metrics = {
        'device': 'GPU' if len(tf.config.list_physical_devices('GPU')) > 0 else 'CPU',
        'batch_size': 8,
        'total_frames_processed': 0,
        'total_sequences': 0,
        'frame_processing_times': [],
        'sequence_inference_times': [],
        'total_inference_time': 0.0,
        'avg_time_per_frame': 0.0,
        'avg_time_per_sequence': 0.0,
        'fps': 0.0
    }

    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE MONITORING")
    print(f"{'='*70}")
    print(f"Device: {perf_metrics['device']}")
    print(f"Batch Size: {perf_metrics['batch_size']}")
    print(f"{'='*70}\n")

    def test_model_inference(model):
        """Test that model inference is working and measure its speed"""
        print("Testing model inference speed...")
        dummy_flow_input = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 2), dtype=np.float32)
        dummy_scalar_input = np.zeros((1, SEQUENCE_LENGTH, 4), dtype=np.float32)

        start = time.time()
        dummy_pred = model.predict([dummy_flow_input, dummy_scalar_input], verbose=0)
        end = time.time()

        print(f"Model inference test: {end - start:.3f} seconds for a single sequence")
        print(f"Prediction shape: {dummy_pred.shape}, values: min={dummy_pred.min():.3f}, max={dummy_pred.max():.3f}")
        return end - start

    warmup_time = test_model_inference(model)


    start_time = time.time()

    def check_timeout():
        if time.time() - start_time > timeout_seconds:
            print(f"Processing timed out after {timeout_seconds} seconds!")
            return True
        return False

    os.makedirs(temp_dir, exist_ok=True)
    print(f"Generating optical flow and features for video: {video_path}")

    frame_start = time.time()
    flow_frames, original_frames = generate_optical_flow_and_features_from_video(
        video_path,
        temp_dir,
        max_frames=200,
        frame_skip=5
    )
    frame_end = time.time()

    perf_metrics['total_frames_processed'] = len(flow_frames)
    perf_metrics['frame_processing_times'].append(frame_end - frame_start)

    if check_timeout():
        return "Timeout (processing took too long)", 0.0, perf_metrics

    if len(flow_frames) < SEQUENCE_LENGTH:
        print(f"Warning: Video has only {len(flow_frames)} frames, which is less than required {SEQUENCE_LENGTH}")
        return "Unknown (not enough frames)", 0.0, perf_metrics

    print(f"Creating sequences from {len(flow_frames)} frames...")

    flow_sequences = []
    orig_sequences = []
    max_sequences = 20

    if len(flow_frames) > SEQUENCE_LENGTH:
        step_size = max(1, (len(flow_frames) - SEQUENCE_LENGTH) // max_sequences)
        start_indices = range(0, len(flow_frames) - SEQUENCE_LENGTH, step_size)
        start_indices = list(start_indices)[:max_sequences]

        for i in start_indices:
            flow_seq = flow_frames[i:i + SEQUENCE_LENGTH]
            orig_seq = original_frames[i:i + SEQUENCE_LENGTH] if i < len(original_frames) else []

            flow_sequences.append(np.array(flow_seq))
            orig_sequences.append(np.array(orig_seq) if orig_seq else None)

            if check_timeout():
                return "Timeout (processing took too long)", 0.0, perf_metrics

    if not flow_sequences:
        return "Unknown (could not create sequences)", 0.0, perf_metrics

    perf_metrics['total_sequences'] = len(flow_sequences)

    print("Calculating additional features for prediction...")

    feature_start = time.time()
    flow_acceleration = calculate_flow_acceleration(flow_sequences)
    flow_divergence = calculate_flow_divergence(flow_sequences)
    scene_changes = calculate_scene_changes(orig_sequences) if all(seq is not None for seq in orig_sequences) else np.zeros((len(flow_sequences), SEQUENCE_LENGTH))
    motion_entropy = calculate_motion_entropy(flow_sequences)

    scalar_features = np.stack([
        flow_acceleration,
        flow_divergence,
        scene_changes,
        motion_entropy
    ], axis=2)
    feature_end = time.time()

    print(f"Prepared {len(flow_sequences)} sequences with additional features for prediction")

    X_flow_pred = np.array(flow_sequences)
    X_scalar_pred = scalar_features

    batch_size = perf_metrics['batch_size']
    all_predictions = []

    print(f"Making predictions in batches of {batch_size}...")

    total_inference_start = time.time()

    for i in range(0, len(X_flow_pred), batch_size):
        batch_flow = X_flow_pred[i:i + batch_size]
        batch_scalar = X_scalar_pred[i:i + batch_size]

        batch_start = time.time()
        batch_preds = model.predict([batch_flow, batch_scalar], verbose=0)
        batch_end = time.time()

        perf_metrics['sequence_inference_times'].append(batch_end - batch_start)
        all_predictions.append(batch_preds)

        print(f"Processed batch {i // batch_size + 1}/{(len(X_flow_pred) + batch_size - 1) // batch_size} "
              f"(Time: {batch_end - batch_start:.3f}s)")

    total_inference_end = time.time()
    perf_metrics['total_inference_time'] = total_inference_end - total_inference_start

    predictions = np.vstack(all_predictions) if len(all_predictions) > 1 else all_predictions[0]
    avg_prediction = np.mean(predictions, axis=0)
    class_idx = np.argmax(avg_prediction)

    categories = ["normal", "moderate", "dense", "risky"]
    category = categories[class_idx]
    confidence = avg_prediction[class_idx]

    if perf_metrics['total_sequences'] > 0:
        perf_metrics['avg_time_per_sequence'] = perf_metrics['total_inference_time'] / perf_metrics['total_sequences']

    if perf_metrics['total_frames_processed'] > 0:
        total_time = time.time() - start_time
        perf_metrics['avg_time_per_frame'] = total_time / perf_metrics['total_frames_processed']
        perf_metrics['fps'] = perf_metrics['total_frames_processed'] / total_time

    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Device Used: {perf_metrics['device']}")
    print(f"Batch Size: {perf_metrics['batch_size']}")
    print(f"Total Frames Processed: {perf_metrics['total_frames_processed']}")
    print(f"Total Sequences Created: {perf_metrics['total_sequences']}")
    print(f"\nTiming Metrics:")
    print(f"  Total Inference Time: {perf_metrics['total_inference_time']:.4f} seconds")
    print(f"  Average Time per Sequence (16 frames): {perf_metrics['avg_time_per_sequence']:.4f} seconds")
    print(f"  Average Time per Frame: {perf_metrics['avg_time_per_frame']:.4f} seconds")
    print(f"  FPS (Frames Per Second): {perf_metrics['fps']:.2f}")
    print(f"{'='*70}\n")

    log_filename = f"inference_performance_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w') as log_file:
        log_file.write("="*70 + "\n")
        log_file.write("INFERENCE PERFORMANCE LOG\n")
        log_file.write("="*70 + "\n")
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"\nPrediction Results:\n")
        log_file.write(f"  Category: {category}\n")
        log_file.write(f"  Confidence: {confidence * 100:.2f}%\n")
        log_file.write(f"\nDevice Configuration:\n")
        log_file.write(f"  Device: {perf_metrics['device']}\n")
        log_file.write(f"  Batch Size: {perf_metrics['batch_size']}\n")
        log_file.write(f"\nProcessing Statistics:\n")
        log_file.write(f"  Total Frames Processed: {perf_metrics['total_frames_processed']}\n")
        log_file.write(f"  Total Sequences Created: {perf_metrics['total_sequences']}\n")
        log_file.write(f"\nTiming Metrics:\n")
        log_file.write(f"  Total Inference Time: {perf_metrics['total_inference_time']:.4f} seconds\n")
        log_file.write(f"  Average Time per Sequence (16 frames): {perf_metrics['avg_time_per_sequence']:.4f} seconds\n")
        log_file.write(f"  Average Time per Frame: {perf_metrics['avg_time_per_frame']:.4f} seconds\n")
        log_file.write(f"  FPS (Frames Per Second): {perf_metrics['fps']:.2f}\n")
        log_file.write("="*70 + "\n")

    print(f"Performance metrics saved to: {log_filename}")

    print("\nDetailed prediction results:")
    for i, cat in enumerate(categories):
        print(f"{cat}: {avg_prediction[i] * 100:.2f}%")

    return category, confidence, perf_metrics


def main():
    import sys
    import time
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Stampede Detection')
    parser.add_argument('--test-only', action='store_true', help='Only test a video with a pre-trained model')
    parser.add_argument('--model-path', type=str, default='enhanced_stampede_detection_checkpoint.h5',
                        help='Path to pre-trained model or checkpoint')
    parser.add_argument('--video-path', type=str,
                        default=r"C:\\MMM\\stampede-detection-main\\Untitled video - Made with Clipchamp (2).mp4", #change this
                        help='Path to test video')
    parser.add_argument('--test-dataset', type=str, default=None,
                        help='Path to dataset folder for test-only mode (expects class subfolders: normal/moderate/dense/risky)')
    parser.add_argument('--use-enhanced', action='store_true', default=True,
                        help='Use the enhanced model with additional features')
    parser.add_argument('--continue-training', action='store_true', default=True,
                        help='Continue training from checkpoint if model exists')
    args = parser.parse_args()

    # If test-only mode, skip training
    if args.test_only:
        print("Running in test-only mode")
        # If a dataset path is provided, run dataset-level evaluation (no pipeline change)
        if args.test_dataset is not None:
            print(f"Running dataset-level test for: {args.test_dataset}")
            test_on_dataset(args.model_path, args.test_dataset)
            return

        # Otherwise fall back to the existing single-video test behavior
        if args.use_enhanced:
            test_enhanced_model(args.model_path, args.video_path)
        else:
            test_video_only(args.model_path, args.video_path)
        return

    # Start time
    start_time = time.time()

    # Set the data path
    data_path = r"C:\MMM\stampede-detection-main\Stampede_detection_dataset"  #change this

    # Load the dataset with enhanced features
    print("Loading optical flow data and calculating additional features...")
    if args.use_enhanced:
        X_flow, X_scalar, y, original_frames, sequence_video_ids = load_optical_flow_data(data_path)
    else:
        print("sequence video ids not loading...")
        X_flow, _, y = load_optical_flow_data(data_path)

    if len(X_flow) == 0:
        print("Error: No data loaded. Please check the data path and directory structure.")
        sys.exit(1)

    # Training metadata file to track epochs
    metadata_file = 'training_metadata.json'
    initial_epoch = 0

    # Load training metadata if continuing training
    if os.path.exists(metadata_file) and args.continue_training:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                initial_epoch = metadata.get('last_epoch', 0)
                print(f"Resuming training from epoch {initial_epoch}")
        except Exception as e:
            print(f"Could not load metadata, starting from epoch 0: {str(e)}")

    # Train the model
    if args.use_enhanced:
        print("\nTraining the enhanced CNN-LSTM model with additional features...")
        '''
        model, history, last_epoch = train_enhanced_model(
            X_flow, X_scalar, y,
            model_path=args.model_path,
            continue_training=args.continue_training,
            initial_epoch=initial_epoch
        )
        '''

        model, history, last_epoch = train_enhanced_model_with_visualizations(
            X_flow, X_scalar, y, original_frames=original_frames,
            sequence_video_ids=sequence_video_ids
        )

        # Save metadata for next run
        with open(metadata_file, 'w') as f:
            json.dump({'last_epoch': last_epoch}, f)
    else:
        print("\nTraining the regular CNN-LSTM model...")
        model, history = train_model(X_flow, y, model_type="cnn_lstm")

    # Test video path
    test_video_path = args.video_path

    # Create temporary directory for optical flow generation
    temp_dir = "temp_optical_flow_test"

    # Make prediction on test video
    print(f"\nPredicting crowd density for video: {test_video_path}")
    if args.use_enhanced:
        category, confidence, perf_metrics = predict_with_enhanced_model(model, test_video_path, temp_dir)
    else:
        category, confidence = predict_optical_flow_video(model, test_video_path, temp_dir)

    # Print results
    print("\n" + "=" * 50)
    print(f"Video Classification Result: {category.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)

    # End time
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # Display feature importance visualization if using enhanced model
    if args.use_enhanced:
        try:
            # Create a simple visualization of sample sequences with feature values
            visualize_feature_importance(X_flow[:5], X_scalar[:5], y[:5])
        except Exception as e:
            print(f"Could not visualize feature importance: {str(e)}")


def visualize_feature_importance(flow_sequences, scalar_features, labels):
    """
    Visualize the additional features to understand their contribution
    """
    categories = ["normal", "moderate", "dense", "risky"]
    feature_names = ["Flow Acceleration", "Flow Divergence", "Scene Changes", "Motion Entropy"]

    plt.figure(figsize=(15, 10))

    # Process each sequence
    for seq_idx in range(min(5, len(flow_sequences))):
        category = categories[labels[seq_idx]]

        # Plot the 4 scalar features over time for this sequence
        plt.subplot(len(flow_sequences), 1, seq_idx + 1)

        for feature_idx in range(4):
            feature_values = scalar_features[seq_idx, :, feature_idx]
            plt.plot(feature_values, label=feature_names[feature_idx])

        plt.title(f"Sequence {seq_idx + 1} (Category: {category})")
        plt.xlabel("Frame")
        plt.ylabel("Feature Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_visualization.png')
    plt.show()


if __name__ == "__main__":
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")

    # Run main function
    main()
