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
def evaluate_model_comprehensive(model, X_flow_val, X_scalar_val, y_val_onehot, y_val_labels, class_names=None):
    """
    Comprehensive evaluation pipeline for multi-class classification

    Args:
        model: Trained keras model
        X_flow_val: Validation flow data
        X_scalar_val: Validation scalar features
        y_val_onehot: One-hot encoded validation labels
        y_val_labels: Original integer labels (0, 1, 2, 3)
        class_names: List of class names (default: ["normal", "moderate", "dense", "risky"])
    """
    if class_names is None:
        class_names = ["normal", "moderate", "dense", "risky"]

    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)

    # Get predictions
    print("\nGenerating predictions...")
    y_pred_proba = model.predict([X_flow_val, X_scalar_val], verbose=0)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)

    # 1. PRECISION (Macro + Per Class)
    print("\n" + "-"*70)
    print("PRECISION METRICS")
    print("-"*70)

    precision_macro = precision_score(y_val_labels, y_pred_labels, average='macro')
    precision_per_class = precision_score(y_val_labels, y_pred_labels, average=None)

    print(f"Macro Precision: {precision_macro:.4f}")
    print("\nPer-Class Precision:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {precision_per_class[i]:.4f}")

    # 2. RECALL (Macro + Per Class)
    print("\n" + "-"*70)
    print("RECALL METRICS")
    print("-"*70)

    recall_macro = recall_score(y_val_labels, y_pred_labels, average='macro')
    recall_per_class = recall_score(y_val_labels, y_pred_labels, average=None)

    print(f"Macro Recall: {recall_macro:.4f}")
    print("\nPer-Class Recall:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {recall_per_class[i]:.4f}")

    # 3. F1-SCORE (Macro + Weighted)
    print("\n" + "-"*70)
    print("F1-SCORE METRICS")
    print("-"*70)

    f1_macro = f1_score(y_val_labels, y_pred_labels, average='macro')
    f1_weighted = f1_score(y_val_labels, y_pred_labels, average='weighted')
    f1_per_class = f1_score(y_val_labels, y_pred_labels, average=None)

    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print("\nPer-Class F1-Score:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {f1_per_class[i]:.4f}")

    # 4. CONFUSION MATRIX
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)

    cm = confusion_matrix(y_val_labels, y_pred_labels)
    print(cm)

    # 5. CLASSIFICATION REPORT
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)

    print(classification_report(y_val_labels, y_pred_labels, target_names=class_names))

    # 6. ROC-AUC (One-vs-Rest)
    print("\n" + "-"*70)
    print("ROC-AUC METRICS (One-vs-Rest)")
    print("-"*70)

    # Compute ROC-AUC for each class
    roc_auc_per_class = {}
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_val_onehot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        roc_auc_per_class[class_name] = roc_auc[i]
        print(f"  {class_name}: {roc_auc[i]:.4f}")

    # Macro and Weighted ROC-AUC
    roc_auc_macro = roc_auc_score(y_val_onehot, y_pred_proba, average='macro', multi_class='ovr')
    roc_auc_weighted = roc_auc_score(y_val_onehot, y_pred_proba, average='weighted', multi_class='ovr')

    print(f"\nMacro ROC-AUC: {roc_auc_macro:.4f}")
    print(f"Weighted ROC-AUC: {roc_auc_weighted:.4f}")

    # ============= VISUALIZATIONS =============

    # Plot 1: Confusion Matrix
    print("\nGenerating Confusion Matrix plot...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("  Saved: confusion_matrix.png")
    plt.close()

    # Plot 2: ROC Curves (One plot with all classes)
    print("Generating ROC Curves plot...")
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange']
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.2f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("  Saved: roc_curves.png")
    plt.close()

    # ============= SAVE METRICS TO CSV =============
    print("\nSaving all metrics to CSV...")

    # Create comprehensive metrics dictionary
    metrics_data = {
        'Metric': [],
        'Value': [],
        'Type': []
    }

    # Overall metrics
    metrics_data['Metric'].append('Macro Precision')
    metrics_data['Value'].append(precision_macro)
    metrics_data['Type'].append('Overall')

    metrics_data['Metric'].append('Macro Recall')
    metrics_data['Value'].append(recall_macro)
    metrics_data['Type'].append('Overall')

    metrics_data['Metric'].append('Macro F1-Score')
    metrics_data['Value'].append(f1_macro)
    metrics_data['Type'].append('Overall')

    metrics_data['Metric'].append('Weighted F1-Score')
    metrics_data['Value'].append(f1_weighted)
    metrics_data['Type'].append('Overall')

    metrics_data['Metric'].append('Macro ROC-AUC')
    metrics_data['Value'].append(roc_auc_macro)
    metrics_data['Type'].append('Overall')

    metrics_data['Metric'].append('Weighted ROC-AUC')
    metrics_data['Value'].append(roc_auc_weighted)
    metrics_data['Type'].append('Overall')

    # Per-class metrics
    for i, class_name in enumerate(class_names):
        metrics_data['Metric'].append(f'Precision - {class_name}')
        metrics_data['Value'].append(precision_per_class[i])
        metrics_data['Type'].append('Per-Class')

        metrics_data['Metric'].append(f'Recall - {class_name}')
        metrics_data['Value'].append(recall_per_class[i])
        metrics_data['Type'].append('Per-Class')

        metrics_data['Metric'].append(f'F1-Score - {class_name}')
        metrics_data['Value'].append(f1_per_class[i])
        metrics_data['Type'].append('Per-Class')

        metrics_data['Metric'].append(f'ROC-AUC - {class_name}')
        metrics_data['Value'].append(roc_auc[i])
        metrics_data['Type'].append('Per-Class')

    # Create DataFrame and save
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv('evaluation_metrics.csv', index=False)
    print("  Saved: evaluation_metrics.csv")

    # Also save confusion matrix separately
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv('confusion_matrix.csv')
    print("  Saved: confusion_matrix.csv")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return {
        'precision_macro': precision_macro,
        'precision_per_class': precision_per_class,
        'recall_macro': recall_macro,
        'recall_per_class': recall_per_class,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'roc_auc_per_class': roc_auc_per_class,
        'fpr': fpr,
        'tpr': tpr
    }


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

            # Normalize histogram to get probability distribution
            hist = hist / (np.sum(hist) if np.sum(hist) > 0 else 1)

            # Calculate entropy
            flow_entropy = entropy(hist, base=2)
            seq_entropy.append(flow_entropy)

        entropies.append(np.array(seq_entropy))

    return np.array(entropies)


def load_optical_flow_data(base_dir, categories=["normal", "moderate", "dense", "risky"]):
    """
    Load the optical flow data from directory structure with JPG images directly in category folders
    """
    X = []  # Will contain sequences of optical flow
    original_frames = []  # Will contain original frames for SSIM calculation
    y = []  # Will contain labels

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
            # If we have JPG files directly in the category folder
            print(f"Found {len(frames)} frames in {category} folder")

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
        else:
            # Check for subdirectories
            subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
            print(f"Found {len(subfolders)} subfolders in {category} category")

            for subfolder in subfolders:
                subfolder_path = os.path.join(category_path, subfolder)
                frames = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.tif'))])
                print(f"Found {len(frames)} frames in subfolder {subfolder}")

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

        print(f"Combined scalar features shape: {scalar_features.shape}")

        return X, scalar_features, y, original_frames

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
        "loss_function": str(model.loss),
        "metrics": [m if isinstance(m, str) else m.__name__ for m in model.compiled_metrics._metrics]
    }

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

    Args:
        original_frame: Original video frame
        prediction_probs: Array of prediction probabilities [prob_class0, prob_class1, ...]
        true_label: True class label (optional)
        save_path: Path to save the visualization
        class_names: List of class names

    Returns:
        overlay_frame: Frame with prediction overlay
    """
    if class_names is None:
        class_names = ["normal", "moderate", "dense", "risky"]

    # Create a copy of the frame
    overlay_frame = original_frame.copy()
    h, w = overlay_frame.shape[:2]

    # Get predicted class
    pred_class = np.argmax(prediction_probs)
    pred_confidence = prediction_probs[pred_class]

    # Create semi-transparent overlay for text background
    overlay = overlay_frame.copy()

    # Define colors for each class (BGR format)
    class_colors = {
        0: (0, 255, 0),    # Normal - Green
        1: (0, 255, 255),  # Moderate - Yellow
        2: (0, 165, 255),  # Dense - Orange
        3: (0, 0, 255)     # Risky - Red
    }

    # Draw prediction box
    box_height = 150
    box_y_start = h - box_height - 10
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
        # Class label
        cv2.putText(overlay_frame, class_name[:4].upper(), (20, bar_y + i*25 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Probability bar background
        cv2.rectangle(overlay_frame, (100, bar_y + i*25),
                      (100 + bar_width, bar_y + i*25 + bar_height), (50, 50, 50), -1)

        # Probability bar fill
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


# Modified train_enhanced_model to include visualization generation
def train_enhanced_model_with_visualizations(X_flow, X_scalar, y, original_frames=None):
    """
    Enhanced training function that generates visualizations after training

    This wraps the existing train_enhanced_model function and adds visualization generation
    """
    import csv
    import os
    import json
    import datetime

    initial_epoch = 0

    # Convert labels to one-hot encoding
    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    # Split data into training and validation sets
    X_flow_train, X_flow_val, X_scalar_train, X_scalar_val, y_train, y_val_onehot, y_train_labels, y_val_labels = train_test_split(
        X_flow, X_scalar, y_onehot, y, test_size=0.2, random_state=42, stratify=y
    )

    # Also split original frames if provided
    if original_frames is not None and len(original_frames) > 0:
        _, original_frames_val = train_test_split(
            original_frames, test_size=0.2, random_state=42, stratify=y
        )
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
        model, X_flow_val, X_scalar_val, y_val_onehot, y_val_labels,
        class_names=["normal", "moderate", "dense", "risky"]
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
                        default=r"C:\Users\rnidh\Downloads\Untitled video - Made with Clipchamp (2).mp4", #change this
                        help='Path to test video')
    parser.add_argument('--use-enhanced', action='store_true', default=True,
                        help='Use the enhanced model with additional features')
    parser.add_argument('--continue-training', action='store_true', default=True,
                        help='Continue training from checkpoint if model exists')
    args = parser.parse_args()

    # If test-only mode, skip training
    if args.test_only:
        print("Running in test-only mode")
        if args.use_enhanced:
            test_enhanced_model(args.model_path, args.video_path)
        else:
            test_video_only(args.model_path, args.video_path)
        return

    # Start time
    start_time = time.time()

    # Set the data path
    data_path = r"C:\Users\rnidh\stampedeDetection\output\Stampede_detection_dataset"  #change this

    # Load the dataset with enhanced features
    print("Loading optical flow data and calculating additional features...")
    if args.use_enhanced:
        X_flow, X_scalar, y, original_frames = load_optical_flow_data(data_path)
    else:
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
            X_flow, X_scalar, y, original_frames=original_frames
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