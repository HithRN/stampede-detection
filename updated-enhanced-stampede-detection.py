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

        return X, scalar_features, y

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


def train_enhanced_model(X_flow, X_scalar, y):
    """
    Train the enhanced model with optical flow and additional scalar features
    """
    # Convert labels to one-hot encoding
    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    # Split data into training and validation sets
    X_flow_train, X_flow_val, X_scalar_train, X_scalar_val, y_train, y_val = train_test_split(
        X_flow, X_scalar, y_onehot, test_size=0.2, random_state=42, stratify=y
    )

    # Create enhanced model
    model = create_enhanced_cnn_lstm_model()

    # Print model summary
    model.summary()

    # Setup callbacks
    checkpoint = ModelCheckpoint(
        'enhanced_stampede_detection_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train model
    history = model.fit(
        [X_flow_train, X_scalar_train],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_flow_val, X_scalar_val], y_val),
        callbacks=[checkpoint, early_stopping]
    )

    # Save the final model
    model.save('enhanced_stampede_detection_final.h5')

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('enhanced_stampede_detection_training_history.png')
    plt.show()

    return model, history


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
    Predict the risk level for a video using the enhanced model with timeout
    """

    def test_model_inference(model):
        """Test that model inference is working and measure its speed"""
        print("Testing model inference speed...")

        # Create dummy inputs with the correct shapes
        dummy_flow_input = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 2), dtype=np.float32)
        dummy_scalar_input = np.zeros((1, SEQUENCE_LENGTH, 4), dtype=np.float32)

        # Time the prediction
        import time
        start = time.time()
        dummy_pred = model.predict([dummy_flow_input, dummy_scalar_input], verbose=0)
        end = time.time()

        print(f"Model inference test: {end - start:.3f} seconds for a single sequence")
        print(f"Prediction shape: {dummy_pred.shape}, values: min={dummy_pred.min():.3f}, max={dummy_pred.max():.3f}")
        return end - start

    # Test model inference at the start
    inference_time = test_model_inference(model)
    import time
    start_time = time.time()

    # Time check function
    def check_timeout():
        if time.time() - start_time > timeout_seconds:
            print(f"Processing timed out after {timeout_seconds} seconds!")
            return True
        return False

    # Generate optical flow frames and original frames
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Generating optical flow and features for video: {video_path}")

    flow_frames, original_frames = generate_optical_flow_and_features_from_video(
        video_path,
        temp_dir,
        max_frames=200,  # Reduced from 500 to 200
        frame_skip=5  # Increased from 3 to 5
    )

    if check_timeout(): return "Timeout (processing took too long)", 0.0

    # Need at least SEQUENCE_LENGTH frames to make a prediction
    if len(flow_frames) < SEQUENCE_LENGTH:
        print(f"Warning: Video has only {len(flow_frames)} frames, which is less than required {SEQUENCE_LENGTH}")
        return "Unknown (not enough frames)", 0.0

    print(f"Creating sequences from {len(flow_frames)} frames...")

    # Create sequences for prediction, but limit the total number to avoid memory issues
    flow_sequences = []
    orig_sequences = []
    max_sequences = 20  # Reduced from 50 to 20

    # Take evenly spaced starting points throughout the video
    if len(flow_frames) > SEQUENCE_LENGTH:
        step_size = max(1, (len(flow_frames) - SEQUENCE_LENGTH) // max_sequences)
        start_indices = range(0, len(flow_frames) - SEQUENCE_LENGTH, step_size)

        # Limit to max_sequences
        start_indices = list(start_indices)[:max_sequences]

        for i in start_indices:
            flow_seq = flow_frames[i:i + SEQUENCE_LENGTH]
            orig_seq = original_frames[i:i + SEQUENCE_LENGTH] if i < len(original_frames) else []

            flow_sequences.append(np.array(flow_seq))
            orig_sequences.append(np.array(orig_seq) if orig_seq else None)
            print(f"Created sequence starting at frame {i}/{len(flow_frames)}")

            if check_timeout(): return "Timeout (processing took too long)", 0.0

    if not flow_sequences:
        return "Unknown (could not create sequences)", 0.0

    # Calculate additional features for each sequence
    print("Calculating additional features for prediction...")

    # Flow acceleration
    flow_acceleration = calculate_flow_acceleration(flow_sequences)

    # Flow divergence
    flow_divergence = calculate_flow_divergence(flow_sequences)

    # Scene change detection (if we have original frames)
    scene_changes = calculate_scene_changes(orig_sequences) if all(seq is not None for seq in orig_sequences) else \
        np.zeros((len(flow_sequences), SEQUENCE_LENGTH))

    # Motion entropy
    motion_entropy = calculate_motion_entropy(flow_sequences)

    # Combine all scalar features
    scalar_features = np.stack([
        flow_acceleration,
        flow_divergence,
        scene_changes,
        motion_entropy
    ], axis=2)  # Shape: [num_sequences, sequence_length, 4]

    print(f"Prepared {len(flow_sequences)} sequences with additional features for prediction")

    # Convert sequences to numpy arrays
    X_flow_pred = np.array(flow_sequences)
    X_scalar_pred = scalar_features

    # Make predictions in smaller batches to avoid memory issues
    batch_size = 8
    all_predictions = []

    print(f"Making predictions in batches of {batch_size}...")
    for i in range(0, len(X_flow_pred), batch_size):
        batch_flow = X_flow_pred[i:i + batch_size]
        batch_scalar = X_scalar_pred[i:i + batch_size]
        batch_preds = model.predict([batch_flow, batch_scalar], verbose=0)
        all_predictions.append(batch_preds)
        print(f"Processed batch {i // batch_size + 1}/{(len(X_flow_pred) + batch_size - 1) // batch_size}")

    # Combine all batch predictions
    predictions = np.vstack(all_predictions) if len(all_predictions) > 1 else all_predictions[0]

    # Calculate average prediction across all sequences
    avg_prediction = np.mean(predictions, axis=0)

    # Get class index with highest probability
    class_idx = np.argmax(avg_prediction)

    # Map class index to category
    categories = ["normal", "moderate", "dense", "risky"]
    category = categories[class_idx]

    # Output confidence score
    confidence = avg_prediction[class_idx]

    # Display more detailed results
    print("\nDetailed prediction results:")
    for i, cat in enumerate(categories):
        print(f"{cat}: {avg_prediction[i] * 100:.2f}%")

    return category, confidence


def train_enhanced_model(X_flow, X_scalar, y, model_path='enhanced_stampede_detection_checkpoint.h5',
                         continue_training=True, initial_epoch=0):
    """
    Train the enhanced model with optical flow and additional scalar features
    Can resume training from a saved checkpoint
    """
    # Convert labels to one-hot encoding
    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    # Split data into training and validation sets
    X_flow_train, X_flow_val, X_scalar_train, X_scalar_val, y_train, y_val = train_test_split(
        X_flow, X_scalar, y_onehot, test_size=0.2, random_state=42, stratify=y
    )

    # Check if model file exists and load it if requested
    # Modified model loading code for train_enhanced_model function
    if os.path.exists(model_path) and continue_training:
        print(f"Loading existing model from {model_path}")
        try:
            # Custom objects might be needed if your model has custom layers
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Model loaded successfully. Continuing training...")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Creating new model instead...")
            model = create_enhanced_cnn_lstm_model()
            initial_epoch = 0
    else:
        print("Creating new model...")
        model = create_enhanced_cnn_lstm_model()
        initial_epoch = 0 # Reset initial epoch for new model

    # Print model summary
    model.summary()

    # Setup callbacks
    checkpoint = ModelCheckpoint(
        'enhanced_stampede_detection_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Also save periodic checkpoints for resuming
    periodic_checkpoint = ModelCheckpoint(
        model_path,
        save_freq='epoch',  # Save after each epoch
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train model
    # Modify your model.fit call to be more robust
    history = model.fit(
        [X_flow_train, X_scalar_train],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=([X_flow_val, X_scalar_val], y_val),
        callbacks=[checkpoint, periodic_checkpoint, early_stopping],
        verbose=1  # Provide more information during training
    )

    # Save the final model
    model.save('enhanced_stampede_detection_final.h5')
    print(f"Final model saved to enhanced_stampede_detection_final.h5")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('enhanced_stampede_detection_training_history.png')
    plt.show()

    return model, history, initial_epoch + EPOCHS


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
                        default=r"C:\Users\rnidh\Downloads\Untitled video - Made with Clipchamp (2).mp4",
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
    data_path = r"C:\Users\rnidh\stampede detection\output\UCSD dataset"

    # Load the dataset with enhanced features
    print("Loading optical flow data and calculating additional features...")
    if args.use_enhanced:
        X_flow, X_scalar, y = load_optical_flow_data(data_path)
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
        model, history, last_epoch = train_enhanced_model(
            X_flow, X_scalar, y,
            model_path=args.model_path,
            continue_training=args.continue_training,
            initial_epoch=initial_epoch
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
        category, confidence = predict_with_enhanced_model(model, test_video_path, temp_dir)
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