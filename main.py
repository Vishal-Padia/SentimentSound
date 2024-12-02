import os
import torch
import wandb
import librosa
import torchaudio

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.utils import class_weight
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold


# Advanced Configuration with More Options
class Config:
    """Enhanced configuration for emotion recognition project"""

    # Data paths
    DATA_DIR = "archive"

    # Audio processing parameters
    SAMPLE_RATE = 22050  # Standard sample rate
    DURATION = 3  # seconds
    N_MFCC = 20

    # Model hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20

    # Feature extraction parameters
    FEATURES = [
        "mfcc",
        "spectral_centroid",
        "chroma",
        "spectral_contrast",
        "zero_crossing_rate",
        "spectral_rolloff",
    ]

    # Augmentation parameters
    AUGMENTATION = True
    NOISE_FACTOR = 0.005
    SCALE_RANGE = (0.9, 1.1)


def extract_advanced_features(file_path):
    """
    Extract multiple audio features with more comprehensive approach

    Args:
        file_path (str): Path to the audio file

    Returns:
        numpy.ndarray: Concatenated feature vector
    """
    # Load the audio file
    y, sr = librosa.load(file_path, duration=Config.DURATION, sr=Config.SAMPLE_RATE)

    # Feature extraction
    features = []

    # MFCC features (increased resolution)
    if "mfcc" in Config.FEATURES:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.N_MFCC)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        features.append(mfccs_processed)

    # Spectral Centroid
    if "spectral_centroid" in Config.FEATURES:
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroids_processed = np.mean(spectral_centroids)
        features.append([spectral_centroids_processed])

    # Chroma Features
    if "chroma" in Config.FEATURES:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_processed = np.mean(chroma.T, axis=0)
        features.append(chroma_processed)

    # Spectral Contrast
    if "spectral_contrast" in Config.FEATURES:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_processed = np.mean(spectral_contrast.T, axis=0)
        features.append(spectral_contrast_processed)

    # Zero Crossing Rate
    if "zero_crossing_rate" in Config.FEATURES:
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_processed = np.mean(zcr)
        features.append([zcr_processed])

    # Spectral Rolloff
    if "spectral_rolloff" in Config.FEATURES:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_processed = np.mean(spectral_rolloff)
        features.append([spectral_rolloff_processed])

    # Concatenate all features
    return np.concatenate(features)


def augment_features(
    features, noise_factor=Config.NOISE_FACTOR, scale_range=Config.SCALE_RANGE
):
    """
    Advanced feature augmentation technique

    Args:
        features (numpy.ndarray): Input feature array
        noise_factor (float): Magnitude of noise to add
        scale_range (tuple): Range for feature scaling

    Returns:
        numpy.ndarray: Augmented features
    """
    if not Config.AUGMENTATION:
        return features

    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, features.shape)
    augmented_features = features + noise

    # Random scaling
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    augmented_features *= scale_factor

    return augmented_features


def prepare_dataset(data_dir):
    """
    Prepare dataset with more robust feature extraction and potential augmentation

    Args:
        data_dir (str): Root directory containing actor subdirectories

    Returns:
        tuple: Features and labels
    """
    features = []
    labels = []

    # Emotion mapping with potential for expansion
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".wav"):
                # Full file path
                file_path = os.path.join(root, filename)

                try:
                    # Extract emotion from filename
                    emotion_code = filename.split("-")[2]
                    emotion = emotion_map.get(emotion_code, "unknown")

                    # Extract original features
                    file_features = extract_advanced_features(file_path)
                    features.append(file_features)
                    labels.append(emotion)

                    # Optional augmentation
                    if Config.AUGMENTATION:
                        augmented_features = augment_features(file_features)
                        features.append(augmented_features)
                        labels.append(emotion)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # Informative print about dataset
    print(f"Dataset Summary:")
    print(f"Total files processed: {len(features)}")

    # Count of emotions
    from collections import Counter

    emotion_counts = Counter(labels)
    for emotion, count in emotion_counts.items():
        print(f"{emotion.capitalize()} emotion: {count} samples")

    return np.array(features), np.array(labels)


class EmotionDataset(Dataset):
    """Enhanced Custom PyTorch Dataset for Emotion Recognition"""

    def __init__(self, features, labels, scaler=None):
        # Standardize features
        if scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = scaler.transform(features)

        self.features = torch.FloatTensor(features)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = torch.LongTensor(self.label_encoder.fit_transform(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_num_classes(self):
        return len(self.label_encoder.classes_)

    def get_class_names(self):
        return self.label_encoder.classes_


class HybridEmotionRecognitionModel(nn.Module):
    """Advanced Hybrid Neural Network for Emotion Recognition"""

    def __init__(self, input_dim, num_classes):
        super().__init__()

        # Enhanced input projection with residual connection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # More complex convolutional layers with residual connections
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ),
            ]
        )

        # Bidirectional LSTM with more layers
        self.lstm_layers = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )

        # More complex fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),  # Note the 512 due to bidirectional LSTM
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)

        # Reshape for conv layers
        x = x.unsqueeze(1)

        # Convolutional layers with residual-like processing
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Prepare for LSTM
        x = x.permute(0, 2, 1)

        # LSTM processing
        lstm_out, _ = self.lstm_layers(x)
        x = lstm_out[:, -1, :]

        # Fully connected layers
        x = self.fc_layers(x)

        return self.output_layer(x)


def train_model(model, train_loader, val_loader, labels, num_epochs=Config.NUM_EPOCHS):
    """
    Advanced training function with improved techniques

    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        labels (numpy.ndarray): Original labels for class weight computation
        num_epochs (int): Number of training epochs
    """
    # Compute class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.FloatTensor(class_weights)

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Initialize wandb
    wandb.init(
        project="SentimentSound",
        config={
            "learning_rate": Config.LEARNING_RATE,
            "batch_size": Config.BATCH_SIZE,
            "epochs": num_epochs,
            "augmentation": Config.AUGMENTATION,
        },
    )

    # Training loop with more advanced techniques
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, batch_labels in train_loader:
            optimizer.zero_grad()

            # Forward and backward pass
            outputs = model(features)
            loss = criterion(outputs, batch_labels)

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, batch_labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        # Compute metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging to wandb
        wandb.log(
            {
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": train_accuracy,
                "val_loss": val_loss / len(val_loader),
                "val_accuracy": val_accuracy,
            }
        )

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_emotion_model.pth")

    # Finish wandb run
    wandb.finish()

    return model


def evaluate_model(model, test_loader, dataset):
    """
    Evaluate the model and generate detailed metrics

    Args:
        model (nn.Module): Trained PyTorch model
        test_loader (DataLoader): Test data loader
        dataset (EmotionDataset): Dataset for class names
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Classification Report
    class_names = dataset.get_class_names()
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Data Preparation
    features, labels = prepare_dataset(Config.DATA_DIR)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)
    test_dataset = EmotionDataset(X_test, y_test)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    # Model Initialization
    model = HybridEmotionRecognitionModel(
        input_dim=len(X_train[0]), num_classes=train_dataset.get_num_classes()
    )

    # Train Model
    train_model(
        model,
        train_loader,
        val_loader,
        labels,
        num_epochs=Config.NUM_EPOCHS,
    )

    # Evaluate Model
    evaluate_model(model, test_loader, train_dataset)


if __name__ == "__main__":
    main()
