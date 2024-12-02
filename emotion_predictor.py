import os
import torch
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from main import Config, HybridEmotionRecognitionModel, extract_advanced_features


class EmotionPredictor:
    def __init__(self, model_path="best_emotion_model.pth"):
        """
        Initialize the emotion predictor

        Args:
            model_path (str): Path to the saved model weights
        """
        # Prepare feature extraction specifics
        self.features = Config.FEATURES

        # Emotion mapping (same as in original script)
        self.emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised",
        }

        # Load the model
        # First, prepare a dummy dataset to get the input dimension and number of classes
        dummy_features, dummy_labels = self._prepare_dummy_dataset()

        # Initialize the model
        self.model = HybridEmotionRecognitionModel(
            input_dim=len(dummy_features[0]), num_classes=len(np.unique(dummy_labels))
        )

        # Load the saved weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode

        # Prepare label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(dummy_labels)

        # Prepare scaler
        self.scaler = StandardScaler()
        self.scaler.fit(dummy_features)

    def _prepare_dummy_dataset(self):
        """
        Prepare a dummy dataset similar to the original preparation method

        Returns:
            tuple: Features and labels
        """
        features = []
        labels = []

        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(Config.DATA_DIR):
            for filename in files:
                if filename.endswith(".wav"):
                    # Full file path
                    file_path = os.path.join(root, filename)

                    try:
                        # Extract emotion from filename
                        emotion_code = filename.split("-")[2]
                        emotion = self.emotion_map.get(emotion_code, "unknown")

                        # Extract features
                        file_features = extract_advanced_features(file_path)
                        features.append(file_features)
                        labels.append(emotion)

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

                    # Limit to a small number of files for efficiency
                    if len(features) >= 100:
                        break

                if len(features) >= 100:
                    break

            if len(features) >= 100:
                break

        return np.array(features), np.array(labels)

    def predict_emotion(self, audio_file_path):
        """
        Predict emotion for a given audio file

        Args:
            audio_file_path (str): Path to the audio file

        Returns:
            str: Predicted emotion
        """
        # Extract features
        try:
            features = extract_advanced_features(audio_file_path)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return "Unknown"

        # Standardize features
        features = self.scaler.transform(features.reshape(1, -1))

        # Convert to tensor
        features_tensor = torch.FloatTensor(features)

        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label_index = predicted.numpy()[0]

        # Convert numeric label to emotion string
        return self.label_encoder.classes_[predicted_label_index]


def main():
    # Initialize predictor
    predictor = EmotionPredictor()

    # Example usage
    print("Emotion Prediction Script")
    print("------------------------")

    # Prompt user to input audio file path
    while True:
        audio_path = input("Enter the path to an audio file (or 'q' to quit): ").strip()

        if audio_path.lower() == "q":
            break

        if not os.path.exists(audio_path):
            print("File does not exist. Please check the path.")
            continue

        try:
            # Predict emotion
            emotion = predictor.predict_emotion(audio_path)
            print(f"Predicted Emotion: {emotion}")

        except Exception as e:
            print(f"Error predicting emotion: {e}")


if __name__ == "__main__":
    main()
