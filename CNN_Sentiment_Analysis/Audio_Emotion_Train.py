import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import itertools

# Configuration
DATA_PATH = "Audio_Dataset"  # Update this path to your dataset location
SR = 22050                    # Sampling rate
N_MFCC = 40                   # Number of MFCC coefficients
MAX_PAD_LEN = 174             # Number of time frames to pad/truncate to
BATCH_SIZE = 32
EPOCHS = 40
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_OUT = "emotion_recognition_model.h5"

# Emotion mapping (customize based on your dataset)
# You may need to adjust this based on how emotions are labeled in your dataset
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_emotion_from_path(filepath):
    """
    Extract emotion label from file path.
    Modify this function based on how your dataset is organized.
    """
    # Example: if emotions are in folder names
    folder_name = os.path.basename(os.path.dirname(filepath))
    if folder_name in EMOTIONS.values():
        return folder_name
    
    # Example: if emotions are encoded in filenames
    filename = os.path.basename(filepath)
    parts = filename.split('-')
    if len(parts) >= 3 and parts[2] in EMOTIONS:
        return EMOTIONS[parts[2]]
    
    # If no pattern matches, return None (file will be skipped)
    return None

def pad_or_truncate(mfcc, max_len=MAX_PAD_LEN):
    """Pad or truncate MFCC features to a fixed length."""
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_features(file_path, sr=SR, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """Extract MFCC features from audio file."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Compute deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack MFCC + delta + delta2
    stacked = np.vstack([mfcc, delta, delta2])
    
    # Pad/truncate to fixed length
    stacked = pad_or_truncate(stacked, max_pad_len)
    return stacked

def load_dataset(data_path):
    """Load and process the audio dataset."""
    pattern = os.path.join(data_path, '**', '*.wav')
    files = glob.glob(pattern, recursive=True)
    
    # If no WAV files found, try other common audio formats
    if len(files) == 0:
        for ext in ['*.mp3', '*.flac', '*.m4a']:
            pattern = os.path.join(data_path, '**', ext)
            files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(files)} audio files.")
    
    X, Y = [], []
    for f in files:
        emo = extract_emotion_from_path(f)
        if emo is None:
            print(f"Skipping {f}: could not extract emotion label")
            continue
            
        feat = extract_features(f)
        if feat is None:
            continue
            
        X.append(feat)
        Y.append(emo)
    
    X = np.array(X)
    Y = np.array(Y)
    print(f"X shape (num_samples, features, time): {X.shape}")
    print(f"Emotions distribution: {pd.Series(Y).value_counts().to_dict()}")
    
    return X, Y

def build_cnn_model(input_shape, num_classes):
    """Build a CNN model for emotion recognition."""
    inp = layers.Input(shape=(input_shape[0], input_shape[1], 1))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """Plot confusion matrix."""
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
        plt.text(j, i, val,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def predict_emotion(model, file_path, le):
    """Predict emotion for a single audio file."""
    features = extract_features(file_path)
    if features is None:
        return "Error: Could not extract features from audio file"
    
    features = features[np.newaxis, ..., np.newaxis]
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)
    emotion_label = le.inverse_transform(predicted_class)[0]
    confidence = np.max(prediction)
    
    return emotion_label, confidence

def main():
    """Main function to train the model and save it."""
    # Load and process dataset
    print("Loading and processing dataset...")
    X, Y = load_dataset(DATA_PATH)
    
    if len(X) == 0:
        print("No valid audio files found. Please check your DATA_PATH and file formats.")
        return
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    # Reshape for CNN
    X = X[..., np.newaxis]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Build model
    input_shape = (X.shape[1], X.shape[2])
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_loss')
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )
    
    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
    
    # Save label encoder for future use
    np.save('label_encoder_classes.npy', le.classes_)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    print("\nClassification report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plot_confusion_matrix(cm, classes=le.classes_, normalize=False, title='Confusion matrix (counts)')
    plot_confusion_matrix(cm, classes=le.classes_, normalize=True, title='Confusion matrix (normalized)')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.show()
    
    print(f"Model saved as {MODEL_OUT}")
    print("Label encoder classes saved as label_encoder_classes.npy")
    
    return model, le

if __name__ == "__main__":
    # Train the model
    model, le = main()
    
    # Example: Predict emotion for a new audio file
    # test_file = "path/to/your/test_audio.wav"
    # emotion, confidence = predict_emotion(model, test_file, le)
    # print(f"Predicted emotion: {emotion} (confidence: {confidence:.2f})")
