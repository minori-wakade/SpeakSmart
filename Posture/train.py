import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm

print("\n🚀 Loading and preparing data...")

# Load the CSV file
df = pd.read_csv("pose_data.csv")

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels (good -> 1, bad -> 0)
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n⚙️ Training SVM model...")

# Initialize SVM with RBF kernel
svm_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)

# Progress tracker
for _ in tqdm(range(1), desc="Training progress"):
    svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_model.predict(X_test_scaled)

# Accuracy and report
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully!")
print(f"📊 Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
joblib.dump(svm_model, "posture_svm.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n💾 Model, scaler, and encoder saved successfully!")
print("📁 Files created: posture_svm.pkl, scaler.pkl, label_encoder.pkl")
