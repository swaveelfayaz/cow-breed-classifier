import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Paths and config
DATA_DIR = Path("data/processed")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("=== EVALUATING COW BREED CLASSIFIER ===")

# 1) Load test dataset (without shuffling)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = test_ds.class_names
print("Classes:", class_names)

# 2) Load fine-tuned model (fallback to base)
model_path_ft = Path("models/cow_breed_classifier_finetuned.h5")
model_path_base = Path("models/cow_breed_classifier.h5")

if model_path_ft.exists():
    print("Loading fine-tuned model...")
    model = tf.keras.models.load_model(model_path_ft)
else:
    print("Fine-tuned model not found, loading base model...")
    model = tf.keras.models.load_model(model_path_base)

# 3) Get predictions and true labels
y_true = []
y_pred = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 4) Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Cow Breeds")
plt.tight_layout()

# Ensure output folder
Path("notebooks").mkdir(exist_ok=True)
cm_path = Path("notebooks/confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"✅ Saved confusion matrix to {cm_path}")

# 5) Classification report
print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
