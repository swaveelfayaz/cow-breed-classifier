import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from pathlib import Path
import numpy as np

print("=== COW BREED CLASSIFIER TRAINING ===")

# -----------------------
# Config
# -----------------------
DATA_DIR = Path("data/processed")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------
# Load datasets
# -----------------------
print("Loading datasets...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"✅ Classes: {class_names}")
print(f"✅ Train: {len(train_ds)*BATCH_SIZE} images, Val: {len(val_ds)*BATCH_SIZE}, Test: {len(test_ds)*BATCH_SIZE}")

# -----------------------
# Data pipeline
# -----------------------
AUTOTUNE = tf.data.AUTOTUNE

# Apply MobileNetV2 preprocessing inside the pipeline
def preprocess_batch(images, labels):
    images = preprocess_input(images)
    return images, labels

train_ds = train_ds.map(preprocess_batch).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.map(preprocess_batch).cache().prefetch(AUTOTUNE)
test_ds  = test_ds.map(preprocess_batch).cache().prefetch(AUTOTUNE)

# Data augmentation (on the fly)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# -----------------------
# Build MobileNetV2 model
# -----------------------
print("\nBuilding MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # first stage: freeze

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# -----------------------
# Stage 1: Base training
# -----------------------
print("\n🚀 Stage 1: Training (10 epochs)...")
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    verbose=1
)

# Save base model
Path("models").mkdir(exist_ok=True)
model.save("models/cow_breed_classifier.h5")
print("✅ Base model saved to models/cow_breed_classifier.h5")

# Evaluate base model
print("\nEvaluating base model on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"🎯 Base Test Accuracy: {test_acc:.2%}")

# -----------------------
# Stage 2: Fine-tuning
# -----------------------
print("\n=== FINE-TUNING STAGE ===")

# Unfreeze last part of base_model
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # unfreeze last 30 layers

for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= fine_tune_at)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\n🚀 Stage 2: Fine-tuning (5 epochs)...")
history_ft = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    verbose=1
)

# Save fine-tuned model
model.save("models/cow_breed_classifier_finetuned.h5")
print("✅ Fine-tuned model saved to models/cow_breed_classifier_finetuned.h5")

# Final evaluation
print("\nEvaluating fine-tuned model on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"🎯 Fine-tuned Test Accuracy: {test_acc:.2%}")
