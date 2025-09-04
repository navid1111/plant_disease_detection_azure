import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import json
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

# --------------------------
# Arguments for flexibility
# --------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="PlantVillage", help="Path to dataset")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--image_size", type=int, default=255, help="Image size")
parser.add_argument("--output_model_path", type=str, default="outputs/plant_model.keras", help="Where to save the trained model")
args = parser.parse_args()


BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size
CHANNELS = 3
EPOCHS = args.epochs

# --------------------------
# Kaggle dataset settings
# --------------------------
DATASET_SLUG = "abdallahalidev/plantvillage-dataset"  # Updated dataset slug

def ensure_kaggle_credentials():
    """Ensure kaggle.json exists either from local file or env vars.
    Exit with clear message if not available."""
    kaggle_home = os.path.expanduser("~/.kaggle")
    cred_path = os.path.join(kaggle_home, "kaggle.json")
    os.makedirs(kaggle_home, exist_ok=True)

    if os.path.exists(cred_path):
        return cred_path

    if os.path.exists("kaggle.json"):
        shutil.copy("kaggle.json", cred_path)
    else:
        ku = os.environ.get("KAGGLE_USERNAME")
        kk = os.environ.get("KAGGLE_KEY")
        if ku and kk:
            with open(cred_path, "w") as f:
                json.dump({"username": ku, "key": kk}, f)
        else:
            print("ERROR: Kaggle credentials not found. Provide kaggle.json or set KAGGLE_USERNAME & KAGGLE_KEY.", file=sys.stderr)
            sys.exit(1)
    # Set restrictive permissions (best effort)
    try:
        os.chmod(cred_path, 0o600)
    except Exception:
        pass
    return cred_path


def download_dataset_if_needed():
    if os.path.exists(args.data_dir) and os.listdir(args.data_dir):
        print(f"Dataset directory '{args.data_dir}' already populated. Skipping download.")
        return
    print(f"Preparing to download PlantVillage dataset into '{args.data_dir}'...")
    ensure_kaggle_credentials()
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Failed to authenticate with Kaggle API: {e}", file=sys.stderr)
        sys.exit(1)
    # Preflight: try listing files to surface license / permission issues early
    try:
        files = api.dataset_list_files(DATASET_SLUG)
        print(f"Dataset '{DATASET_SLUG}' file count: {len(files.files)}")
    except Exception as e:
        print(f"Preflight list failed for dataset '{DATASET_SLUG}': {e}", file=sys.stderr)
        print("If 403: ensure you have accepted the dataset license in the Kaggle UI.", file=sys.stderr)
        sys.exit(1)
    try:
        api.dataset_download_files(DATASET_SLUG, path=args.data_dir, unzip=True)
    except Exception as e:
        print(f"Failed to download dataset: {e}", file=sys.stderr)
        print("Troubleshooting steps:\n"
              " 1. Verify KAGGLE_USERNAME / KAGGLE_KEY (regenerate key if needed).\n"
              " 2. Ensure you accepted the dataset license for this dataset in the Kaggle UI.\n"
              " 3. Check network egress / firewall in Azure ML compute.\n"
              f" 4. Confirm dataset slug '{DATASET_SLUG}' is correct.\n", file=sys.stderr)
        sys.exit(1)


# Trigger dataset download if needed
download_dataset_if_needed()


# --------------------------
# Load dataset
# --------------------------
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_dir,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
n_classes = len(class_names)

# --------------------------
# Dataset splitting
# --------------------------
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, shuffle=True, shuffle_size=1000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    remaining = ds.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, validation_ds, test_ds = get_dataset_partitions_tf(dataset)

# Optimize data pipeline
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# --------------------------
# Data preprocessing & augmentation
# --------------------------
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
])

# --------------------------
# Model architecture
# --------------------------
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# --------------------------
# Train the model
# --------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=validation_ds
)

# --------------------------
# Save the model
# --------------------------

os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
model.save(args.output_model_path)
print(f"Model saved to {args.output_model_path}")

# --------------------------
# Optional: plot accuracy
# --------------------------
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
