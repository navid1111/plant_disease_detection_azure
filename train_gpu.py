import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import argparse
import json, shutil, sys
from kaggle.api.kaggle_api_extended import KaggleApi

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_slug', default='emmarex/plantdisease')
parser.add_argument('--data_dir', default='PlantVillage')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

def ensure_kaggle():
    kd = os.path.expanduser('~/.kaggle')
    os.makedirs(kd, exist_ok=True)
    target = os.path.join(kd, 'kaggle.json')
    if not os.path.exists(target):
        ku = os.getenv('KAGGLE_USERNAME')
        kk = os.getenv('KAGGLE_KEY')
        if not (ku and kk):
            print('Missing Kaggle creds', file=sys.stderr)
            sys.exit(1)
        with open(target, 'w') as f: json.dump({'username': ku, 'key': kk}, f)
        try: os.chmod(target, 0o600)
        except: pass

def download():
    if os.path.exists(args.data_dir) and os.listdir(args.data_dir):
        return
    ensure_kaggle()
    api = KaggleApi(); api.authenticate()
    files = api.dataset_list_files(args.dataset_slug)
    print(f"Dataset '{args.dataset_slug}' files: {len(files.files)}")
    api.dataset_download_files(args.dataset_slug, path=args.data_dir, unzip=True)

download()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_dir, validation_split=0.2, subset='training', seed=123,
    image_size=(args.image_size, args.image_size), batch_size=args.batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_dir, validation_split=0.2, subset='validation', seed=123,
    image_size=(args.image_size, args.image_size), batch_size=args.batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(512).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(args.image_size, args.image_size, 3)),
    layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
os.makedirs('outputs', exist_ok=True)
model.save('outputs/plant_model.keras')
print('Saved model to outputs/plant_model.keras')