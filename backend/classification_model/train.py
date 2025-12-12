import os
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# pyright: ignore[reportMissingImports]
# Config — adjust as needed
DATA_DIR = os.path.join(os.getcwd(), "data_prepared")
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 20
NUM_CLASSES = 3
# Prefer the native Keras SavedModel / .keras format (recommended).
# Keep an .h5 fallback for compatibility if needed.
OUT_MODEL = "model_best.keras"
OUT_MODEL_FALLBACK = "model_best.h5"
CLASS_JSON = "class_indices.json"
LEARNING_RATE = 1e-4

# (Fine-tuning / unfreeze options removed per user request)

def get_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123,
        color_mode="rgb"
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        color_mode="rgb"
    )
    # capture class names before we transform the dataset (prefetch returns
    # a tf.data.Dataset wrapper that doesn't expose the `class_names` attr)
    class_names = getattr(train_ds, "class_names", None)
    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def build_model():
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=IMG_SIZE + (3,), weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    return model

def main():
    train_ds, val_ds, class_names = get_datasets()
    # save class indices (class_names comes from the directory loader)
    if class_names is None:
        # fallback: try to get names from the dataset object if available
        class_names = getattr(train_ds, "class_names", None)
    if class_names is None:
        raise RuntimeError(f"Could not determine class names. Ensure '{os.path.join(DATA_DIR,'train')}' contains one subfolder per class with images.")
    class_indices = {name: i for i, name in enumerate(class_names)}
    with open(CLASS_JSON, "w") as f:
        json.dump(class_indices, f)
    print("Class mapping saved to", CLASS_JSON, class_indices)

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        ModelCheckpoint(OUT_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    print("Training finished. Best model saved at:", OUT_MODEL)

    # Fine-tuning step skipped (per request). If you later want to enable
    # unfreezing of the backbone, set the base layers to trainable and recompile
    # before calling fit().

if __name__ == "__main__":
    main()
