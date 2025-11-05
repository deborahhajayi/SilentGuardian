# src/train_model.py
import os
import math
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# ---------- CONFIG ----------
DATA_ROOT = "data"                # expects data/train, data/val, data/test
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

MODEL_DIR = "models"
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE = (128, 128)             # must match preprocessing
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 30
SEED = 42
# ----------------------------

def build_simple_cnn(input_shape=(128,128,3)):
    """A reasonable starter CNN (can be swapped with MobileNetV2 later)."""
    inp = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inp)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name="simple_cnn")
    return model

def make_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='binary',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='binary',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='binary',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def plot_history(history, out_dir=PLOTS_DIR):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history.get('val_loss',[]), label='val loss')
    plt.legend(); plt.title('Loss'); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(history.history.get('accuracy',[]), label='train acc')
    plt.plot(history.history.get('val_accuracy',[]), label='val acc')
    plt.legend(); plt.title('Accuracy'); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy.png"))
    plt.close()

def evaluate_and_report(model, test_ds, out_dir=MODEL_DIR):
    # collect all true labels and predictions
    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        preds = (probs >= 0.5).astype(int)
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # classification metrics
    report = classification_report(y_true, y_pred, target_names=['fall','no_fall'], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # save report
    report_path = os.path.join(out_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix\n")
        f.write(str(cm) + "\n")

    print("=== EVALUATION ===")
    print(report)
    print("Confusion matrix:\n", cm)

    # plot confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['fall','no_fall'])
    plt.yticks(tick_marks, ['fall','no_fall'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

def main():
    print("TensorFlow version:", tf.__version__)
    print("Preparing datasets...")
    train_ds, val_ds, test_ds = make_datasets()

    # quick dataset sizes
    def count_batches(ds):
        n = 0
        for _ in ds:
            n += 1
        return n
    print("Train batches:", count_batches(train_ds), "Validation batches:", count_batches(val_ds), "Test batches:", count_batches(test_ds))

    model = build_simple_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    best_model_path = os.path.join(MODEL_DIR, f"model_best_{timestamp}.h5")
    saved_model_dir = os.path.join(MODEL_DIR, "saved_model")

    cb_list = [
        callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss'),
                callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_list
    )

    # Save final model (also keep best_model saved by checkpoint)
    final_h5 = os.path.join(MODEL_DIR, "model_final.h5")
    print("Saving final model to", final_h5)
    model.save(final_h5)
    # Also save SavedModel for later TFLite conversion
    model.save(saved_model_dir, save_format="tf")

    # Plot history
    plot_history(history, out_dir=PLOTS_DIR)

    # Evaluate on test set and write report
    evaluate_and_report(model, test_ds, out_dir=MODEL_DIR)
    print("Done. Models and plots saved under:", MODEL_DIR)

if __name__ == "__main__":
    main()
