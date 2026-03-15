#!/usr/bin/env python3

import tensorflow as tf

MODEL_IN = "models/fall_geom_lstm.keras"
MODEL_OUT = "models/fall_geom_lstm.tflite"

SEQ_LEN = 8
FEATURES = 13

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_IN, compile=False)

# Force static input shape (critical for LSTM conversion)
model.build(input_shape=(1, SEQ_LEN, FEATURES))
model.summary()

print("Creating TFLite converter...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Required for LSTM support
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Required for TensorList ops used by LSTM
converter._experimental_lower_tensor_list_ops = False

# Optional optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model...")

tflite_model = converter.convert()

with open(MODEL_OUT, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite conversion successful!")
print("Saved to:", MODEL_OUT)