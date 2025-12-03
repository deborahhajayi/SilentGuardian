import tensorflow as tf

model = tf.keras.models.load_model("models/fall_geom_lstm.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

#  REQUIRED FOR LSTM MODELS
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("models/fall_geom_lstm.tflite", "wb") as f:
    f.write(tflite_model)

print(" TFLite conversion successful!")
