import tensorflow as tf

def convert(h5_path, out_path):
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {out_path}")

convert("model/plant_disease.h5", "plant_disease.tflite")
convert("model/rice_disease.h5", "rice_disease.tflite")
