import tensorflow as tf
import numpy as np

MODELS = {}

def load_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, input_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_array.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

@app.on_event("startup")
def load_models():
    print("[INFO] Loading TFLite models...")
    MODELS["plant"] = load_tflite("model/plant_disease.tflite")
    MODELS["rice"] = load_tflite("model/rice_disease.tflite")
    print("[OK] Models loaded")
