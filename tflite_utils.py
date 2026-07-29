"""
tflite_utils.py
---------------
Small helper used by quantize.py, benchmark_pi.py and deploy_pi.py.

It hides the difference between a float TFLite model (input float32,
values 0-255) and a fully INT8-quantised model (input uint8, output
requiring dequantisation), so the calling code stays identical.

The TensorFlow Lite runtime is imported in whichever form is available:
the light `tflite_runtime` package on the Raspberry Pi, or full
TensorFlow on the training machine.
"""

import numpy as np

try:                                      # Raspberry Pi (light runtime)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:                                  # newer standalone runtime
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:                    # full TensorFlow (laptop / Colab)
        # note: `from tensorflow.lite import Interpreter` fails on recent
        # TensorFlow versions because tf.lite is a lazily loaded module,
        # so the attribute must be accessed through the tf namespace.
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


class TFLiteModel:
    def __init__(self, model_path, num_threads=4):
        self.interpreter = Interpreter(model_path=model_path,
                                       num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.input_dtype = self.inp["dtype"]
        self.in_scale, self.in_zero = self.inp["quantization"]
        self.out_scale, self.out_zero = self.out["quantization"]

    @property
    def input_size(self):
        return int(self.inp["shape"][1]), int(self.inp["shape"][2])

    def predict(self, image):
        """image: HxWx3 array with values in 0-255 (uint8 or float).
        Returns a 1-D float array of class probabilities."""
        x = np.asarray(image)
        if x.ndim == 3:
            x = np.expand_dims(x, 0)

        if self.input_dtype == np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        elif self.input_dtype == np.int8:
            # scale 0-255 into the int8 quantised range
            x = x.astype(np.float32)
            if self.in_scale:
                x = x / self.in_scale + self.in_zero
            x = np.clip(x, -128, 127).astype(np.int8)
        else:
            x = x.astype(np.float32)

        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])[0]

        if self.out["dtype"] in (np.uint8, np.int8) and self.out_scale:
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale
        return np.asarray(y, dtype=np.float32)
