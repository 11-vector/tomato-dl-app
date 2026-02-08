import typing as tp
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
import time


class InferenceResult(tp.TypedDict):
    outputs: list[tp.Any]
    elapsed_time: float
    labelled: dict[str, float]


class TfliteInference:
    _interpreter: tp.Optional[Interpreter]

    def __init__(self, model_path: str, labels: list[str] | None = None):
        self.model_path = model_path
        self._interpreter = None
        self.labels = labels

    def load_model(self):
        self._interpreter = tf.lite.Interpreter(
            model_path=self.model_path,)
        self._interpreter.allocate_tensors()
        self._output_details = self._interpreter.get_output_details()
        self._input_details = self._interpreter.get_input_details()

    def inference(self, inputs: list[tp.Any]) -> InferenceResult:
        if self._interpreter is None:
            self.load_model()

        for idx, input_ in enumerate(inputs):
            self._interpreter.set_tensor(
                self._input_details[idx]['index'], input_)
        start_time = time.perf_counter_ns()
        self._interpreter.invoke()
        elapsed_time = time.perf_counter_ns() - start_time
        outputs = list(map(lambda output_detail: self._interpreter.get_tensor(
            output_detail['index']), self._output_details))
        labelled = None
        outputs = outputs[0][0].tolist()
        # print(f"{outputs}")
        if self.labels:
            labelled = dict(zip(self.labels, outputs))
        # print(f"{labelled}")
        result = InferenceResult(
            outputs=outputs, elapsed_time=elapsed_time, labelled=labelled)
        return result

    @classmethod
    def convert_to_tflite(cls, model: tf.keras.Model, model_path: str):
        # tensorflow lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.experimental_new_converter = True
        tflite_model_raw = converter.convert()
        # Save the model.
        with open(model_path, 'wb') as f:
            f.write(tflite_model_raw)
        return cls(model_path=model_path)
