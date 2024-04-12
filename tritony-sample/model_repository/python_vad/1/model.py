import json
import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from pyannote.audio import Inference
from pyannote.audio.core.model import Model

HERE = os.path.dirname(os.path.abspath(__file__))


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output_configs = model_config["output"]

        self.duration: float = 3.0
        self.batch_size: int = 2048
        self.samplerate: int = 16000
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = torch.device(f"{device}: {device_id}" if device == "GPU" else device)

        self.output_name_list = [output_config["name"] for output_config in output_configs]
        self.output_dtype_list = [
            pb_utils.triton_string_to_numpy(output_config["data_type"]) for output_config in output_configs
        ]
        # pytorch_model.bin from https://huggingface.co/pyannote/segmentation
        # self.model = Model.from_pretrained(os.path.join(HERE, "pytorch_model.bin"))
        self.model = Model.from_pretrained("pyannote/segmentation")
        self.vad = Inference(
            self.model,
            duration=self.duration,
            batch_size=self.batch_size,
            pre_aggregation_hook=self._pre_hook,
            device=self.device,
        )

    def execute(self, requests):
        """
        Parameters
        ----------
        requests : list
            List of pb_utils.InferenceRequest

        Returns
        -------
        list
            List of pb_utils.InferenceResponse
        """
        waveform_dtype = self.output_dtype_list[0]

        responses = None
        out_tensor = []
        for request in requests:
            raw_audio = pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()

            audio_tensor = torch.from_numpy(raw_audio.reshape(1, -1))

            out = self.vad({"waveform": audio_tensor, "sample_rate": self.samplerate}).data

            out_tensor.append(pb_utils.Tensor("OUTPUT_0", out.astype(waveform_dtype)))

        responses = [pb_utils.InferenceResponse(output_tensors=out_tensor)]

        return responses

    def finalize(self):
        print("Cleaning up...")
        pass

    def _pre_hook(self, score):
        return np.max(score, axis=-1, keepdims=True)
