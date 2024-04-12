import os

import grpc
import numpy as np
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype

from .wave_util import load_wave


class TritonClient:
    def __init__(
        self,
        model_name: str,
        model_version: str = "1",
        host: str = "localhost",
        protocol: str = "grpc",
    ):
        url = f"{host}: 8101" if protocol == "grpc" else f"{host}: 8100"
        self.model_name = model_name
        self.model_version = model_version
        # Create a gRPC client for communicating with the server
        self._channel = grpc.insecure_channel(url)
        self._grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(self._channel)

        # Setting Requests
        self._model_metadata_request = None

    # Get model metadata
    @property
    def metadata(self):
        meta = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inputs": [],
            "outputs": [],
        }
        self._model_metadata_request = (
            service_pb2.ModelMetadataRequest(name=self.model_name, version=self.model_version)
            if self._model_metadata_request is None
            else self._model_metadata_request
        )
        model_metadata = self._grpc_stub.ModelMetadata(self._model_metadata_request)

        for input_meta in model_metadata.inputs:
            meta["inputs"].append({"name": input_meta.name, "dtype": input_meta.datatype})

        for output_meta in model_metadata.outputs:
            meta["outputs"].append({"name": output_meta.name, "dtype": output_meta.datatype})

        return meta

    def infer(self, audio_path: str):
        assert os.path.exists(audio_path), f"File not found: {os.path.abspath(audio_path)}"
        assert audio_path.endswith(".wav"), "Only .wav files are supported"

        # Prepare request
        model_infer_request = service_pb2.ModelInferRequest(
            model_name=self.model_name,
            model_version=self.model_version,
        )

        # Load audio file
        np_waveform = load_wave(audio_path)

        input_tensor = model_infer_request.InferInputTensor(
            name=self.metadata["inputs"][0]["name"],
            datatype=self.metadata["inputs"][0]["dtype"],
            shape=[len(np_waveform)],
        )
        model_infer_request.inputs.extend([input_tensor])

        output_tensor = model_infer_request.InferRequestedOutputTensor(name=self.metadata["outputs"][0]["name"])
        model_infer_request.outputs.extend([output_tensor])

        model_infer_request.raw_input_contents.extend([np_waveform.tobytes()])

        try:
            response = self._grpc_stub.ModelInfer(model_infer_request)
        except Exception as e:
            print(f"Request failed: {e}")
            exit(1)

        response = np.frombuffer(
            response.raw_output_contents[0],
            dtype=triton_to_np_dtype(self.metadata["outputs"][0]["dtype"]),
        )
        response = response.reshape(-1, 1)
        return response
