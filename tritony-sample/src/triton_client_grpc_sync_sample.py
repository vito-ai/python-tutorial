import argparse
import os

import grpc
import numpy as np
from pydub import AudioSegment
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype


def load_wave(file_path, np_dtype=np.float32):
    waveform = AudioSegment.from_wav(file_path)
    waveform = waveform.set_frame_rate(16000).set_channels(1).get_array_of_samples()

    np_waveform = np.array(waveform).astype(np_dtype)
    np_waveform /= np.iinfo(waveform.typecode).max

    return np_waveform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="0.0.0.0:8101",
        help="Inference Server URL",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="python_vad",
        help="Model Name",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="1",
        help="Model Version",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "short_sample.wav"),
    )
    FLAGS = parser.parse_args()

    # Create a gRPC client for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Get model metadata
    model_metadata_request = service_pb2.ModelMetadataRequest(name=FLAGS.model_name, version=FLAGS.model_version)
    model_metadata = grpc_stub.ModelMetadata(model_metadata_request)
    input_name = model_metadata.inputs[0].name
    output_name = model_metadata.outputs[0].name
    input_dtype = model_metadata.inputs[0].datatype
    output_dtype = model_metadata.outputs[0].datatype

    # Prepare request
    request = service_pb2.ModelInferRequest(model_name=FLAGS.model_name, model_version=FLAGS.model_version)

    # Load Wave file to numpy array
    np_waveform = load_wave(FLAGS.audio_path, triton_to_np_dtype(input_dtype))  # Covert audio to numpy array

    # Prepare input
    input_tensor = service_pb2.ModelInferRequest().InferInputTensor(
        name=input_name, datatype=input_dtype, shape=[len(np_waveform)]
    )
    # Set input data
    request.inputs.extend([input_tensor])

    # Prepare output
    output_tensor = service_pb2.ModelInferRequest().InferRequestedOutputTensor(name=output_name)
    # Set output data
    request.outputs.extend([output_tensor])

    # Set raw input data
    request.raw_input_contents.extend([np_waveform.tobytes()])  # Convert numpy array to bytes

    # Send request
    try:
        response = grpc_stub.ModelInfer(request)
    except Exception as e:
        print(f"Request failed: {e}")
        exit(1)

    # Get output
    response = np.frombuffer(
        response.raw_output_contents[0], dtype=triton_to_np_dtype(output_dtype)
    )  # Convert bytes to numpy array
    print(f"Output: {response.shape}\n")
