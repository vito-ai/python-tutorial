import argparse
import os

import numpy as np
from pydub import AudioSegment
from tritonclient.utils import triton_to_np_dtype
from tritony import InferenceClient


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

    # Create a tritony gRPC client
    client = InferenceClient.create_with(
        model=FLAGS.model_name,
        model_version=FLAGS.model_version,
        url=FLAGS.url,
        protocol="grpc",
        run_async=False,
    )

    # Get model metadata
    model_spec = client.model_specs[(FLAGS.model_name, FLAGS.model_version)]
    model_input_name = model_spec.model_input[0].name
    model_input_dtype = model_spec.model_input[0].dtype

    # Prepare request
    requests = []

    np_waveform = load_wave(file_path=FLAGS.audio_path, np_dtype=triton_to_np_dtype(model_input_dtype))
    requests.append(np_waveform)

    response = client({model_input_name: requests})
    print(f"Output: {response[0].shape}")
