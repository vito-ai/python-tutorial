import numpy as np
from tritonclient.utils import triton_to_np_dtype
from tritony import InferenceClient
from utils.triton_client import TritonClient
from utils.wave_util import closest_frame, load_wave

from .common_fixtures import TRITON_HOST, config

__all__ = ["config"]


def get_client(protocol, port, run_async, model_name):
    return InferenceClient.create_with(
        model=model_name,
        model_version="1",
        url=f"{TRITON_HOST}: {port}",
        protocol=protocol,
        run_async=run_async,
    )


def test_trtiony_client_inference(config):
    client = get_client(*config, model_name="python_vad")
    model_spec = client.model_specs[("python_vad", "1")]
    model_input_name = model_spec.model_input[0].name
    model_input_dtype = model_spec.model_input[0].dtype

    requests = []
    audio_path = "resources/mid_sample.wav"
    np_waveform = load_wave(file_path=audio_path, np_dtype=triton_to_np_dtype(model_input_dtype))
    requests.append(np_waveform)

    response = client({model_input_name: requests})
    st = 2.0
    ed = 5.0
    st_frame = closest_frame(st)
    ed_frame = closest_frame(ed)
    mean_prob = np.mean(response[0][st_frame:ed_frame, :])

    assert mean_prob > 0.5, f"{st_frame} ~ {ed_frame} mean_prob: {mean_prob}"

    st = 6.0
    ed = 7.0
    st_frame = closest_frame(st)
    ed_frame = closest_frame(ed)
    mean_prob = np.mean(response[0][st_frame:ed_frame, :])

    assert mean_prob < 0.2, f"{st_frame} ~ {ed_frame} mean_prob: {mean_prob}"


def test_triton_client_inference(config):
    protocol, _, _ = config
    client = TritonClient(model_name="python_vad", host=TRITON_HOST, protocol=protocol)

    audio_path = "resources/mid_sample.wav"

    response = client.infer(audio_path)
    st = 2.0
    ed = 5.0
    st_frame = closest_frame(st)
    ed_frame = closest_frame(ed)
    mean_prob = np.mean(response[st_frame:ed_frame, :])

    assert mean_prob > 0.5, f"{st_frame} ~ {ed_frame} mean_prob: {mean_prob}"

    st = 6.0
    ed = 7.0
    st_frame = closest_frame(st)
    ed_frame = closest_frame(ed)
    mean_prob = np.mean(response[st_frame:ed_frame, :])

    assert mean_prob < 0.2, f"{st_frame} ~ {ed_frame} mean_prob: {mean_prob}"
