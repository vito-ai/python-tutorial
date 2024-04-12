import numpy as np
from pydub import AudioSegment

BASIC_DURATION = 0.017064846416382253
BASIC_STEP = 0.017064846416382253


def load_wave(file_path, np_dtype=np.float32):
    waveform = AudioSegment.from_wav(file_path)
    waveform = waveform.set_frame_rate(16000).set_channels(1).get_array_of_samples()

    np_waveform = np.array(waveform).astype(np_dtype)
    np_waveform /= np.iinfo(waveform.typecode).max

    return np_waveform


def closest_frame(t: float, duration: float = BASIC_DURATION, step: float = BASIC_STEP):
    return int(np.rint(t - 0.5 * duration) / step)
