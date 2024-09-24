"""

To Download definition (.proto) file

$ wget https://raw.github.com/vito-ai/openapi-grpc/main/protos/vito-stt-client.proto

To generate gRPC code

$ pip install grpcio-tools
$ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./vito-stt-client.proto

NOTE: This module requires the dependencies `grpcio`, `requests` `soundfile`.
To install using pip:
    pip install grpcio
    pip install requests
    pip install soundfile

Example usage:
    python vitoopenapi-stt-streaming-sample.py sample/filepath

"""

import argparse
import configparser
import logging
import os
import time
from io import DEFAULT_BUFFER_SIZE

import grpc
import soundfile as sf
import vito_stt_client_pb2 as pb
import vito_stt_client_pb2_grpc as pb_grpc
from requests import Session

API_BASE = "https://openapi.vito.ai"
GRPC_SERVER_URL = "grpc-openapi.vito.ai:443"

SAMPLE_RATE = 8000
ENCODING = pb.DecoderConfig.AudioEncoding.LINEAR16
BYTES_PER_SAMPLE = 2


# 본 예제에서는 스트리밍 입력을 음성파일을 읽어서 시뮬레이션 합니다.
# 실제사용시에는 마이크 입력 등의 실시간 음성 스트림이 들어와야합니다.
class FileStreamer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None

    def __enter__(self):
        self.file = open(self.filepath, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        os.remove(self.filepath)

    def read(self, size):
        if size > 1024 * 1024:
            size = 1024 * 1024
        time.sleep(size / (SAMPLE_RATE * BYTES_PER_SAMPLE))
        content = self.file.read(size)
        return content


class RTZROpenAPIClient:
    def __init__(self, client_id, client_secret):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.client_id = client_id
        self.client_secret = client_secret
        self._sess = Session()
        self._token = None

    @property
    def token(self):
        if self._token is None or self._token["expire_at"] < time.time():
            resp = self._sess.post(
                API_BASE + "/v1/authenticate",
                data={"client_id": self.client_id, "client_secret": self.client_secret},
            )
            resp.raise_for_status()
            self._token = resp.json()
        return self._token["access_token"]

    def transcribe_streaming_grpc(self, filepath, config):
        base = GRPC_SERVER_URL
        with grpc.secure_channel(base, credentials=grpc.ssl_channel_credentials()) as channel:
            stub = pb_grpc.OnlineDecoderStub(channel)
            cred = grpc.access_token_call_credentials(self.token)

            def req_iterator():
                yield pb.DecoderRequest(streaming_config=config)
                with FileStreamer(filepath) as f:
                    while True:
                        buff = f.read(size=DEFAULT_BUFFER_SIZE)
                        if buff is None or len(buff) == 0:
                            break
                        yield pb.DecoderRequest(audio_content=buff)

            req_iter = req_iterator()
            resp_iter = stub.Decode(req_iter, credentials=cred)
            for resp in resp_iter:
                resp: pb.DecoderResponse
                for res in resp.results:
                    if not res.is_final:
                        print(
                            "\033[K" + "{:.2f} : {}".format(res.start_at / 1000, res.alternatives[0].text),
                            end="\r",
                            flush=True,
                        )
                    else:
                        print(
                            "\033[K"
                            + "{:.2f} - {:.2f} : {}".format(
                                res.start_at / 1000,
                                (res.start_at + res.duration) / 1000,
                                res.alternatives[0].text,
                            ),
                            end="\n",
                        )


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(file_dir)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env_config = configparser.ConfigParser()
    env_config.read(os.path.join(par_dir, "config.ini"))
    parser.add_argument("stream", help="File to stream to the API")
    args = parser.parse_args()

    # Audio file samplerate check for
    if args.stream.endswith(".wav"):  # Only for .wav file
        _, samplerate = sf.read(args.stream)
        if ENCODING == pb.DecoderConfig.AudioEncoding.LINEAR16:
            assert (
                samplerate == SAMPLE_RATE
            ), f"SAMPLE_RATE must be same for using LINEAR16 encoding. Your Audio file Sample Rate is {samplerate}"
        else:
            pass
    else:
        pass

    config = pb.DecoderConfig(
        sample_rate=SAMPLE_RATE,
        encoding=ENCODING,
        use_itn=True,
        use_disfluency_filter=False,
        use_profanity_filter=False,
        keywords=["농협은행:5.0", "넘 예쁘네:-5.0", "리턴제로"],
    )

    client = RTZROpenAPIClient(env_config["DEFAULT"]["CLIENT_ID"], env_config["DEFAULT"]["CLIENT_SECRET"])
    client.transcribe_streaming_grpc(args.stream, config)
