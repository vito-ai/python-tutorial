"""

To Download definition (.proto) file

$ wget https://raw.github.com/vito-ai/openapi-grpc/main/protos/vito-stt-client.proto

To generate gRPC code

$ pip install grpcio-tools
$ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./vito-stt-client.proto

NOTE: This module requires the dependencies `grpcio` and `requests`.
To install using pip:
    pip install grpcio
    pip install requests

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
import pyaudio
import queue
import soundfile as sf
import vito_stt_client_pb2 as pb
import vito_stt_client_pb2_grpc as pb_grpc
from requests import Session

API_BASE = "https://openapi.vito.ai"
GRPC_SERVER_URL = "grpc-openapi.vito.ai:443"

SAMPLE_RATE = 16000
ENCODING = pb.DecoderConfig.AudioEncoding.LINEAR16

CHUNK = 1024
FORMAT = pyaudio.paInt16 
CHANNELS = 1 # Only supports 1-Channel Input 

class MicrophoneStream:
    """
    Ref[1]: https://cloud.google.com/speech-to-text/docs/transcribe-streaming-audio

    Recording Stream을 생성하고 오디오 청크를 생성하는 제너레이터를 반환하는 클래스.
    """

    def __init__(self: object, rate: int = SAMPLE_RATE, chunk: int = CHUNK, channels: int = CHANNELS, format = FORMAT) -> None:
        self._rate = rate
        self._chunk = chunk
        self._channels = channels
        self._format = format

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False

    def terminate(
        self: object,
    ) -> None:
        """
        Stream을 닫고, 제너레이터를 종료하는 함수
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """
        오디오 Stream으로부터 데이터를 수집하고 버퍼에 저장하는 콜백 함수.

        Args:
            in_data: 바이트 오브젝트로 된 오디오 데이터
            frame_count: 프레임 카운트
            time_info: 시간 정보
            status_flags: 상태 플래그

        Returns:
            바이트 오브젝트로 된 오디오 데이터
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """
        Stream으로부터 오디오 청크를 생성하는 Generator.

        Args:
            self: The MicrophoneStream object

        Returns:
            오디오 청크를 생성하는 Generator
        """
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

class RTZROpenAPIClient:
    def __init__(self, client_id, client_secret):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.client_id = client_id
        self.client_secret = client_secret
        self._sess = Session()
        self._token = None

        self.stream = MicrophoneStream(SAMPLE_RATE, CHUNK, CHANNELS, FORMAT) # 마이크 입력을 오디오 인터페이스 사용하기 위한 Stream 객체 생성

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

    def transcribe_streaming_grpc(self, config):
        base = GRPC_SERVER_URL
        with grpc.secure_channel(base, credentials=grpc.ssl_channel_credentials()) as channel:
            stub = pb_grpc.OnlineDecoderStub(channel)
            cred = grpc.access_token_call_credentials(self.token)

            audio_generator = self.stream.generator()

            def req_iterator():
                yield pb.DecoderRequest(streaming_config=config)
                for chunk in audio_generator: # (2). yield from Stream Generator
                    yield pb.DecoderRequest(audio_content=chunk) # chunk를 넘겨서, 스트리밍 STT 수행

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

    def __del__(self):
        self.stream.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    env_config = configparser.ConfigParser()
    env_config.read("config.ini")
    # parser.add_argument("stream", help="File to stream to the API") # 마이크로 입력을 받기에 파일의 위치는 사용되지 않습니다.
    args = parser.parse_args()

    config = pb.DecoderConfig(
        sample_rate=SAMPLE_RATE,
        encoding=ENCODING,
        use_itn=True,
        use_disfluency_filter=False,
        use_profanity_filter=False,
    )

    client = RTZROpenAPIClient(env_config["DEFAULT"]["CLIENT_ID"], env_config["DEFAULT"]["CLIENT_SECRET"])
    try:
        client.transcribe_streaming_grpc(config)

    except KeyboardInterrupt:
        print("Program terminated by user.")
        del client
