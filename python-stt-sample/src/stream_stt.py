import configparser
import logging
import os
import queue
import sys
import time

import grpc
import pyaudio
import vito_stt_client_pb2 as pb
import vito_stt_client_pb2_grpc as pb_grpc
from requests import Session

API_BASE = "https://openapi.vito.ai"
GRPC_SERVER_URL = "grpc-openapi.vito.ai:443"

SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == "darwin" else 2
RECORD_SECONDS = 10
CHUNK = int(SAMPLE_RATE / 10)  # 100ms chunk
ENCODING = pb.DecoderConfig.AudioEncoding.LINEAR16


def get_config(keywords=None):
    """
    keywords를 받아서, config를 동적으로 생성하는 함수
    """
    return pb.DecoderConfig(
        sample_rate=SAMPLE_RATE,
        encoding=ENCODING,
        use_itn=True,
        use_disfluency_filter=False,
        use_profanity_filter=False,
        keywords=keywords,
    )


class MicrophoneStream:
    """
    Ref[1]: https://cloud.google.com/speech-to-text/docs/transcribe-streaming-audio

    Recording Stream을 생성하고 오디오 청크를 생성하는 제너레이터를 반환하는 클래스.
    """

    def __init__(
        self: object,
        rate: int = SAMPLE_RATE,
        chunk: int = CHUNK,
        channels: int = CHANNELS,
        format=FORMAT,
    ) -> None:
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
            channels=1,
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

    def reset_stream(self):
        self.stream = MicrophoneStream(SAMPLE_RATE, CHUNK, CHANNELS, FORMAT)

    @property
    def token(self):
        """
        Access Token을 서버에 요청해서 얻어오는 프로퍼티.
        """
        if self._token is None or self._token["expire_at"] < time.time():
            resp = self._sess.post(
                API_BASE + "/v1/authenticate",
                data={"client_id": self.client_id, "client_secret": self.client_secret},
            )
            resp.raise_for_status()
            self._token = resp.json()
        return self._token["access_token"]

    def print_transcript(self, start_time, transcript, is_final=False):
        clear_line_escape = "\033[K"  # clear line Escape Sequence
        if not is_final:
            end_time_str = "~"
            end_chr = "\r"
        else:
            end_time_str = f"~ {(start_time + transcript.duration / 1000):.2f}"
            end_chr = None
        print(f"{clear_line_escape}{start_time:.2f} {end_time_str} : {transcript.alternatives[0].text}", end=end_chr)

    def transcribe_streaming_grpc(self):
        """
        grpc를 이용해서 스트리밍 STT 수행 하는 메소드
        """
        base = GRPC_SERVER_URL
        with grpc.secure_channel(base, credentials=grpc.ssl_channel_credentials()) as channel:
            stub = pb_grpc.OnlineDecoderStub(channel)
            cred = grpc.access_token_call_credentials(self.token)

            def req_iterator(keywords):
                yield pb.DecoderRequest(streaming_config=get_config(keywords))

                self.reset_stream()
                for chunk in self.stream.generator():
                    yield pb.DecoderRequest(audio_content=chunk)

            keyword_idx = False
            keywords = {
                True: ["농협은행:5.0", "넘 예쁘네:-5.0", "리턴제로"],
                False: ["넘 예쁘네", "이턴제로"],
            }

            global_st_time = time.time()
            while True:
                keyword_idx = not keyword_idx

                req_iter = req_iterator(keywords=keywords[keyword_idx])
                resp_iter = stub.Decode(req_iter, credentials=cred)
                session_st_time = time.time() - global_st_time
                for resp in resp_iter:
                    resp: pb.DecoderResponse
                    for res in resp.results:
                        if res.is_final:
                            self.stream.terminate()
                        self.print_transcript(session_st_time, res, is_final=res.is_final)

    def __del__(self):
        self.stream.terminate()


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(file_dir)

    env_config = configparser.ConfigParser()
    env_config.read(os.path.join(par_dir, "config.ini"))

    client = RTZROpenAPIClient(env_config["DEFAULT"]["CLIENT_ID"], env_config["DEFAULT"]["CLIENT_SECRET"])
    try:
        client.transcribe_streaming_grpc()

    except KeyboardInterrupt:
        print("Program terminated by user.")
        del client
