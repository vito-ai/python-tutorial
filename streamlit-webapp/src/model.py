import json

import requests


class RtzrAPI:
    def __init__(
        self,
        client_id: str,
        client_sceret: str,
        dev: bool,
        file_path: dict,
        speaker_num: int,
        domain: str,
        profanity_filter: bool,
        keyword: list,
        summary_model,
        summary_tokenizer,
    ) -> None:
        """api 사용에 필요한 인자 초기화"""
        self.dev: str = "dev-" if dev else ""
        self.client_id: str = client_id
        self.client_sceret: str = client_sceret
        self.file_path: dict = file_path
        self.speaker_num: int = speaker_num
        self.config: dict = {"domain": "GENERAL"} if domain == "일반" else {"domain": "CALL"}

        if speaker_num != 0:
            self.config["use_diarization"] = True
            self.config["diarization"] = {"spk_count": speaker_num}
        if profanity_filter:
            self.config["use_profanity_filter"] = True
        if keyword:
            self.config["keyword"] = keyword

        self.model = summary_model
        self.tokenizer = summary_tokenizer

        self.raw_data: json = None
        self.voice_data: str = None
        self.summary_data: str = None
        self.access_token: str = self.auth_check(client_id, client_sceret)
        self.transcribe_id: str = self.api_post(self.access_token)

    def auth_check(self, client_id: str, client_sceret: str) -> str:
        """client id와 client sceret을 인자로 받고 api요청에 필요한 access token을 리턴하는 메소드"""
        resp = requests.post(
            f"https://{self.dev}openapi.vito.ai/v1/authenticate",
            data={"client_id": client_id, "client_secret": client_sceret},
        )

        resp.raise_for_status()
        return resp.json()["access_token"]

    def api_post(self, access_token: str) -> None:
        """access_token을 인자로 받고 api요청을 하는 함수. Get에 필요한 transcribe_id 리턴하는 메소드"""
        resp = requests.post(
            f"https://{self.dev}openapi.vito.ai/v1/transcribe",
            headers={"Authorization": f"Bearer {access_token}"},
            files=self.file_path,
            data={"config": json.dumps(self.config)},
        )

        resp.raise_for_status()
        return resp.json()["id"]

    def api_get(
        self,
    ) -> None:
        """텍스트로 변환된 값을 받는 메소드"""
        resp = requests.get(
            f"https://{self.dev}openapi.vito.ai/v1/transcribe/" + self.transcribe_id,
            headers={"Authorization": "bearer " + self.access_token},
        )
        resp.raise_for_status()
        if resp.json()["status"] == "transcribing":
            self.raw_data = None
        else:
            self.raw_data = resp.json()
            self.voice_data = self.preprocessing(self.raw_data)

    def preprocessing(self, raw_data: dict) -> str:
        """
        텍스트를 전처리 하는 메소드
        화자가 두 명 이상일 경우 speaker n] text... 형태로 출력
        """
        if len(set([x["spk"] for x in raw_data["results"]["utterances"]])) == 1:
            return " ".join([data["msg"] for data in raw_data["results"]["utterances"]])
        else:
            return "  \n".join(
                [f"화자{text_data['spk']} ] {text_data['msg']}" for text_data in raw_data["results"]["utterances"]]
            )

    def summary_inference(
        self,
    ) -> None:
        """hugging face 모델을 사용해 텍스트 내용을 요약해주는 함수. 문장이 짧으면 'Text too short를 반환"""

        # short data handling
        if len(self.voice_data) < 40:
            self.summary_data = "Text too short!!"
            return None

        # Encoding
        inputs = self.tokenizer(
            self.voice_data,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1026,
        )

        # Generate Summary Text
        summary_text_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            length_penalty=1.0,
            max_length=300,
            min_length=12,
            num_beams=6,
            repetition_penalty=1.5,
            no_repeat_ngram_size=15,
        )

        # Decoding
        self.summary_data = self.tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

    def get_raw_data(self) -> dict:
        """api를 통해 얻은 원시 데이터 반환"""
        return self.raw_data

    def get_text_data(self) -> str:
        """텍스트로 변환된 데이터 반환"""
        return self.voice_data

    def get_summary_data(self) -> str:
        """텍스트가 요약된 데이터 반환"""
        return self.summary_data
