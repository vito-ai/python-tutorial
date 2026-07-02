from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from requests import Session


class RTZROpenAPIClient:
    """Minimal client for RTZR OpenAPI auth and file STT."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        base_url: str = "https://openapi.vito.ai",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or os.getenv("RTZR_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("RTZR_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Missing credentials. Set RTZR_CLIENT_ID and RTZR_CLIENT_SECRET."
            )

        self._session = Session()
        self._token: dict[str, Any] | None = None

    @property
    def token(self) -> str:
        if self._token is None or self._token.get("expire_at", 0) < time.time() - 1800:
            response = self._session.post(
                f"{self.base_url}/v1/authenticate",
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=30,
            )
            response.raise_for_status()
            self._token = response.json()

        access_token = self._token.get("access_token")
        if not access_token:
            raise RuntimeError("authenticate: 'access_token' not found in response")
        return access_token

    def transcribe_file(self, file_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
        audio_path = Path(file_path)
        with audio_path.open("rb") as audio_file:
            response = self._session.post(
                f"{self.base_url}/v1/transcribe",
                headers={"Authorization": f"Bearer {self.token}"},
                files={"file": (audio_path.name, audio_file)},
                data={"config": json.dumps(config, ensure_ascii=False)},
                timeout=60,
            )
        response.raise_for_status()
        return response.json()

    def get_transcription(self, transcribe_id: str) -> dict[str, Any]:
        response = self._session.get(
            f"{self.base_url}/v1/transcribe/{transcribe_id}",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_result(
        self,
        transcribe_id: str,
        poll_interval_sec: int = 5,
        timeout_sec: int = 900,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Timed out while waiting for transcription: {transcribe_id}"
                )

            result = self.get_transcription(transcribe_id)
            status = result.get("status")
            if status in ("completed", "failed"):
                return result

            print(f"status={status}; waiting {poll_interval_sec}s...")
            time.sleep(poll_interval_sec)
