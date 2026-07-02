import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


API_BASE_URL = "https://openapi.vito.ai/v1"


def import_requests() -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install dependencies first: python3 -m pip install -r requirements.txt") from exc
    return requests


def authenticate(client_id: str, client_secret: str) -> str:
    requests = import_requests()
    response = requests.post(
        f"{API_BASE_URL}/authenticate",
        data={"client_id": client_id, "client_secret": client_secret},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_name": args.model_name,
        "language": args.language,
        "use_disfluency_filter": args.use_disfluency_filter,
    }

    if args.use_paragraph_splitter:
        config["use_paragraph_splitter"] = True
        config["paragraph_splitter"] = {"max": args.paragraph_max}

    return config


def submit_transcription(audio_path: Path, access_token: str, config: dict[str, Any]) -> str:
    requests = import_requests()
    with audio_path.open("rb") as audio_file:
        response = requests.post(
            f"{API_BASE_URL}/transcribe",
            headers={"Authorization": f"Bearer {access_token}"},
            files={"file": (audio_path.name, audio_file)},
            data={"config": json.dumps(config, ensure_ascii=False)},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()["id"]


def poll_transcription(
    transcribe_id: str,
    access_token: str,
    poll_interval: int,
    timeout: int,
) -> dict[str, Any]:
    requests = import_requests()
    started_at = time.monotonic()

    while True:
        response = requests.get(
            f"{API_BASE_URL}/transcribe/{transcribe_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")

        if status == "completed":
            return payload
        if status == "failed":
            raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))
        if time.monotonic() - started_at > timeout:
            raise TimeoutError(f"Timed out while waiting for transcription: {transcribe_id}")

        print(f"status={status}; waiting {poll_interval}s...")
        time.sleep(poll_interval)


def default_output_path(audio_path: Path) -> Path:
    return Path("data/transcripts") / f"{audio_path.stem}.transcript.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with RTZR STT API."
    )
    parser.add_argument("audio", type=Path, help="Path to an audio file.")
    parser.add_argument("-o", "--output", type=Path, help="Where to save transcript JSON.")
    parser.add_argument("--model-name", default="whisper", help="RTZR model_name value.")
    parser.add_argument("--language", default="ko", help="Language code for the selected model.")
    parser.add_argument(
        "--use-paragraph-splitter",
        action="store_true",
        help="Ask RTZR to return paragraph-like utterance units.",
    )
    parser.add_argument(
        "--paragraph-max",
        type=int,
        default=40,
        help="Maximum paragraph length used by RTZR paragraph splitter.",
    )
    disfluency_group = parser.add_mutually_exclusive_group()
    disfluency_group.add_argument(
        "--use-disfluency-filter",
        dest="use_disfluency_filter",
        action="store_true",
        default=True,
        help="Remove filler and repeated speech expressions. This is the RTZR default.",
    )
    disfluency_group.add_argument(
        "--no-disfluency-filter",
        dest="use_disfluency_filter",
        action="store_false",
        help="Keep filler and repeated speech expressions in the transcript.",
    )
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    client_id = os.getenv("RTZR_CLIENT_ID")
    client_secret = os.getenv("RTZR_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set RTZR_CLIENT_ID and RTZR_CLIENT_SECRET before running.")

    output_path = args.output or default_output_path(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    access_token = authenticate(client_id, client_secret)
    config = build_config(args)
    transcribe_id = submit_transcription(audio_path, access_token, config)
    print(f"transcribe_id={transcribe_id}")

    transcript = poll_transcription(
        transcribe_id,
        access_token,
        args.poll_interval,
        args.timeout,
    )
    output_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
