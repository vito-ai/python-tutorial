import argparse
import json
from pathlib import Path
from typing import Any

from rtzr_openapi_client import RTZROpenAPIClient


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

    output_path = args.output or default_output_path(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = RTZROpenAPIClient()
    config = build_config(args)
    submit = client.transcribe_file(audio_path, config)
    transcribe_id = submit["id"]
    print(f"transcribe_id={transcribe_id}")

    transcript = client.wait_for_result(
        transcribe_id,
        poll_interval_sec=args.poll_interval,
        timeout_sec=args.timeout,
    )
    if transcript.get("status") == "failed":
        raise RuntimeError(json.dumps(transcript, ensure_ascii=False, indent=2))

    output_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
