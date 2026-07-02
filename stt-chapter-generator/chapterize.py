from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub.errors import GatedRepoError
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_BATCH_SIZE = 16
RANK_RADIUS = 3
WINDOW_SIZE = 5
BOUNDARY_RANK_THRESHOLD = 0.385
MAX_BOUNDARIES = 10
BOUNDARY_TEXT_CHARS = 1000
REPRESENTATIVE_KEYWORD_COUNT = 8
KEYWORD_POS_TAGS = {"NNG", "NNP", "SL"}
REPRESENTATIVE_MIN_CHARS = 35
REPRESENTATIVE_TARGET_CHARS = 70
REPRESENTATIVE_MAX_CHARS = 90

@dataclass(frozen=True)
class Segment:
    start_at: int
    text: str


@dataclass
class Chapter:
    start_at: int
    segments: list[Segment]
    representative_keywords: list[str] | None = None
    representative_text: str = ""
    boundary_score: float | None = None

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)


def ms_to_timestamp(ms: int) -> str:
    total_seconds = max(ms, 0) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_segments(transcript_path: Path) -> list[Segment]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    utterances = extract_utterances(payload)

    segments = []
    for item in utterances:
        text = str(item.get("msg", "")).strip()
        if not text:
            continue
        segments.append(
            Segment(
                start_at=int(item.get("start_at", 0)),
                text=text,
            )
        )

    if not segments:
        raise ValueError("No transcript utterances found.")
    return segments


def extract_utterances(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        results = payload.get("results", {})
        if isinstance(results, dict):
            utterances = results.get("utterances", [])
            if isinstance(utterances, list):
                return utterances
    return []


def encode_segments(
    segments: list[Segment],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    try:
        model = SentenceTransformer(model_name)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Cannot access embedding model '{model_name}'. "
            "Open the model page on Hugging Face, accept the license terms, "
            "then authenticate in this terminal with `hf auth login` "
            "or set `HF_TOKEN`."
        ) from exc

    embeddings = model.encode(
        [segment.text for segment in segments],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings)


def split_into_chapters_c99(
    segments: list[Segment],
    embeddings: np.ndarray,
) -> list[Chapter]:
    boundary_scores = calculate_c99_boundary_scores(
        embeddings,
        rank_radius=RANK_RADIUS,
        boundary_window=WINDOW_SIZE,
    )
    if not boundary_scores:
        return [Chapter(start_at=segments[0].start_at, segments=segments)]

    max_boundaries = auto_max_boundaries(segments)
    boundaries = select_ranked_boundaries(
        boundary_scores,
        segment_count=len(segments),
        min_segments=WINDOW_SIZE,
        max_boundaries=max_boundaries,
        rank_threshold=BOUNDARY_RANK_THRESHOLD,
    )
    return build_chapters_from_boundaries(segments, boundaries, boundary_scores)


def auto_max_boundaries(segments: list[Segment]) -> int:
    total_chars = sum(len(segment.text) for segment in segments)
    return min(MAX_BOUNDARIES, max(2, round(total_chars / BOUNDARY_TEXT_CHARS)))


def calculate_c99_boundary_scores(
    embeddings: np.ndarray,
    rank_radius: int,
    boundary_window: int,
) -> dict[int, float]:
    segment_count = len(embeddings)
    if segment_count < 2:
        return {}

    similarities = np.asarray(embeddings @ embeddings.T, dtype=np.float32)
    rank_matrix = local_rank_matrix(similarities, rank_radius)
    prefix = padded_prefix_sum(rank_matrix)

    scores: dict[int, float] = {}
    for gap in range(1, segment_count):
        left_start = max(0, gap - boundary_window)
        left_end = gap
        right_start = gap
        right_end = min(segment_count, gap + boundary_window)

        left_mean = block_mean(prefix, left_start, left_end, left_start, left_end)
        right_mean = block_mean(prefix, right_start, right_end, right_start, right_end)
        cross_mean = block_mean(prefix, left_start, left_end, right_start, right_end)
        scores[gap] = ((left_mean + right_mean) / 2.0) - cross_mean
    return scores


def local_rank_matrix(similarities: np.ndarray, radius: int) -> np.ndarray:
    size = similarities.shape[0]
    ranks = np.empty_like(similarities, dtype=np.float32)
    for row in range(size):
        row_start = max(0, row - radius)
        row_end = min(size, row + radius + 1)
        for column in range(size):
            column_start = max(0, column - radius)
            column_end = min(size, column + radius + 1)
            window = similarities[row_start:row_end, column_start:column_end]
            ranks[row, column] = np.mean(window <= similarities[row, column])
    return ranks


def padded_prefix_sum(matrix: np.ndarray) -> np.ndarray:
    prefix = matrix.cumsum(axis=0).cumsum(axis=1)
    return np.pad(prefix, ((1, 0), (1, 0)), mode="constant")


def block_sum(
    prefix: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> float:
    return float(
        prefix[row_end, col_end]
        - prefix[row_start, col_end]
        - prefix[row_end, col_start]
        + prefix[row_start, col_start]
    )


def block_mean(
    prefix: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> float:
    area = (row_end - row_start) * (col_end - col_start)
    if area <= 0:
        return 0.0
    return block_sum(prefix, row_start, row_end, col_start, col_end) / area


def select_ranked_boundaries(
    boundary_scores: dict[int, float],
    segment_count: int,
    min_segments: int,
    max_boundaries: int,
    rank_threshold: float,
) -> list[int]:
    if max_boundaries < 1:
        return []

    ranked = sorted(boundary_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []

    selected: list[int] = []
    denominator = max(len(ranked) - 1, 1)
    for rank, (gap, _) in enumerate(ranked):
        rank_ratio = 1.0 - (rank / denominator)
        if rank_ratio < rank_threshold:
            continue
        if gap < min_segments or segment_count - gap < min_segments:
            continue
        if any(abs(gap - selected_gap) < min_segments for selected_gap in selected):
            continue

        selected.append(gap)
        if len(selected) >= max_boundaries:
            break

    return sorted(selected)


def build_chapters_from_boundaries(
    segments: list[Segment],
    boundaries: list[int],
    boundary_scores: dict[int, float],
) -> list[Chapter]:
    starts = [0] + boundaries
    ends = boundaries + [len(segments)]
    chapters = []
    for start, end in zip(starts, ends):
        chapters.append(
            Chapter(
                start_at=segments[start].start_at,
                segments=segments[start:end],
                boundary_score=boundary_scores.get(start) if start else None,
            )
        )
    return chapters


def add_representative_keywords(
    chapters: list[Chapter],
    keyword_count: int = REPRESENTATIVE_KEYWORD_COUNT,
) -> None:
    chapter_tokens = [tokenize_keywords(chapter.text) for chapter in chapters]
    document_frequency = Counter(
        token for tokens in chapter_tokens for token in set(tokens)
    )
    total_documents = len(chapter_tokens)

    for chapter, tokens in zip(chapters, chapter_tokens):
        keywords = extract_keywords(
            tokens,
            document_frequency,
            total_documents,
            keyword_count,
        )
        chapter.representative_keywords = keywords


def add_representative_texts(chapters: list[Chapter]) -> None:
    for chapter in chapters:
        keywords = chapter.representative_keywords or []
        chapter.representative_text = select_representative_segment(
            chapter.segments,
            keywords,
        )


def select_representative_segment(
    segments: list[Segment],
    keywords: list[str],
    target_chars: int = REPRESENTATIVE_TARGET_CHARS,
) -> str:
    if not segments:
        return ""

    scored: list[tuple[float, int, str]] = []
    for start_index, candidate in representative_candidates(segments, target_chars * 2):
        keyword_hits = keyword_hit_counts(candidate, keywords)
        total_hits = sum(keyword_hits)
        unique_hits = sum(1 for hits in keyword_hits if hits > 0)
        length_score = min(len(candidate), target_chars) / target_chars
        score = (unique_hits * 3.0) + total_hits + length_score
        scored.append((score, -start_index, candidate))

    best_text = max(scored, key=lambda item: item[:2])[2]
    return best_keyword_window(best_text, keywords, target_chars)


def representative_candidates(
    segments: list[Segment],
    max_chars: int,
    max_segments: int = 3,
) -> list[tuple[int, str]]:
    candidates = []
    for start in range(len(segments)):
        parts = []
        for end in range(start, min(len(segments), start + max_segments)):
            parts.append(segments[end].text)
            candidate = " ".join(parts)
            if len(candidate) > max_chars and end > start:
                break
            candidates.append((start, candidate))
    return candidates


def keyword_hit_counts(text: str, keywords: list[str]) -> list[int]:
    return [
        len(re.findall(re.escape(keyword), text, re.IGNORECASE))
        for keyword in keywords
    ]


def best_keyword_window(text: str, keywords: list[str], target_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= target_chars:
        return text
    if not keywords:
        return shorten_text(text, target_chars)

    best_score = -1
    best_start = 0
    last_start = max(0, len(text) - target_chars)
    for start in range(last_start + 1):
        window = text[start : start + target_chars]
        keyword_hits = keyword_hit_counts(window, keywords)
        unique_hits = sum(1 for hits in keyword_hits if hits > 0)
        total_hits = sum(keyword_hits)
        score = (unique_hits * 3) + total_hits
        if score > best_score:
            best_score = score
            best_start = start

    if best_score <= 0:
        return shorten_text(text, target_chars)

    return format_excerpt(text, best_start, target_chars)


def format_excerpt(text: str, start: int, target_chars: int) -> str:
    start = move_to_next_word_start(text, start)
    end = select_excerpt_end(text, start, target_chars)
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = f"...{excerpt}"
    if end < len(text) and not ends_with_sentence_punctuation(excerpt):
        excerpt = f"{excerpt}..."
    return excerpt


def select_excerpt_end(text: str, start: int, target_chars: int) -> int:
    min_end = min(len(text), start + REPRESENTATIVE_MIN_CHARS)
    target_end = min(len(text), start + target_chars)
    max_end = min(len(text), start + REPRESENTATIVE_MAX_CHARS)
    punctuation_ends = [
        match.end()
        for match in re.finditer(r"[.!。！]\s*", text[start:max_end])
        if start + match.end() >= min_end
    ]
    if punctuation_ends:
        return start + min(punctuation_ends)

    if max_end == len(text):
        return max_end

    space_ends = [
        match.start()
        for match in re.finditer(r"\s+", text[start:max_end])
        if start + match.start() >= min_end
    ]
    if space_ends:
        return start + min(space_ends, key=lambda end: abs((start + end) - target_end))

    return target_end


def ends_with_sentence_punctuation(text: str) -> bool:
    return bool(re.search(r"[.!?。！？]$", text.strip()))


def move_to_next_word_start(text: str, start: int) -> int:
    if start <= 0 or start >= len(text):
        return start
    if text[start - 1].isspace() or text[start].isspace():
        return start

    next_space = text.find(" ", start)
    if next_space == -1 or next_space - start > 12:
        return start
    return next_space + 1


def shorten_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    shortened = text[: max_chars + 1].rstrip()
    split_at = max(shortened.rfind(" "), shortened.rfind(","), shortened.rfind("."))
    if split_at >= max_chars * 0.6:
        shortened = shortened[:split_at].rstrip()
    else:
        shortened = shortened[:max_chars].rstrip()
    return f"{shortened}..."


def tokenize_keywords(text: str) -> list[str]:
    kiwi = get_kiwi()
    tokens = []
    for token in kiwi.tokenize(text):
        if token.tag not in KEYWORD_POS_TAGS:
            continue
        keyword = normalize_keyword(token.form)
        if is_keyword_candidate(keyword):
            tokens.append(keyword)
    return tokens


@lru_cache(maxsize=1)
def get_kiwi() -> Kiwi:
    return Kiwi()


def normalize_keyword(token: str) -> str:
    token = token.lower() if re.search(r"[A-Za-z]", token) else token
    return token.strip()


def is_keyword_candidate(token: str) -> bool:
    return len(token) >= 2


def extract_keywords(
    tokens: list[str],
    document_frequency: Counter[str],
    total_documents: int,
    keyword_count: int,
) -> list[str]:
    term_frequency = Counter(tokens)
    scored = []
    for token, frequency in term_frequency.items():
        idf = math.log((total_documents + 1) / (document_frequency[token] + 1)) + 1.0
        score = frequency * idf
        scored.append((score, token))

    keywords = []
    for _, token in sorted(scored, key=lambda item: (-item[0], item[1])):
        if token not in keywords:
            keywords.append(token)
        if len(keywords) == keyword_count:
            break
    return keywords


def render_chapter(chapter: Chapter, number: int) -> dict[str, Any]:
    return {
        "number": number,
        "start_at": chapter.start_at,
        "start": ms_to_timestamp(chapter.start_at),
        "representative_text": chapter.representative_text,
        "text": chapter.text,
        "segment_count": len(chapter.segments),
        "boundary_score": chapter.boundary_score,
    }


def render_markdown(chapters: list[dict[str, Any]], source_name: str) -> str:
    lines = [f"# Chapters: {source_name}", ""]
    for chapter in chapters:
        lines.append(f"- **{chapter['start']}**")
        if chapter["representative_text"]:
            lines.append(f"  - 대표 발화: {chapter['representative_text']}")
    lines.append("")
    return "\n".join(lines)


def default_output_paths(transcript_path: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = transcript_path.name.replace(".transcript.json", "").replace(".json", "")
    return output_dir / f"{stem}.chapters.json", output_dir / f"{stem}.chapters.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create timestamped chapters from an RTZR transcript."
    )
    parser.add_argument("transcript", type=Path, help="Path to RTZR transcript JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcript_path = args.transcript.expanduser().resolve()
    segments = load_segments(transcript_path)
    embeddings = encode_segments(
        segments,
        EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    chapters = split_into_chapters_c99(
        segments,
        embeddings,
    )
    add_representative_keywords(chapters)
    add_representative_texts(chapters)

    rendered = [render_chapter(chapter, index) for index, chapter in enumerate(chapters, 1)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = default_output_paths(transcript_path, args.output_dir)
    json_path.write_text(
        json.dumps(rendered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown(rendered, transcript_path.stem.replace(".transcript", "")),
        encoding="utf-8",
    )

    print(f"saved={json_path}")
    print(f"saved={md_path}")


if __name__ == "__main__":
    main()
