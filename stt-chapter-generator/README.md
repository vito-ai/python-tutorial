# RTZR STT Chapter Generator

RTZR STT API로 음성 파일을 전사한 뒤, 오픈소스 문장 임베딩 모델로 챕터 경계를 자동 생성하는 Python 예제입니다.
챕터 표시는 별도 LLM 없이, 실제 전사문에서 고른 대표 발화를 사용합니다.

역할은 단순하게 나눕니다.

- `transcribe.py`: RTZR STT API에 오디오 파일을 보내고 transcript JSON을 저장합니다.
- `chapterize.py`: transcript의 문단을 문장 임베딩 모델로 비교해 C99-rank 방식으로 챕터 경계를 찾습니다. 챕터 표시는 Kiwi 형태소 분석기로 추출한 내부 키워드를 이용해 대표 발화를 고릅니다.

## 1. Setup

```bash
cd stt-chapter-generator
uv sync
```

이 예제는 `pyproject.toml`에 의존성을 정의하고 `uv`로 실행 환경을 관리합니다.
`.python-version`에는 기본 Python 버전을 명시했습니다.
`uv sync`를 실행하면 `.venv`와 `uv.lock`을 기준으로 필요한 패키지가 설치됩니다.

프로젝트 폴더의 `.env.example`을 복사해 `.env` 파일을 만들고 RTZR 키를 저장합니다.

```bash
cp .env.example .env
```

```env
RTZR_CLIENT_ID=...
RTZR_CLIENT_SECRET=...
```

실행할 때는 `uv run --env-file .env`를 사용해 `.env` 값을 함께 불러옵니다.

```bash
uv run --env-file .env -- python transcribe.py --help
```

## 2. Transcribe

분석할 음성 파일을 준비한 뒤 경로를 인자로 넘깁니다.

```bash
uv run --env-file .env -- python transcribe.py path/to/audio.wav \
  --model-name whisper \
  --language ko \
  --use-paragraph-splitter \
  --paragraph-max 40 \
  --use-disfluency-filter
```

결과는 기본적으로 아래 경로에 저장됩니다.

```text
data/transcripts/audio.transcript.json
```

`--use-disfluency-filter`는 `음`, `아`, 반복 발화처럼 의미가 약한 구어체 표현을 줄이는 RTZR 옵션입니다. RTZR의 기본값은 켜짐입니다. 원본 구어체에 가까운 전사 결과와 비교하고 싶다면 아래처럼 끌 수 있습니다.

```bash
uv run --env-file .env -- python transcribe.py path/to/audio.wav \
  --model-name whisper \
  --language ko \
  --use-paragraph-splitter \
  --paragraph-max 40 \
  --no-disfluency-filter \
  --output data/transcripts/audio.raw.transcript.json
```

## 3. Generate Chapters

기본 실행은 문장 임베딩 모델과 C99-rank 경계 점수로 챕터 경계를 생성합니다. 이후 각 챕터 안에서 자주 등장하면서 전체 전사에서는 상대적으로 덜 흔한 명사 키워드를 내부적으로 고르고, 키워드와 가까운 실제 발화를 대표 발화로 표시합니다.

챕터 수 범위는 전체 전사 길이를 기준으로 자동 계산됩니다. 짧은 음성은 적은 수의 챕터로, 긴 음성은 더 많은 챕터로 나뉘며, 너무 잘게 나뉘지 않도록 상한을 둡니다.

```bash
uv run python chapterize.py data/transcripts/audio.transcript.json
```

결과:

```text
data/outputs/audio.chapters.json
data/outputs/audio.chapters.md
```

Markdown 결과를 바로 확인하려면:

```bash
cat data/outputs/audio.chapters.md
```

## Fixed Defaults

튜토리얼에서는 사용자가 경계 탐지 값을 직접 튜닝하지 않도록 주요 값을 코드 내부 기본값으로 고정했습니다.

- 경계 간 최소 간격: `5`개 문단
- 경계 계산 window: `5`
- 챕터 수 범위: 전체 전사 길이를 기준으로 자동 계산
- 자동 챕터 수 선택: 경계 점수 곡선의 기울기 변화가 큰 지점 사용
- C99-rank 반경: `3`
- 챕터 표시 방식: 실제 전사문에서 고른 대표 발화
- 내부 키워드 개수: `8`개

사용자가 실행 시 바꿀 수 있는 옵션은 출력 위치를 정하는 `--output-dir`입니다.

## Speech-Like Transcripts

실제 음성 전사는 글보다 구어체 표현, 반복, 머뭇거림이 많습니다. 이 예제에서는 전사 단계에서 `--use-disfluency-filter`를 사용해 간투어를 줄이고, 챕터 경계는 정리된 전사 문단을 기준으로 계산합니다.

다만 필터가 모든 구어체 문제를 해결하는 것은 아닙니다. 그래서 튜토리얼이나 실험에서는 같은 음성을 `--use-disfluency-filter`와 `--no-disfluency-filter`로 각각 전사한 뒤 챕터 결과를 비교해보는 것이 좋습니다.

## Representative Text

챕터 표시는 LLM으로 새 문장을 생성하지 않습니다. Kiwi 형태소 분석기로 일반 명사, 고유 명사, 외국어 토큰을 추출한 뒤 TF-IDF 점수로 챕터별 내부 키워드를 고릅니다.

이 키워드는 출력에 직접 노출하지 않고, 챕터를 잘 대표하는 발화를 고르는 데만 사용합니다. 이 방식은 별도 로컬 LLM을 설치하지 않아도 되고 실행이 빠릅니다. 대신 사람이 쓴 제목처럼 자연스러운 문장을 만드는 방식이 아니라, 실제 전사에서 고른 대표 발화를 보여주는 방식입니다.

## Example Output

저장소에는 음성 파일과 전사 결과를 포함하지 않습니다. 아래는 국립민속박물관의 Creative Commons 라이선스 영상인 [\[전시라이브러리\] '그 겨울의 행복' 길상 특별전](https://www.youtube.com/watch?v=-VD6kS4d_kE)을 기준으로 한 실행 결과 형식 예시입니다.

> 예제 영상 출처  
> - 제목: [\[전시라이브러리\] '그 겨울의 행복' 길상 특별전](https://www.youtube.com/watch?v=-VD6kS4d_kE)  
> - 채널: 국립민속박물관  
> - 라이선스: YouTube Creative Commons Attribution license (reuse allowed)

```md
# Chapters: gilsang_winter_happiness

- **00:00:12**
  - 대표 발화: 전시회 구성은 1부에서 길상과 행복의 의미를 환기시킨 후에 2부와 3부에서 본격적으로 길상의 모습을 살펴볼 수 있도록 했습니다.
- **00:02:39**
  - 대표 발화: ...고양이 역시 70세 노인을 뜻하는 한자와 발음이 같아 장수를 상징하며, 오래 사는 열 가지인 십장생 또한 대표적인 장수의 상징입니다.
- **00:05:11**
  - 대표 발화: ...가치에 대한 측면이었지만, 행복에는 즐거움, 만족감 같은 정서적인 측면도 있습니다.
- **00:06:01**
  - 대표 발화: 전시 관람이라는 행위 자체가 행복한 경험이 되기를 바라는 취지에서 공간을 조성했고, 휴식과 관람이 조화를 이룰 수 있도록 구성하였습니다.
```

## Models

기본 모델:

- Embedding: `google/embeddinggemma-300m`

처음 실행할 때 Hugging Face에서 모델 파일을 내려받기 때문에 시간이 걸릴 수 있습니다. 한 번 받은 뒤에는 로컬 캐시를 사용합니다.

`google/embeddinggemma-300m`은 Hugging Face에서 Google Gemma 사용 조건 동의가 필요할 수 있습니다. 처음 실행할 때 접근 권한 오류가 나면 Hugging Face에 로그인한 뒤 모델 페이지에서 라이선스 조건에 동의하고 다시 실행합니다.
