# Triton, Tritony VAD Client sample

## Requirements
This repository requires `pydub`, `tritonclient`, `tritony` python libraries. So you need to run following command.

```bash
pip install -r requirements.txt
```

Next, you can build Docker image using shell script.

```bash
/bin/bash ./build_triton_server.sh
```
or

For manually building Docker image, you must assign your docker image tag `triton-vad-server:latest`.

## Run Triton Inference Server
To start Triton Inference Server, run the below command.
You will need to set the `HF_TOKEN` environment variable to your Hugging Face API token. ( https://huggingface.co/docs/hub/security-tokens )

```bash
export HF_TOKEN=YOUR_HF_TOKEN
/bin/bash ./bin/run_triton_server.sh
```
## Test
First, you must run Triton Inference Server.
```bash
pytest -s
```