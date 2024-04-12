# RTZR Streaming STT API(gRPC) with Mic. Interface

## Requirements
This repository need `PyAudio` library. First, you need to read installation instruction `PyAudio` [this link](https://pypi.org/project/PyAudio/). 

And Then,

```bash
pip install -r requirements.txt
```
 Download `.proto` file from this [Link](https://github.com/vito-ai/openapi-grpc/blob/main/protos/vito-stt-client.proto). And save `.proto` file in the project directory.
 
 And run this command to generate grpc file.
 ```bash
 python -m grpc_tools.protoc -I. --python_out=./src --grpc_python_out=./src ./vito-stt-client.proto
 ```
 