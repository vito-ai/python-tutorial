import os

import pytest

MODEL_NAME = os.environ.get("MODEL_NAME", "python_vad")
TRITON_HOST = os.environ.get("TRITON_HOST", "localhost")
TRITON_GRPC = os.environ.get("TRITON_GRPC", "8101")


@pytest.fixture(params=[("grpc", TRITON_GRPC, False)])
def config(request):
    return request.param
