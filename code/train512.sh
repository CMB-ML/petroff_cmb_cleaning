#!/bin/bash

# Python implementation needed to avoid google.protobuf.message.DecdoeError
# Probably due to buffer being too big: https://github.com/tensorflow/tensorflow/issues/582
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python numactl --cpunodebind=1 --membind=1 python3 train-component-separation512.py 2>&1 | tee logs/output`date +%s`.txt
