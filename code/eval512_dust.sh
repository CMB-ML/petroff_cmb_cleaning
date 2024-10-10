#!/bin/bash

# Python implementation needed to avoid google.protobuf.message.DecdoeError
# Probably due to buffer being too big: https://github.com/tensorflow/tensorflow/issues/582
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 eval-component-separation512_dust.py 2>&1 | tee logs/eval_dust_output`date +%s`.txt
