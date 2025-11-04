#!/bin/bash

/usr/src/tensorrt/bin/trtexec \
        --onnx=/offboard/FoundationStereo/assets/foundation_stereo_inferred.onnx \
        --saveEngine=/offboard/FoundationStereo/assets/trt_cache/foundation_stereo.engine \
        --fp16 \
        --memPoolSize=workspace:8589934592