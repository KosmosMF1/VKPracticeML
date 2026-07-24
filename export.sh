#!/usr/bin/env bash
tensorflowjs_converter \
  --input_format=keras \
  --output_format=tfjs_layers_model \
  --quantize_float16 \
  best_model.h5 tfjs_model
