#!/usr/bin/env bash
set -euo pipefail

# На Windows при наличии локального virtualenv используем его конвертер.
if [[ -x ".venv/Scripts/tensorflowjs_converter.exe" ]]; then
  CONVERTER=".venv/Scripts/tensorflowjs_converter.exe"
  PYTHON=".venv/Scripts/python.exe"
else
  CONVERTER="tensorflowjs_converter"
  PYTHON="python"
fi

"$CONVERTER" \
  --input_format=keras \
  --output_format=tfjs_layers_model \
  best_model.h5 tfjs_model \
  --quantize_float16

# tensorflowjs 3.x сохраняет Keras 3 InputLayer/inbound_nodes в новом формате,
# который TF.js Layers 4.x не читает без этой совместимой нормализации.
"$PYTHON" patch_tfjs_model.py tfjs_model/model.json
