"""Приводит Keras 3 JSON из старого tensorflowjs-converter к формату TF.js Layers."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def histories(value: object) -> list[list[object]]:
    """Извлекает legacy Keras history из Keras 3 __keras_tensor__ описаний."""
    result: list[list[object]] = []
    if isinstance(value, dict):
        if value.get("class_name") == "__keras_tensor__":
            history = value["config"]["keras_history"]
            result.append([history[0], history[1], history[2], {}])
        else:
            for nested in value.values():
                result.extend(histories(nested))
    elif isinstance(value, list):
        for nested in value:
            result.extend(histories(nested))
    return result


def patch_config(value: object) -> None:
    """Рекурсивно преобразует Keras 3 InputLayer и inbound_nodes."""
    if isinstance(value, list):
        for item in value:
            patch_config(item)
        return
    if not isinstance(value, dict):
        return

    if value.get("class_name") == "InputLayer":
        config = value.get("config", {})
        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")

    inbound_nodes = value.get("inbound_nodes")
    if isinstance(inbound_nodes, list) and inbound_nodes and isinstance(inbound_nodes[0], dict):
        value["inbound_nodes"] = [histories(node.get("args", [])) for node in inbound_nodes]

    for nested in value.values():
        patch_config(nested)


def depthwise_layer_names(value: object) -> set[str]:
    """Возвращает имена всех слоёв DepthwiseConv2D в topology."""
    names: set[str] = set()
    if isinstance(value, dict):
        if value.get("class_name") == "DepthwiseConv2D":
            name = value.get("config", {}).get("name")
            if isinstance(name, str):
                names.add(name)
        for nested in value.values():
            names.update(depthwise_layer_names(nested))
    elif isinstance(value, list):
        for nested in value:
            names.update(depthwise_layer_names(nested))
    return names


def patch_weight_names(model_json: dict[str, object]) -> None:
    """Исправляет Keras 3 kernel -> ожидаемый TF.js depthwise_kernel.

    У обычных Conv2D имя ``kernel`` корректно. Замена нужна исключительно
    для DepthwiseConv2D, который в TF.js создаёт переменную depthwise_kernel.
    """
    depthwise_names = depthwise_layer_names(model_json["modelTopology"])
    for group in model_json.get("weightsManifest", []):
        for weight in group.get("weights", []):
            layer_name, separator, variable_name = weight["name"].rpartition("/")
            if separator and layer_name in depthwise_names and variable_name == "kernel":
                weight["name"] = f"{layer_name}/depthwise_kernel"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Использование: python patch_tfjs_model.py <путь_к_model.json>")
    path = Path(sys.argv[1])
    with path.open(encoding="utf-8") as file:
        model_json = json.load(file)
    patch_config(model_json["modelTopology"])
    patch_weight_names(model_json)
    with path.open("w", encoding="utf-8") as file:
        json.dump(model_json, file, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
