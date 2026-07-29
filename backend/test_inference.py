"""Локальная проверка трёх коэффициентов, предсказанных обученной моделью."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Инференс модели улучшения фото")
    parser.add_argument("--image", type=Path, required=True, help="Тестовая фотография")
    parser.add_argument("--model", type=Path, default=Path("best_model.h5"), help="Файл Keras-модели")
    parser.add_argument(
        "--output", type=Path, default=Path("improved_photo.jpg"), help="Путь для улучшенной фотографии"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file() or not args.model.is_file():
        raise FileNotFoundError("Не найдена тестовая фотография или файл модели")

    # HDF5-файл предназначен для предсказаний, поэтому состояние optimizer не загружаем.
    model = tf.keras.models.load_model(args.model, compile=False)
    image = tf.keras.utils.load_img(args.image, target_size=(256, 256), color_mode="rgb")
    # Та же нормализация, что и в PhotoEnhancementSequence.
    x = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    brightness, contrast, saturation = model.predict(x[None, ...], verbose=0)[0]

    print("Предсказанные коэффициенты:")
    print(f"Brightness: {brightness:.4f}")
    print(f"Contrast:   {contrast:.4f}")
    print(f"Saturation: {saturation:.4f}")

    # CSV хранит обратные коэффициенты к генерации Brightness -> Contrast -> Color.
    # Поэтому для восстановления применяем операции в обратном порядке.
    with Image.open(args.image) as opened_image:
        original = opened_image.convert("RGB")
    improved = ImageEnhance.Color(original).enhance(float(saturation))
    improved = ImageEnhance.Contrast(improved).enhance(float(contrast))
    improved = ImageEnhance.Brightness(improved).enhance(float(brightness))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    improved.save(args.output)
    print(f"Улучшенная фотография сохранена: {args.output}")


if __name__ == "__main__":
    main()
