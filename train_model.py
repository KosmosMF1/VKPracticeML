"""Обучение MobileNetV2 для предсказания Brightness, Contrast и Saturation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


IMAGE_SIZE = (256, 256)
TARGET_COLUMNS = ("target_brightness", "target_contrast", "target_saturation")


class PhotoEnhancementSequence(tf.keras.utils.Sequence):
    """Загружает батчи фото и трёх целевых коэффициентов из dataset.csv.

    X_batch имеет форму (batch, 256, 256, 3), значения пикселей находятся в [0, 1].
    y_batch имеет форму (batch, 3) в порядке Brightness, Contrast, Saturation.
    """

    def __init__(
        self,
        csv_path: str | Path,
        batch_size: int = 32,
        shuffle: bool = True,
        row_indices: np.ndarray | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if batch_size < 1:
            raise ValueError("batch_size должен быть не меньше 1")

        csv_path = Path(csv_path)
        dataframe = pd.read_csv(csv_path)
        required_columns = {"image_path", *TARGET_COLUMNS}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"В dataset.csv отсутствуют колонки: {sorted(missing_columns)}")
        if row_indices is not None:
            dataframe = dataframe.iloc[row_indices].reset_index(drop=True)
        if dataframe.empty:
            raise ValueError("В генератор передано ноль строк")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.paths = np.asarray(
            [self._resolve_path(path, csv_path.parent) for path in dataframe["image_path"]],
            dtype=str,
        )
        self.targets = dataframe.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        self.indices = np.arange(len(dataframe))
        self.on_epoch_end()

    @staticmethod
    def _resolve_path(image_path: str, csv_directory: Path) -> str:
        """Относительные пути из CSV интерпретируются относительно папки CSV."""
        path = Path(image_path)
        return str(path if path.is_absolute() else csv_directory / path)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]
        if not len(batch_indices):
            raise IndexError(f"Некорректный номер батча: {batch_index}")

        x_batch = np.empty((len(batch_indices), *IMAGE_SIZE, 3), dtype=np.float32)
        for position, dataset_index in enumerate(batch_indices):
            image = tf.keras.utils.load_img(
                self.paths[dataset_index], target_size=IMAGE_SIZE, color_mode="rgb"
            )
            # Требование генератора: нормализация пикселей к диапазону [0, 1].
            x_batch[position] = tf.keras.utils.img_to_array(image) / 255.0

        return x_batch, self.targets[batch_indices]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_model() -> tf.keras.Model:
    """Собирает замороженную MobileNetV2 и регрессионную голову из трёх ReLU-нейронов."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name="image")
    # Generator выдаёт [0, 1], MobileNetV2 с ImageNet-весами ожидает [-1, 1].
    # Rescaling сериализуется в HDF5 надёжнее, чем выражение с TensorFlow-операциями.
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_preprocessing")(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    predictions = tf.keras.layers.Dense(3, activation="relu", name="enhancement_factors")(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name="photo_enhancement_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели улучшения фотографий")
    parser.add_argument("--csv-path", type=Path, default=Path("dataset.csv"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("best_checkpoint.keras"),
        help="Временный файл лучшей модели для ModelCheckpoint",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("best_model.h5"),
        help="Итоговая HDF5-модель для инференса и TensorFlow.js",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs и --batch-size должны быть не меньше 1")
    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("--validation-split должно быть между 0 и 1")

    dataframe = pd.read_csv(args.csv_path)
    if len(dataframe) < 2:
        raise ValueError("Для разбиения нужны минимум две строки dataset.csv")
    indices = np.arange(len(dataframe))
    np.random.default_rng(args.seed).shuffle(indices)
    validation_size = max(1, int(len(indices) * args.validation_split))

    train_data = PhotoEnhancementSequence(args.csv_path, args.batch_size, True, indices[validation_size:])
    validation_data = PhotoEnhancementSequence(args.csv_path, args.batch_size, False, indices[:validation_size])

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            # Keras 3 требует расширение .keras именно для ModelCheckpoint.
            filepath=args.checkpoint_path, monitor="val_loss", save_best_only=True, mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
        ),
    ]
    model = build_model()
    model.fit(train_data, validation_data=validation_data, epochs=args.epochs, callbacks=callbacks)
    # Конвертируем лучший checkpoint в требуемый старый HDF5-формат уже после обучения.
    best_model = tf.keras.models.load_model(args.checkpoint_path)
    # Optimizer не нужен для инференса/экспорта; без него HDF5 корректно читается Keras 3.
    best_model.save(args.model_path, include_optimizer=False)
    print(f"Лучшая модель сохранена в: {args.model_path}")


if __name__ == "__main__":
    main()
