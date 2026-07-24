"""Создание набора аугментированных изображений и CSV-разметки.

Примеры:
    python generate_dataset.py --input-dir images --copies-per-image 10
    python generate_dataset.py --kaggle-dataset sharansmenon/div2k --copies-per-image 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ресайз изображений, аугментация цвета и создание dataset.csv."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-dir", type=Path, help="Папка с исходными изображениями")
    source_group.add_argument(
        "--kaggle-dataset",
        default=None,
        help="Идентификатор набора Kaggle, например sharansmenon/div2k",
    )
    parser.add_argument(
        "--kaggle-download-dir",
        type=Path,
        default=Path("kaggle_data"),
        help="Куда скачать набор Kaggle (по умолчанию: kaggle_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("augmented_images"),
        help="Папка для изменённых изображений (по умолчанию: augmented_images)",
    )
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=5,
        help="Количество копий для каждого исходного изображения (по умолчанию: 5)",
    )
    parser.add_argument(
        "--min-factor",
        type=float,
        default=0.7,
        help="Минимальный фактор аугментации (по умолчанию: 0.7)",
    )
    parser.add_argument(
        "--max-factor",
        type=float,
        default=1.4,
        help="Максимальный фактор аугментации (по умолчанию: 1.4)",
    )
    parser.add_argument(
        "--identity-probability",
        type=float,
        default=0.25,
        help="Доля неизменённых копий с target [1, 1, 1] (по умолчанию: 0.25)",
    )
    parser.add_argument(
        "--single-effect-probability",
        type=float,
        default=0.5,
        help="Доля изменённых копий с одним изменяемым параметром (по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--profile",
        choices=("balanced", "extreme", "mixed"),
        default="balanced",
        help="Профиль аугментаций: balanced, extreme или mixed (по умолчанию: balanced)",
    )
    parser.add_argument(
        "--extreme-probability",
        type=float,
        default=0.5,
        help="Доля extreme-примеров в профиле mixed (по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("dataset.csv"),
        help="Путь к CSV-файлу (по умолчанию: dataset.csv)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed генератора случайных чисел")
    return parser.parse_args()


def augment(image: Image.Image, brightness: float, contrast: float, color: float) -> Image.Image:
    """Применяет заданные преобразования к уже приведённому к RGB изображению."""
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(color)


def sample_extreme_factor(index: int, rng: random.Random) -> float:
    """Создаёт безопасный экстремум: сильное ухудшение без яркого клиппинга."""
    # 0 = Brightness, 1 = Contrast, 2 = Color.
    if index == 0:
        return rng.uniform(0.2, 0.6) if rng.random() < 0.8 else rng.uniform(1.0, 1.25)
    return rng.uniform(0.35, 0.7) if rng.random() < 0.8 else rng.uniform(1.0, 1.3)


def sample_factors(args: argparse.Namespace, rng: random.Random) -> tuple[float, float, float, str]:
    """Создаёт аугментацию; extreme усиливает тёмные и малоконтрастные случаи."""
    if rng.random() < args.identity_probability:
        return 1.0, 1.0, 1.0, "identity"

    profile = args.profile
    if profile == "mixed":
        profile = "extreme" if rng.random() < args.extreme_probability else "balanced"

    factors = (
        [sample_extreme_factor(index, rng) for index in range(3)]
        if profile == "extreme"
        else [rng.uniform(args.min_factor, args.max_factor) for _ in range(3)]
    )
    if rng.random() < args.single_effect_probability:
        active_index = rng.randrange(3)
        for index in range(3):
            if index != active_index:
                factors[index] = 1.0
        parameter = "brightness" if active_index == 0 else "contrast" if active_index == 1 else "color"
        return *factors, f"{profile}_{parameter}"
    return *factors, f"{profile}_combined"


def get_source_dir(args: argparse.Namespace) -> Path:
    """Возвращает локальную папку с источниками, при необходимости скачивая Kaggle dataset."""
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise FileNotFoundError(f"Входная папка не найдена: {args.input_dir}")
        return args.input_dir

    try:
        import kagglehub
    except ImportError as error:
        raise ImportError(
            "Для скачивания с Kaggle установите пакет: pip install kagglehub"
        ) from error

    destination = args.kaggle_download_dir / args.kaggle_dataset.replace("/", "_")
    print(f"Скачивание Kaggle dataset: {args.kaggle_dataset}")
    downloaded_path = kagglehub.dataset_download(
        args.kaggle_dataset, output_dir=str(destination)
    )
    return Path(downloaded_path)


def main() -> None:
    args = parse_args()
    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image должно быть не меньше 1")
    if not 0 < args.min_factor < args.max_factor:
        raise ValueError("Должно выполняться 0 < --min-factor < --max-factor")
    if not 0 <= args.identity_probability <= 1:
        raise ValueError("--identity-probability должно быть между 0 и 1")
    if not 0 <= args.single_effect_probability <= 1:
        raise ValueError("--single-effect-probability должно быть между 0 и 1")
    if not 0 <= args.extreme_probability <= 1:
        raise ValueError("--extreme-probability должно быть между 0 и 1")

    source_dir = get_source_dir(args)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_resolved = args.output_dir.resolve()
    source_files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and output_dir_resolved not in path.resolve().parents
    )
    if not source_files:
        raise FileNotFoundError(f"В папке {source_dir} не найдено поддерживаемых изображений")

    rows: list[dict[str, float | str]] = []
    for source_index, source_path in enumerate(source_files):
        with Image.open(source_path) as opened_image:
            # RGB гарантирует корректную работу ImageEnhance и единый формат PNG.
            original = opened_image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)

        for copy_index in range(args.copies_per_image):
            brightness_factor, contrast_factor, color_factor, augmentation_type = sample_factors(args, rng)

            result = augment(original, brightness_factor, contrast_factor, color_factor)
            output_path = args.output_dir / f"{source_path.stem}_{source_index:05d}_aug_{copy_index:03d}.png"
            result.save(output_path, format="PNG")

            rows.append(
                {
                    "image_path": output_path.as_posix(),
                    "source_path": source_path.as_posix(),
                    "augmentation_type": augmentation_type,
                    "brightness_factor": brightness_factor,
                    "contrast_factor": contrast_factor,
                    "color_factor": color_factor,
                    "target_brightness": 1.0 / brightness_factor,
                    "target_contrast": 1.0 / contrast_factor,
                    "target_saturation": 1.0 / color_factor,
                }
            )

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv_path, index=False)
    print(f"Создано изображений: {len(rows)}")
    print(f"CSV сохранён: {args.csv_path}")


if __name__ == "__main__":
    main()
