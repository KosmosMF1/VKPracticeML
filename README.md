# VKPracticeML

Модель на базе MobileNetV2 предсказывает три коэффициента улучшения фотографии:
Brightness, Contrast и Saturation.

## Структура проекта

```text
generate_dataset.py  # Создание аугментированных изображений и dataset.csv
train_model.py       # Генератор данных, архитектура MobileNetV2 и обучение
test_inference.py    # Предсказание коэффициентов и сохранение улучшенного фото
export.sh            # Экспорт best_model.h5 в TensorFlow.js с float16-квантованием
images/              # Исходные изображения (локально, Git не отслеживает)
```

## Зависимости

```powershell
pip install tensorflow pandas Pillow tensorflowjs
```

## 1. Создание синтетического датасета

Поместите исходные изображения в `images/`, затем выполните:

```powershell
python generate_dataset.py --input-dir images --copies-per-image 5
```

Команда создаст папку `augmented_images/` и `dataset.csv`. Для обучения требуются
колонки:

```text
image_path,target_brightness,target_contrast,target_saturation
```

Целевые значения — обратные к использованным коэффициентам аугментации.

## 2. Обучение

```powershell
python train_model.py --csv-path dataset.csv --epochs 30 --batch-size 32
```

`train_model.py` создаёт кастомный `tf.keras.utils.Sequence`, нормализует пиксели
в диапазон `[0, 1]`, использует замороженную MobileNetV2 с ImageNet-весами,
`GlobalAveragePooling2D` и `Dense(3, activation="relu")`.

Лучшая checkpoint-модель временно хранится в `best_checkpoint.keras`, а после
обучения автоматически экспортируется в `best_model.h5`.

## 3. Инференс и сохранение улучшенной фотографии

```powershell
python test_inference.py --image images/example.jpg --model best_model.h5 --output improved_photo.jpg
```

Скрипт выводит три предсказанных коэффициента и сохраняет обработанное изображение.

## 4. Экспорт в TensorFlow.js

В Git Bash, WSL или другом Bash-терминале:

```bash
bash export.sh
```

Скрипт создаёт `tfjs_model/` в формате `tfjs_layers_model` и использует
`--quantize_float16` для уменьшения размера весов.

## Что не публикуется

`.gitignore` исключает исходные и аугментированные изображения, `dataset.csv`,
обученные модели, экспорт TensorFlow.js, `.agents` и все папки `__pycache__`.
