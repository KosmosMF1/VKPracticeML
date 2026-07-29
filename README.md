# VKPracticeML — улучшение фотографий в браузере

Проект обучает модель на базе **MobileNetV2** предсказывать коэффициенты коррекции фотографии:
яркость, контраст и насыщенность. Готовая модель запускается прямо в браузере через
TensorFlow.js: исходные изображения не отправляются на сервер.

> Модель подбирает глобальные коэффициенты, а не восстанавливает потерянные детали.
> Поэтому сильно пересвеченные или почти чёрные области нельзя гарантированно восстановить
> только по одной фотографии.

## Демо

После включения GitHub Pages демо доступно по адресу:

<https://kosmosmf1.github.io/VKPracticeML/>

Сайт разворачивается автоматически из папки `frontend/` при каждом push в `main`.

При первом развёртывании откройте **Settings → Pages** в GitHub-репозитории и в поле
**Build and deployment / Source** выберите **GitHub Actions**. После push откройте
вкладку **Actions**, дождитесь успешного запуска `Deploy demo to GitHub Pages` и
перейдите по ссылке из шага deploy.

## Структура проекта

```text
backend/
  generate_dataset.py   # создание синтетического датасета
  train_model.py        # генератор данных, MobileNetV2 и обучение
  test_inference.py     # локальная проверка .h5-модели
  export.sh             # экспорт .h5 -> TensorFlow.js с float16-квантованием
  patch_tfjs_model.py   # совместимость экспорта Keras 3 и TensorFlow.js

frontend/
  index.html            # статическая страница приложения
  src/api.js            # API обработки изображения
  src/worker.js         # Web Worker: инференс TF.js и коррекция изображения
  model/                # model.json и .bin-веса для фронтенда

images/                 # исходные фото для датасета, не попадают в Git
augmented_images/       # сгенерированные изображения, не попадают в Git
dataset.csv             # разметка датасета, не попадает в Git
```

## Быстрый запуск сайта

Фронтенд статический — сборка и Node.js не требуются. Из корня проекта выполните:

```powershell
python -m http.server 8080 --directory frontend
```

Откройте [http://localhost:8080](http://localhost:8080) в браузере.

Не открывайте `frontend/index.html` двойным кликом через `file://`: Web Worker,
ES-модули и загрузка модели требуют HTTP(S)-сервера.

Приложение принимает JPG, PNG, BMP и HEIC/HEIF, обрабатывает изображения до 15 Мп
и выполняет вычисления в Web Worker, чтобы не блокировать интерфейс. Для хороших
фотографий используется мягкая коррекция, а для явно тёмных, пересвеченных или
контрастных — более сильная.

## Установка Python-зависимостей

Нужен Python 3.10+ и доступ в интернет при первом запуске обучения: TensorFlow
скачает ImageNet-веса MobileNetV2.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow pandas Pillow tensorflowjs
cd ..
```

Для загрузки датасета с Kaggle дополнительно установите:

```powershell
python -m pip install kagglehub
```

## 1. Создание датасета

Поместите исходные качественные изображения в `images/` и выполните из корня
репозитория:

```powershell
python backend/generate_dataset.py --input-dir images --copies-per-image 5 --profile mixed
```

Будут созданы `augmented_images/` и `dataset.csv`. CSV содержит пути к изменённым
изображениям и целевые коэффициенты:

```text
image_path,target_brightness,target_contrast,target_saturation
```

`--profile mixed` сочетает обычные и более сложные примеры. Генератор также добавляет
неизменённые изображения с целевым значением `[1.0, 1.0, 1.0]`, чтобы модель училась
не портить уже хорошие фото.

### Источник — Kaggle DIV2K

После настройки доступа Kaggle можно скачать изображения автоматически:

```powershell
python backend/generate_dataset.py --kaggle-dataset sharansmenon/div2k --copies-per-image 5 --profile mixed
```

Скачанные данные появятся в `kaggle_data/` и не должны добавляться в Git.

## 2. Обучение модели

Перейдите в `backend/`, чтобы файлы модели сохранялись рядом со скриптами:

```powershell
cd backend
python train_model.py --csv-path ..\dataset.csv --epochs 30 --batch-size 32
```

Архитектура: замороженный ImageNet `MobileNetV2` → `GlobalAveragePooling2D` →
`Dense(3, activation="relu")`. Вход модели — изображение `[256, 256, 3]` с пикселями
в диапазоне `[0, 1]`; выход — `[brightness, contrast, saturation]`, где `1.0` означает
отсутствие изменения.

Во время обучения используются `ModelCheckpoint` и `EarlyStopping`. Лучший checkpoint
сохраняется в `backend/best_checkpoint.keras`, итоговая модель для проверки и экспорта —
в `backend/best_model.h5`.

## 3. Локальная проверка модели

Оставаясь в папке `backend/`, выполните:

```powershell
python test_inference.py --image ..\images\example.jpg --model best_model.h5 --output ..\improved_photo.jpg
```

Скрипт выведет три предсказанных коэффициента и сохранит обработанную фотографию.

## 4. Экспорт для сайта

Из `backend/` запустите экспорт. Нужен Bash (Git Bash, WSL или другой Bash-терминал):

```bash
bash export.sh
```

Скрипт создаёт `backend/tfjs_model/` в формате `tfjs_layers_model`, применяет
`--quantize_float16` и автоматически исправляет JSON для совместимости Keras 3 и
TensorFlow.js.

В PowerShell при установленном Git for Windows можно вызвать Git Bash так:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./export.sh
```

Затем скопируйте экспортированную модель во фронтенд:

```powershell
Copy-Item -Path tfjs_model\* -Destination ..\frontend\model -Recurse -Force
```

В `frontend/model/` должны находиться `model.json` и оба файла весов `.bin`. Эти файлы
нужно добавить в Git, иначе опубликованный сайт переключится на эвристический фолбэк.

```powershell
cd ..
git add frontend/model
git commit -m "feat: обновлена модель TensorFlow.js"
git push
```

После публикации GitHub Pages обновите страницу с очисткой кэша: `Ctrl + F5`.

## Что не попадает в Git

`.gitignore` исключает исходные и аугментированные изображения, `dataset.csv`, загрузку
Kaggle, виртуальные окружения, кэш Python и временные результаты. Файлы модели во
`frontend/model/` и `backend/best_model.h5` намеренно не исключены: их можно хранить
в репозитории и использовать для демонстрации.
