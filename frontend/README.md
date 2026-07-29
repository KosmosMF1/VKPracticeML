# Frontend — улучшение изображений в браузере (яркость / контраст / цветность)

Реализация клиентской части проекта: полностью браузерный пайплайн, который
запускает ML-модель прямо в браузере пользователя, без отправки изображений на сервер.

## Структура

```
frontend/
  index.html          # демо-страница (загрузка файла, прогресс, до/после, скачивание)
  src/
    api.js            # публичный API модуля (см. контракт ниже)
    worker.js          # Web Worker: tfjs-инференс + обработка изображения
    heicSupport.js     # конвертация HEIC/HEIF -> PNG перед обработкой
  model/
    model.json         # <-- сюда кладём файлы из tfjs_model/ (см. "Подключение модели")
    group1-shard1of1.bin
```

## Как открыть

Статические файлы, сборка не нужна. Достаточно раздать папку любым
статическим сервером (нельзя открывать `index.html` через `file://` — Worker
и dynamic `import()` требуют http/https):

```bash
npx serve frontend
# или
python -m http.server --directory frontend 8080
```


## Подключение модели (для бэкенд-части проекта)

1. Выполнить `export.sh` из корня проекта — он создаёт `tfjs_model/`
   (`tfjs_layers_model`, веса квантованы в float16).
2. Скопировать содержимое `tfjs_model/` в `frontend/model/` так, чтобы
   получилось `frontend/model/model.json` + соответствующие `.bin`-шарды.
3. Ничего в коде менять не нужно — `worker.js` загружает модель по пути
   `../model/model.json` относительно себя.

**Пока модель не подключена**, `worker.js` автоматически переключается на
эвристический алгоритм (расчёт коэффициентов по гистограмме
яркости/насыщенности), чтобы демо оставалось рабочим и его можно было
тестировать до готовности весов.

Ожидаемый контракт модели (см. `train_model.py`):

- Вход: `[1, 224, 224, 3]`, float32, нормализация `[0, 1]`
- Выход: `[1, 3]` — множители `[brightness, contrast, saturation]`,
  где `1.0` = без изменений

## Публичный API (`src/api.js`)

Соответствует разделу ТЗ "Рекомендуемые API модуля":

| ТЗ | Метод |
|---|---|
| Метод постановки задачи | `await api.submitTask(file)` → `taskId` |
| Метод получения статуса задачи | `api.getTaskStatus(taskId)` → `{status, progress}` |
| Метод прерывания задачи | `await api.cancelTask(taskId)` → `{success}` |
| Метод получения готового изображения | `await api.getResult(taskId)` → `Blob` |
| Событие изменения статуса задачи | `api.addEventListener('statuschange', e => e.detail)` |

Пример использования:

```js
import { ImageEnhancerAPI } from './src/api.js';

const api = new ImageEnhancerAPI();

api.addEventListener('statuschange', ({ detail }) => {
  console.log(detail.taskId, detail.status, detail.progress);
});

const taskId = await api.submitTask(fileFromInput);
// ... дождаться status === 'done' через событие ...
const resultBlob = await api.getResult(taskId);
```

Статусы задачи: `queued → decoding → preprocessing → inference → enhancing
→ encoding → done`, либо `error` / `cancelled` на любом шаге.

## Как выполняются требования ТЗ

| Требование | Решение |
|---|---|
| Все массовые современные браузеры | Только стандартные Web API: `Worker`, `OffscreenCanvas`, `createImageBitmap`, canvas `filter`. Без сборки/полифилов, работает в актуальных Chrome/Firefox/Edge/Safari. |
| До 10 МБ суммарного кода | Собственный код — единицы КБ. tfjs и heic2any подключаются с CDN по требованию (heic2any — только если файл действительно HEIC), в репозитории не хранятся. Веса модели (float16-квантование через `export.sh`) — основная статья бюджета, контролируется бэкенд-частью. |
| До 15 Мпк | Явная проверка размера в `worker.js`, задача завершается с понятной ошибкой при превышении. |
| Максимум 30 с | `setTimeout`-предохранитель в `api.js`: если задача не завершилась за 30 с, она принудительно отменяется и помечается ошибкой. |
| JPG / PNG / HEIC / BMP | JPG/PNG/BMP декодируются нативно через `createImageBitmap`; HEIC/HEIF заранее конвертируются в PNG через `heic2any` (libheif/WASM), т.к. большинство браузеров не умеют декодировать HEIC нативно. |
| Асинхронность, без блокировки UI | Весь тяжёлый код (декодирование, инференс, применение фильтра, кодирование) выполняется в `Worker`; основной поток только отправляет/получает сообщения. |
| Информирование о прогрессе | Событие `statuschange` на каждом этапе пайплайна с процентом выполнения. |
