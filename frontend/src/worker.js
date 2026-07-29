const TFJS_VERSION = '4.20.0';
importScripts(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@${TFJS_VERSION}/dist/tf.min.js`);

const MODEL_INPUT_SIZE = 256;
const MAX_MEGAPIXELS = 15;

let model = null;
let modelLoadError = null;
let cancelledTaskId = null;

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      model = await tf.loadLayersModel(msg.modelUrl);
      // "Прогрев" модели, чтобы первая реальная задача не платила
      // за компиляцию шейдеров/kernels.
      const warmup = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
      const out = model.predict(warmup);
      tf.dispose([warmup, out]);
    } catch (err) {
      // Модель ещё не доложена в model/ (см. README) - используем
      // эвристический фолбэк, чтобы пайплайн оставался рабочим.
      modelLoadError = err.message || String(err);
      console.warn('[worker] Модель не загружена, включён эвристический фолбэк:', modelLoadError);
    }
    return;
  }

  if (msg.type === 'cancel') {
    cancelledTaskId = msg.taskId;
    return;
  }

  if (msg.type === 'process') {
    await processTask(msg.taskId, msg.buffer, msg.mime);
    return;
  }
};

function report(taskId, status, progress) {
  self.postMessage({ type: 'progress', taskId, status, progress });
}

function isCancelled(taskId) {
  return cancelledTaskId === taskId;
}

async function processTask(taskId, buffer, mime) {
  try {
    report(taskId, 'decoding', 10);
    const blob = new Blob([buffer], { type: mime });
    const bitmap = await createImageBitmap(blob);

    const megapixels = (bitmap.width * bitmap.height) / 1_000_000;
    if (megapixels > MAX_MEGAPIXELS) {
      bitmap.close();
      throw new Error(
        `Изображение слишком велико: ${megapixels.toFixed(1)} Мпк (максимум ${MAX_MEGAPIXELS} Мпк)`
      );
    }
    if (isCancelled(taskId)) return finishCancelled(taskId);

    report(taskId, 'preprocessing', 25);
    const { brightness, contrast, saturation } = await predictCoefficients(bitmap, taskId);
    if (isCancelled(taskId)) return finishCancelled(taskId);

    report(taskId, 'inference', 55);
    // predictCoefficients уже включает инференс, статус выше выставлен
    // постфактум ради простоты линейного прогресса на UI.

    report(taskId, 'enhancing', 75);
    const resultCanvas = applyEnhancement(bitmap, brightness, contrast, saturation);
    bitmap.close();
    if (isCancelled(taskId)) return finishCancelled(taskId);

    report(taskId, 'encoding', 90);
    const outMime = mime === 'image/png' ? 'image/png' : 'image/jpeg';
    const resultBlob = await resultCanvas.convertToBlob({ type: outMime, quality: 0.92 });

    self.postMessage({ type: 'result', taskId, blob: resultBlob });
  } catch (err) {
    self.postMessage({ type: 'error', taskId, error: err.message || String(err) });
  } finally {
    if (cancelledTaskId === taskId) cancelledTaskId = null;
  }
}

function finishCancelled(taskId) {
  cancelledTaskId = null;
  self.postMessage({ type: 'cancelled', taskId });
}

/**
 * Уменьшает изображение до входного размера модели и возвращает
 * предсказанные коэффициенты коррекции.
 * Если модель не загрузилась, используется эвристический фолбэк
 * на основе гистограммы яркости/насыщенности.
 */
async function predictCoefficients(bitmap, taskId) {
  const canvas = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const imageData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const quality = analyzeImageQuality(imageData);

  if (model) {
    return tf.tidy(() => {
      const tensor = tf.browser
        .fromPixels(imageData)
        .toFloat()
        .div(255)
        .expandDims(0);
      const prediction = model.predict(tensor);
      const [brightness, contrast, saturation] = prediction.dataSync();
      return adaptCoefficients({ brightness, contrast, saturation }, quality);
    });
  }

  return heuristicCoefficients(quality);
}

/**
 * Простая эвристика на основе гистограммы, используется только пока
 * обученная модель (best_model.h5 -> tfjs_model/) ещё не подключена.
 * Цель - подобрать множители так, чтобы средняя яркость и разброс
 * значений канала приближались к "нормальным" (0.5 / достаточный
 * контраст), а насыщенность оставалась в разумных пределах.
 */
function analyzeImageQuality(imageData) {
  const { data } = imageData;
  let sum = 0;
  let sumSquared = 0;
  let sumSat = 0;
  let darkPixels = 0;
  let lightPixels = 0;
  const n = data.length / 4;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    sum += luminance;
    sumSquared += luminance * luminance;
    sumSat += max === 0 ? 0 : (max - min) / max;
    if (luminance < 0.06) darkPixels += 1;
    if (luminance > 0.94) lightPixels += 1;
  }

  const meanBrightness = sum / n;
  const meanSaturation = sumSat / n;
  const contrast = Math.sqrt(Math.max(0, sumSquared / n - meanBrightness ** 2));

  // Нулевое значение означает обычный диапазон, единица — явную проблему.
  // Пороги консервативны, чтобы не менять хорошие снимки и творческие фото.
  const brightnessProblem = Math.max(
    clamp((0.30 - meanBrightness) / 0.20, 0, 1),
    clamp((meanBrightness - 0.72) / 0.18, 0, 1)
  );
  const contrastProblem = Math.max(
    clamp((0.12 - contrast) / 0.08, 0, 1),
    clamp((contrast - 0.36) / 0.12, 0, 1)
  );
  const darkRatio = darkPixels / n;
  const lightRatio = lightPixels / n;
  const clippingProblem = clamp((darkRatio + lightRatio - 0.16) / 0.28, 0, 1);

  // Белые карточки, сканы и логотипы часто имеют намеренно белый фон и
  // контрастные детали. Это не пересвет: глобальная коррекция только портит
  // такой контент, поэтому распознаём его отдельно.
  const isWhiteBackground = meanBrightness > 0.78 && lightRatio > 0.40;

  return {
    meanBrightness,
    meanSaturation,
    contrast,
    severity: Math.max(brightnessProblem, contrastProblem, clippingProblem),
    isWhiteBackground,
  };
}

function heuristicCoefficients(quality) {
  const brightness = clamp(0.52 / Math.max(quality.meanBrightness, 0.08), 0.55, 1.8);
  const contrast = clamp(0.23 / Math.max(quality.contrast, 0.08), 0.70, 1.30);
  const saturation = clamp(0.45 / Math.max(quality.meanSaturation, 0.08), 0.75, 1.25);

  return { brightness, contrast, saturation };
}

function adaptCoefficients(prediction, quality) {
  if (quality.isWhiteBackground) {
    return { brightness: 1, contrast: 1, saturation: 1 };
  }

  const heuristic = heuristicCoefficients(quality);

  // На нормальном фото применяется 25% поправки. Чем очевиднее дефект,
  // тем больше влияние модели; на экстремальных случаях её страхует гистограмма.
  const strength = 0.25 + 0.75 * quality.severity;
  const heuristicWeight = 0.70 * quality.severity;
  const adjusted = {};

  for (const name of ['brightness', 'contrast', 'saturation']) {
    const safePrediction = clamp(prediction[name], 0.55, 1.8);
    const hybrid = safePrediction * (1 - heuristicWeight) + heuristic[name] * heuristicWeight;
    adjusted[name] = 1 + (hybrid - 1) * strength;
  }
  return adjusted;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

/**
 * Применяет коэффициенты к полноразмерному изображению через
 * встроенный canvas-фильтр (аппаратно ускоряется браузером, что
 * значительно быстрее ручного покиксельного прохода на изображениях
 * вплоть до 15 Мпк).
 */
function applyEnhancement(bitmap, brightness, contrast, saturation) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.filter = `brightness(${pct(brightness)}) contrast(${pct(contrast)}) saturate(${pct(saturation)})`;
  ctx.drawImage(bitmap, 0, 0);
  return canvas;
}

function pct(coefficient) {
  return `${(coefficient * 100).toFixed(1)}%`;
}
