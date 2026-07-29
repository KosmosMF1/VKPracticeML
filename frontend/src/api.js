
const SUPPORTED_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/bmp',
  'image/heic',
  'image/heif',
]);

const MAX_MEGAPIXELS = 15;
const MAX_PROCESSING_MS = 30_000;

class ImageEnhancerAPI extends EventTarget {
  constructor({ workerUrl = new URL('./worker.js', import.meta.url), modelUrl = '../model/model.json' } = {}) {
    super();
    this._tasks = new Map(); // taskId -> {status, progress, resultBlob, error, file}
    this._queue = [];
    this._processing = false;
    this._modelUrl = modelUrl;
    // Классический (не модульный) воркер: worker.js использует importScripts()
    // для подключения tfjs, что совместимо со всеми массовыми браузерами
    // (module-воркеры пока не везде стабильны, особенно в Safari).
    this._worker = new Worker(workerUrl);
    this._worker.addEventListener('message', (e) => this._onWorkerMessage(e.data));
    this._worker.postMessage({ type: 'init', modelUrl: this._modelUrl });
  }

  /**
   * Метод постановки задачи.
   * Принимает File/Blob с исходным изображением, возвращает идентификатор задачи.
   */
  async submitTask(file) {
    if (!(file instanceof Blob)) {
      throw new TypeError('submitTask ожидает File или Blob с изображением');
    }
    if (file.type && !SUPPORTED_TYPES.has(file.type) && !this._looksLikeHeic(file)) {
      throw new Error(`Неподдерживаемый формат: ${file.type || 'unknown'}. Поддерживаются JPG, PNG, HEIC, BMP.`);
    }

    const taskId = crypto.randomUUID();
    const task = { status: 'queued', progress: 0, resultBlob: null, error: null, file, startedAt: null };
    this._tasks.set(taskId, task);
    this._queue.push(taskId);
    this._emitStatus(taskId);
    this._pump();
    return taskId;
  }

  /**
   * Метод получения статуса задачи.
   */
  getTaskStatus(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) throw new Error(`Неизвестная задача: ${taskId}`);
    return { status: task.status, progress: task.progress };
  }

  /**
   * Метод прерывания задачи.
   */
  async cancelTask(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) return { success: false };

    if (task.status === 'done' || task.status === 'error' || task.status === 'cancelled') {
      return { success: false };
    }

    // Задача ещё в очереди и не отправлена в воркер - просто убираем.
    const queueIdx = this._queue.indexOf(taskId);
    if (queueIdx !== -1) {
      this._queue.splice(queueIdx, 1);
      this._setStatus(taskId, 'cancelled', task.progress);
      return { success: true };
    }

    // Задача уже обрабатывается воркером - просим прервать.
    if (this._currentTaskId === taskId) {
      this._worker.postMessage({ type: 'cancel', taskId });
      return { success: true };
    }

    return { success: false };
  }

  /**
   * Метод получения готового изображения.
   */
  async getResult(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) throw new Error(`Неизвестная задача: ${taskId}`);
    if (task.status === 'error') throw new Error(task.error || 'Задача завершилась с ошибкой');
    if (task.status !== 'done') throw new Error(`Результат ещё не готов (статус: ${task.status})`);
    return task.resultBlob;
  }

  // --------------------------- внутреннее ---------------------------

  _looksLikeHeic(file) {
    const name = (file.name || '').toLowerCase();
    return name.endsWith('.heic') || name.endsWith('.heif');
  }

  _setStatus(taskId, status, progress) {
    const task = this._tasks.get(taskId);
    if (!task) return;
    task.status = status;
    task.progress = progress;
    this._emitStatus(taskId);
  }

  _emitStatus(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) return;
    this.dispatchEvent(new CustomEvent('statuschange', {
      detail: { taskId, status: task.status, progress: task.progress },
    }));
  }

  async _pump() {
    if (this._processing) return;
    const taskId = this._queue.shift();
    if (!taskId) return;

    const task = this._tasks.get(taskId);
    if (!task) return this._pump();

    this._processing = true;
    this._currentTaskId = taskId;
    task.startedAt = performance.now();
    this._setStatus(taskId, 'decoding', 5);

    // Общий предохранитель по времени согласно ТЗ (макс. 30с на изображение).
    this._watchdog = setTimeout(() => {
      if (this._currentTaskId === taskId && task.status !== 'done' && task.status !== 'error') {
        this._worker.postMessage({ type: 'cancel', taskId });
        task.error = 'Превышено максимальное время обработки (30с)';
        this._setStatus(taskId, 'error', task.progress);
        this._finishCurrent();
      }
    }, MAX_PROCESSING_MS);

    try {
      // HEIC/HEIF декодируется на основном потоке (см. heicSupport.js),
      // т.к. это самый совместимый способ подключить libheif-wasm.
      let payload = task.file;
      if (this._looksLikeHeic(task.file) || task.file.type === 'image/heic' || task.file.type === 'image/heif') {
        const { convertHeicToPng } = await import('./heicSupport.js');
        payload = await convertHeicToPng(task.file);
      }

      const buffer = await payload.arrayBuffer();
      this._worker.postMessage(
        { type: 'process', taskId, buffer, mime: payload.type || 'image/jpeg' },
        [buffer]
      );
    } catch (err) {
      task.error = err.message || String(err);
      this._setStatus(taskId, 'error', task.progress);
      this._finishCurrent();
    }
  }

  _finishCurrent() {
    clearTimeout(this._watchdog);
    this._processing = false;
    this._currentTaskId = null;
    this._pump();
  }

  _onWorkerMessage(msg) {
    const { taskId, type } = msg;
    const task = this._tasks.get(taskId);
    if (!task) return;

    switch (type) {
      case 'progress':
        this._setStatus(taskId, msg.status, msg.progress);
        break;
      case 'result':
        task.resultBlob = msg.blob;
        this._setStatus(taskId, 'done', 100);
        this._finishCurrent();
        break;
      case 'error':
        task.error = msg.error;
        this._setStatus(taskId, 'error', task.progress);
        this._finishCurrent();
        break;
      case 'cancelled':
        this._setStatus(taskId, 'cancelled', task.progress);
        this._finishCurrent();
        break;
      default:
        break;
    }
  }
}

export { ImageEnhancerAPI, MAX_MEGAPIXELS, MAX_PROCESSING_MS };
