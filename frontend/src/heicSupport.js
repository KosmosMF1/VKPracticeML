let heic2anyPromise = null;

function loadHeic2Any() {
  if (!heic2anyPromise) {
    // esm.sh отдаёт ESM-сборку, пригодную для dynamic import() в браузере.
    heic2anyPromise = import('https://esm.sh/heic2any@0.0.4').then((m) => m.default || m);
  }
  return heic2anyPromise;
}

/**
 * Конвертирует HEIC/HEIF Blob в PNG Blob.
 * @param {Blob} heicBlob
 * @returns {Promise<Blob>}
 */
export async function convertHeicToPng(heicBlob) {
  const heic2any = await loadHeic2Any();
  const result = await heic2any({ blob: heicBlob, toType: 'image/png', quality: 1 });
  // heic2any может вернуть массив Blob (multi-image HEIC) - берём первый кадр.
  return Array.isArray(result) ? result[0] : result;
}
