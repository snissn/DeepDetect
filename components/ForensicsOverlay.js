import React, { useRef, useEffect, useState } from 'react';

export function denoise(cv, gray) {
  const denoised = new cv.Mat();
  if (typeof cv.fastNlMeansDenoising === 'function') {
    cv.fastNlMeansDenoising(gray, denoised, 10, 7, 21);
  } else {
    console.warn('[ForensicsOverlay] fastNlMeansDenoising unavailable, using GaussianBlur');
    const ksize = new cv.Size(3, 3);
    cv.GaussianBlur(gray, denoised, ksize, 0, 0, cv.BORDER_DEFAULT);
  }
  return denoised;
}

/**
 * ForensicsOverlay
 * Highlights suspicious editing artifacts in an image using:
 *  1) Local variance detection
 *  2) JPEG grid artifact detection
 *  3) PRNU noise-variance
 *  4) Error Level Analysis (ELA)
 *
 * Props:
 *   src: string - URL or data URI of the image to analyze
 *   blockSize: number - block size in pixels (default 32)
 *   thresholdPercent: number - top percentile to highlight (default 80)
 */
export default function ForensicsOverlay({ src, blockSize = 32, thresholdPercent = 80 }) {
  const canvasRef = useRef(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const runAnalysis = () => {
      const cv = window.cv;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = src;

      img.onerror = (e) => console.error('[ForensicsOverlay] Image load error', e);
      img.onload = () => {
        const w = img.width;
        const h = img.height;
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(img, 0, 0);

        // Read into OpenCV
        const srcMat = cv.imread(canvas);
        const gray = new cv.Mat();
        cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);

        const byCount = Math.ceil(h / blockSize);
        const bxCount = Math.ceil(w / blockSize);
        const matsToDelete = [srcMat, gray];

        // 1) Local variance
        const varMap = Array.from({ length: byCount }, () => new Array(bxCount).fill(0));
        for (let by = 0; by < byCount; by++) {
          for (let bx = 0; bx < bxCount; bx++) {
            const y = by * blockSize;
            const x = bx * blockSize;
            const hBlk = Math.min(blockSize, h - y);
            const wBlk = Math.min(blockSize, w - x);
            const roi = gray.roi(new cv.Rect(x, y, wBlk, hBlk));
            const mean = cv.mean(roi)[0];
            const sq = new cv.Mat(); matsToDelete.push(sq);
            cv.multiply(roi, roi, sq);
            const meanSq = cv.mean(sq)[0];
            varMap[by][bx] = meanSq - mean * mean;
            roi.delete();
          }
        }

        // 2) JPEG grid artifacts
        const gridMap = Array.from({ length: byCount }, () => new Array(bxCount).fill(0));
        const gridSize = 8;
        for (let by = 0; by < byCount; by++) {
          for (let bx = 0; bx < bxCount; bx++) {
            let diffs = [];
            for (let yOff = 0; yOff < blockSize; yOff += gridSize) {
              for (let xOff = 0; xOff < blockSize; xOff += gridSize) {
                const y0 = by*blockSize + yOff;
                const x0 = bx*blockSize + xOff;
                if (y0 + gridSize > h || x0 + gridSize > w) continue;
                const blk = gray.roi(new cv.Rect(x0, y0, gridSize, gridSize));
                // vertical boundary diff
                if (x0 + gridSize < w) {
                  const rightCol = blk.col(gridSize - 1);
                  const nextCol = gray.roi(new cv.Rect(x0 + gridSize, y0, 1, gridSize));
                  const tmp = new cv.Mat(); matsToDelete.push(tmp);
                  cv.absdiff(rightCol, nextCol, tmp);
                  diffs.push(cv.mean(tmp)[0]);
                  rightCol.delete(); nextCol.delete();
                }
                // horizontal boundary diff
                if (y0 + gridSize < h) {
                  const bottomRow = blk.row(gridSize - 1);
                  const nextRow = gray.roi(new cv.Rect(x0, y0 + gridSize, gridSize, 1));
                  const tmp = new cv.Mat(); matsToDelete.push(tmp);
                  cv.absdiff(bottomRow, nextRow, tmp);
                  diffs.push(cv.mean(tmp)[0]);
                  bottomRow.delete(); nextRow.delete();
                }
                blk.delete();
              }
            }
            gridMap[by][bx] = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : 0;
          }
        }

        // 3) PRNU variance
        const denoised = denoise(cv, gray); matsToDelete.push(denoised);
        const residual = new cv.Mat(); matsToDelete.push(residual);
        cv.subtract(gray, denoised, residual);
        const prnuMap = Array.from({ length: byCount }, () => new Array(bxCount).fill(0));
        for (let by = 0; by < byCount; by++) {
          for (let bx = 0; bx < bxCount; bx++) {
            const y = by * blockSize;
            const x = bx * blockSize;
            const hBlk = Math.min(blockSize, h - y);
            const wBlk = Math.min(blockSize, w - x);
            const roi = residual.roi(new cv.Rect(x, y, wBlk, hBlk));
            const mean = cv.mean(roi)[0];
            const sq = new cv.Mat(); matsToDelete.push(sq);
            cv.multiply(roi, roi, sq);
            const meanSq = cv.mean(sq)[0];
            prnuMap[by][bx] = meanSq - mean * mean;
            roi.delete();
          }
        }

        // 4) ELA via offscreen canvas
        const off = document.createElement('canvas');
        const offCtx = off.getContext('2d');
        off.width = w; off.height = h;
        offCtx.drawImage(img, 0, 0);
        const origData = ctx.getImageData(0, 0, w, h).data;
        hiddenToBlob(off).then(blob => {
          const url = URL.createObjectURL(blob);
          const img2 = new Image(); img2.crossOrigin = 'anonymous';
          img2.onload = () => {
            offCtx.drawImage(img2, 0, 0);
            const newData = offCtx.getImageData(0, 0, w, h).data;
            const elaMap = Array.from({ length: byCount }, () => new Array(bxCount).fill(0));
            for (let by = 0; by < byCount; by++) {
              for (let bx = 0; bx < bxCount; bx++) {
                let sum = 0;
                for (let yy = 0; yy < blockSize; yy++) {
                  for (let xx = 0; xx < blockSize; xx++) {
                    const py = by*blockSize + yy;
                    const px = bx*blockSize + xx;
                    if (py >= h || px >= w) continue;
                    const idx = (py * w + px) * 4;
                    sum += Math.abs(origData[idx] - newData[idx]);
                  }
                }
                elaMap[by][bx] = sum / (blockSize * blockSize);
              }
            }

            // normalize helper
            const normalize = mat => {
              const flat = mat.flat();
              const min = Math.min(...flat);
              const max = Math.max(...flat);
              return mat.map(r => r.map(v => (v - min) / (max - min + 1e-8)));
            };

            const vN = normalize(varMap).map(r => r.map(v => 1 - v));
            const gN = normalize(gridMap).map(r => r.map(v => 1 - v));
            const pN = normalize(prnuMap).map(r => r.map(v => 1 - v));
            const eN = normalize(elaMap);
            const composite = vN.map((row, by) =>
              row.map((v, bx) => (v + gN[by][bx] + pN[by][bx] + eN[by][bx]) / 4)
            );

            const flatC = composite.flat().sort((a, b) => a - b);
            const thr = flatC[Math.floor(flatC.length * thresholdPercent / 100)];

            ctx.drawImage(img, 0, 0);
            ctx.fillStyle = 'rgba(255,0,0,0.4)';
            composite.forEach((row, by) => row.forEach((score, bx) => {
              if (score >= thr) {
                ctx.fillRect(bx * blockSize, by * blockSize, blockSize, blockSize);
              }
            }));

            URL.revokeObjectURL(url);
            matsToDelete.forEach(m => m.delete());
            setLoading(false);
          };
          img2.src = url;
        });
      };
    };

    if (cv && cv.Mat) {
      console.log('[ForensicsOverlay] OpenCV ready');
      runAnalysis();
    } else {
      console.log('[ForensicsOverlay] Waiting for OpenCV');
      cv['onRuntimeInitialized'] = runAnalysis;
    }
  }, [src, blockSize, thresholdPercent]);

  return (
    <>
      {loading && <div>Analyzing…</div>}
      <canvas ref={canvasRef} style={{ width: '100%', height: 'auto' }} />
    </>
  );
}

// helper to blob
function hiddenToBlob(canvas) {
  return new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.9));
}
