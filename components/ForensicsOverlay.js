import React, { useRef, useEffect } from 'react';

/**
 * ForensicsOverlay
 * Highlights suspicious editing artifacts in an image using local variance,
 * JPEG grid artifacts, PRNU noise-variance, and Error Level Analysis (ELA).
 *
 * Props:
 *   src: string - URL or data URI of the image to analyze
 *   blockSize: number - block size in pixels (default 32)
 *   thresholdPercent: number - top percentile to highlight (default 80)
 */
export default function ForensicsOverlay({ src, blockSize = 32, thresholdPercent = 80 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    console.log('[ForensicsOverlay] effect triggered', { src, blockSize, thresholdPercent });
    const run = () => {
      console.log('[ForensicsOverlay] Starting analysis');
      try {
        const cv = window.cv;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = src;

        img.onerror = (e) => {
          console.error('[ForensicsOverlay] Image failed to load', e);
        };

      img.onload = () => {
        try {
          console.log('[ForensicsOverlay] Image loaded, running detectors');
          const w = img.width;
          const h = img.height;
          canvas.width = w;
          canvas.height = h;
          ctx.drawImage(img, 0, 0);

        let srcMat = cv.imread(canvas);
        let gray = new cv.Mat();
        cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);

        const blocksY = Math.floor(h / blockSize);
        const blocksX = Math.floor(w / blockSize);

        let varMap = Array.from({ length: blocksY }, () => new Array(blocksX).fill(0));
        for (let by = 0; by < blocksY; by++) {
          for (let bx = 0; bx < blocksX; bx++) {
            const roi = gray.roi(new cv.Rect(bx*blockSize, by*blockSize, blockSize, blockSize));
            const mean = cv.mean(roi)[0];
            const sq = new cv.Mat();
            cv.multiply(roi, roi, sq);
            const meanSq = cv.mean(sq)[0];
            varMap[by][bx] = meanSq - mean*mean;
            roi.delete(); sq.delete();
          }
        }

        let gridMap = Array.from({ length: blocksY }, () => new Array(blocksX).fill(0));
        const gridSize = 8;
        for (let by = 0; by < blocksY; by++) {
          for (let bx = 0; bx < blocksX; bx++) {
            let diffs = [];
            for (let gy = 0; gy < blockSize/gridSize; gy++) {
              for (let gx = 0; gx < blockSize/gridSize; gx++) {
                const y0 = by*blockSize + gy*gridSize;
                const x0 = bx*blockSize + gx*gridSize;
                const block = gray.roi(new cv.Rect(x0, y0, gridSize, gridSize));
                if (x0 + gridSize < w) {
                  const col1 = block.col(gridSize-1);
                  const col2 = gray.roi(new cv.Rect(x0+gridSize, y0, 1, gridSize));
                  diffs.push(cv.mean(cv.absdiff(col1, col2))[0]);
                  col1.delete(); col2.delete();
                }
                if (y0 + gridSize < h) {
                  const row1 = block.row(gridSize-1);
                  const row2 = gray.roi(new cv.Rect(x0, y0+gridSize, gridSize, 1));
                  diffs.push(cv.mean(cv.absdiff(row1, row2))[0]);
                  row1.delete(); row2.delete();
                }
                block.delete();
              }
            }
            gridMap[by][bx] = diffs.reduce((a,b) => a + b,0) / diffs.length;
          }
        }

        let denoised = new cv.Mat();
        cv.fastNlMeansDenoising(gray, denoised, 10, 7, 21);
        const residual = new cv.Mat();
        cv.subtract(gray, denoised, residual);
        let prnuMap = Array.from({ length: blocksY }, () => new Array(blocksX).fill(0));
        for (let by = 0; by < blocksY; by++) {
          for (let bx = 0; bx < blocksX; bx++) {
            const roi = residual.roi(new cv.Rect(bx*blockSize, by*blockSize, blockSize, blockSize));
            const mean = cv.mean(roi)[0];
            const sq = new cv.Mat();
            cv.multiply(roi, roi, sq);
            const meanSq = cv.mean(sq)[0];
            prnuMap[by][bx] = meanSq - mean*mean;
            roi.delete(); sq.delete();
          }
        }
        denoised.delete(); residual.delete();

        const hidden = document.createElement('canvas');
        const hCtx = hidden.getContext('2d');
        hidden.width = w; hidden.height = h;
        hCtx.drawImage(img, 0, 0);
        hidden.toBlob(blob => {
          const url = URL.createObjectURL(blob);
          const img2 = new Image();
          img2.onload = () => {
            hCtx.drawImage(img2, 0, 0);
            const d1 = ctx.getImageData(0,0,w,h).data;
            const d2 = hCtx.getImageData(0,0,w,h).data;
            let elaMap = Array.from({ length: blocksY }, () => new Array(blocksX).fill(0));
            for (let by=0; by<blocksY; by++) {
              for (let bx=0; bx<blocksX; bx++) {
                let sum=0;
                for (let y=0; y<blockSize; y++) {
                  for (let x=0; x<blockSize; x++) {
                    const idx = ((by*blockSize+y)*w + (bx*blockSize+x))*4;
                    sum += Math.abs(d1[idx] - d2[idx]);
                  }
                }
                elaMap[by][bx] = sum / (blockSize*blockSize);
              }
            }

            function normalize(mat) {
              const flat = mat.flat();
              const mi = Math.min(...flat);
              const ma = Math.max(...flat);
              return mat.map(row => row.map(v => (v - mi)/(ma - mi + 1e-8)));
            }

            const varN   = normalize(varMap).map(r=>r.map(v=>1-v));
            const gridN  = normalize(gridMap).map(r=>r.map(v=>1-v));
            const prnuN  = normalize(prnuMap).map(r=>r.map(v=>1-v));
            const elaN   = normalize(elaMap);
            let composite = varN.map((r,by) =>
              r.map((v,bx) => (v + gridN[by][bx] + prnuN[by][bx] + elaN[by][bx])/4)
            );

            const flatC = composite.flat().sort((a,b)=>a-b);
            const thr = flatC[Math.floor(flatC.length * thresholdPercent/100)];

            ctx.drawImage(img, 0, 0);
            ctx.fillStyle='rgba(255,0,0,0.4)';
            for (let by=0; by<blocksY; by++) {
              for (let bx=0; bx<blocksX; bx++) {
                if (composite[by][bx] >= thr) {
                  ctx.fillRect(bx*blockSize, by*blockSize, blockSize, blockSize);
                }
              }
            }

            URL.revokeObjectURL(url);
            srcMat.delete(); gray.delete();
          };
          img2.src = url;
        }, 'image/jpeg', 0.9);
        } catch (e) {
          console.error('[ForensicsOverlay] Error during analysis', e);
        }
      };
      } catch (e) {
        console.error('[ForensicsOverlay] Error initializing analysis', e);
      }
    };
    const init = () => {
      const cv = window.cv;
      if (!cv) {
        console.warn('[ForensicsOverlay] OpenCV not loaded yet');
        return false;
      }
      if (cv.Mat) {
        console.log('[ForensicsOverlay] OpenCV ready');
        run();
      } else {
        console.log('[ForensicsOverlay] Waiting for OpenCV runtime');
        cv['onRuntimeInitialized'] = run;
      }
      return true;
    };

    if (!init()) {
      console.log('[ForensicsOverlay] OpenCV not ready, retrying...');
      const id = setInterval(() => {
        if (init()) clearInterval(id);
      }, 100);
      return () => clearInterval(id);
    }
  }, [src, blockSize, thresholdPercent]);

  return <canvas ref={canvasRef} style={{ width: '100%', height: 'auto' }} />;
}
