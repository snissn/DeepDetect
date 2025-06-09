import { denoise } from '../ForensicsOverlay';

describe('denoise', () => {
  test('uses fastNlMeansDenoising when available', () => {
    const cv = {
      fastNlMeansDenoising: jest.fn(),
      GaussianBlur: jest.fn(),
      Mat: function() {},
      Size: function(w, h) { return { width: w, height: h }; },
      BORDER_DEFAULT: 0,
    };
    const gray = {};
    denoise(cv, gray);
    expect(cv.fastNlMeansDenoising).toHaveBeenCalledWith(gray, expect.any(cv.Mat), 10, 7, 21);
    expect(cv.GaussianBlur).not.toHaveBeenCalled();
  });

  test('falls back to GaussianBlur when fastNlMeansDenoising missing', () => {
    const cv = {
      GaussianBlur: jest.fn(),
      Mat: function() {},
      Size: function(w, h) { return { width: w, height: h }; },
      BORDER_DEFAULT: 0,
    };
    const gray = {};
    denoise(cv, gray);
    expect(cv.GaussianBlur).toHaveBeenCalled();
  });
});
