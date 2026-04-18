"""Unit tests for layout detection constants and preprocessing."""

import numpy as np
from ocr.preprocessing import preprocess


class TestPreprocess:
    def test_returns_three_arrays(self):
        """Preprocessing should return grayscale, blurred, and threshold images."""
        # Create a simple 100x100 RGB test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray, blurred, thresh = preprocess(image)

        assert gray.shape == (100, 100)
        assert blurred.shape == (100, 100)
        assert thresh.shape == (100, 100)

    def test_threshold_is_binary(self):
        """Threshold output should only contain 0 and 255."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        _, _, thresh = preprocess(image)

        unique_vals = set(np.unique(thresh))
        assert unique_vals.issubset({0, 255})
