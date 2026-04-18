"""Shared image preprocessing utilities to avoid redundant computation."""

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an image to grayscale, blur, and threshold it.

    Uses Otsu's method for adaptive thresholding instead of a fixed value,
    which handles varying lighting conditions much better.

    Args:
        image: Input image in RGB format (H, W, 3).

    Returns:
        A tuple of (grayscale, blurred, binary_threshold) images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's method automatically picks the optimal threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, blurred, thresh
