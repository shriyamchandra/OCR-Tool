"""Layout detection for images — identifies single-column, multi-column, and table layouts.

Uses multiple detection strategies:
1. Grid intersection analysis for tables
2. Vertical projection profile analysis for multi-column text (book pages, newspapers)

The projection profile works by:
- Finding the horizontal extent of all text content
- Searching WITHIN that extent for vertical gutters (strips with near-zero ink)
- Filtering out narrow word-space gaps by keeping only significantly wide gaps
- Validating that each resulting column contains real text
"""

import cv2
import numpy as np

from ocr.preprocessing import preprocess

# Table detection: minimum fraction of total image pixels at grid intersections
TABLE_AREA_RATIO = 0.001


def _find_column_gaps(binary: np.ndarray, min_col_ratio: float = 0.12) -> list[int]:
    """Find vertical column gaps using the vertical projection profile.

    Args:
        binary: Binary image — white (255) = ink/text, black (0) = background.
        min_col_ratio: Minimum width of a column as a fraction of IMAGE width.

    Returns:
        List of x-coordinates where column splits should occur.
        Empty list if no multi-column structure is detected.
    """
    h, w = binary.shape[:2]
    min_col_px = max(int(w * min_col_ratio), 30)

    # Vertical projection: sum of ink pixels per x-column
    projection = np.sum(binary, axis=0).astype(np.float64) / 255.0

    # ── Step 1: Find the actual text bounding box ──
    # Only search for gutters where text actually exists.
    # This prevents false positives from empty margins.
    text_threshold = np.max(projection) * 0.02  # 2% of peak
    text_cols = np.where(projection > text_threshold)[0]
    if len(text_cols) < 20:
        return []  # Not enough text to have columns

    text_left = int(text_cols[0])
    text_right = int(text_cols[-1])
    text_width = text_right - text_left

    if text_width < min_col_px * 2:
        return []  # Not wide enough for 2 columns

    # ── Step 2: Analyze projection within text region ──
    # Add a small inward margin to avoid edge effects
    inner_margin = max(int(text_width * 0.03), 3)
    search_left = text_left + inner_margin
    search_right = text_right - inner_margin

    if search_right <= search_left:
        return []

    region_proj = projection[search_left:search_right]

    # Compute the average ink density in the text region
    avg_ink = np.mean(region_proj)
    if avg_ink < 1.0:
        return []

    # A gutter has very low ink density compared to text columns
    empty_threshold = avg_ink * 0.10  # 10% of average density

    is_empty = region_proj <= empty_threshold

    # ── Step 3: Find contiguous runs of empty columns ──
    gaps = []
    in_gap = False
    gap_start = 0

    for x in range(len(region_proj)):
        if is_empty[x] and not in_gap:
            gap_start = x
            in_gap = True
        elif not is_empty[x] and in_gap:
            gap_width = x - gap_start
            gaps.append((gap_start + search_left, x + search_left, gap_width))
            in_gap = False

    if not gaps:
        return []

    # ── Step 4: Filter — keep only real column gutters ──
    # Column gutters are significantly wider than inter-word spaces.
    # Use the median gap width to set a threshold:
    # real gutters should be at least 2x the median gap width.
    gap_widths = [g[2] for g in gaps]
    median_gap = np.median(gap_widths)
    max_gap = max(gap_widths)

    # If the widest gap is not significantly wider than the median,
    # there's no clear column structure — just word spacing variation.
    if max_gap < median_gap * 1.8:
        return []

    # Keep gaps that are significantly wider than the median
    # (at least 1.5x median and at least 40% of the max gap)
    sig_threshold = max(median_gap * 1.5, max_gap * 0.40)
    column_gaps = [(s, e) for s, e, gw in gaps if gw >= sig_threshold]

    if not column_gaps:
        return []

    # ── Step 5: Convert to split points and validate columns ──
    split_points = sorted([int((g[0] + g[1]) / 2) for g in column_gaps])

    boundaries = [0] + split_points + [w]

    # Each column must be wide enough
    for i in range(len(boundaries) - 1):
        col_width = boundaries[i + 1] - boundaries[i]
        if col_width < min_col_px:
            return []

    # Each column must contain actual text (not just empty space)
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        col_proj = projection[left:right]
        col_avg = np.mean(col_proj)
        if col_avg < avg_ink * 0.15:
            return []  # This "column" is mostly empty — false positive

    return split_points


def detect_layout(image: np.ndarray) -> tuple[str, list[int]]:
    """Detect the layout type and column split points of an image.

    Args:
        image: Input image in RGB format (H, W, 3).

    Returns:
        A tuple of (layout_type, column_splits) where:
        - layout_type is one of 'table', 'multi-column', or 'single-column'
        - column_splits is a list of x-coordinates for column boundaries
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    # Clean binary image with light blur for column gap detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_light = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray_light, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Heavier preprocess for table detection
    _, _, thresh = preprocess(image)

    # ── Step 1: Table detection ──
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 2, 1), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 1)))

    horizontal_lines = cv2.dilate(
        cv2.erode(thresh, horizontal_kernel, iterations=2),
        horizontal_kernel, iterations=2
    )
    vertical_lines = cv2.dilate(
        cv2.erode(thresh, vertical_kernel, iterations=2),
        vertical_kernel, iterations=2
    )

    intersections = horizontal_lines & vertical_lines
    table_area = cv2.countNonZero(intersections)

    if table_area > total_pixels * TABLE_AREA_RATIO:
        contours_intersections, _ = cv2.findContours(
            intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours_intersections) >= 4:
            return 'table', []

    # ── Step 2: Multi-column detection ──
    column_splits = _find_column_gaps(binary)
    if column_splits:
        return 'multi-column', column_splits

    # ── Step 3: Default ──
    return 'single-column', []
