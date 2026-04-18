"""OCR processing functions for different layout types (single-column, multi-column, table).

Key design decision: For single-column and multi-column layouts, we let EasyOCR
handle its own text detection on the full image/region rather than pre-splitting
into line ROIs. EasyOCR's built-in CRAFT detector is far more accurate than
naive morphological line segmentation, which tends to clip words at boundaries
and produce garbled output (e.g. "of" → "Ot", "foolishness" → "toolishness").
"""

import re

import cv2
import numpy as np

from ocr.preprocessing import preprocess
from ocr.reader import load_reader

# Minimum vertical gap (in pixels) between detections to consider a new line
_LINE_GAP_RATIO = 0.5  # fraction of average text height

# Confidence threshold — detections below this are discarded as noise
_MIN_CONFIDENCE = 0.25

# Detections that are ONLY punctuation/symbols and below this confidence are junk
# (e.g. EasyOCR reads a comma as ")" with ~0.3 confidence)
_PUNCT_CONFIDENCE = 0.60

# Regex matching strings that are purely punctuation / symbols (no letters or digits)
_PUNCT_ONLY = re.compile(r'^[^\w\s]+$')


def _enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance image for better OCR accuracy.

    Pipeline:
    1. Upscale small images (EasyOCR struggles below ~640px width)
    2. Apply CLAHE for adaptive contrast
    3. Sharpen to make edges and thin strokes (apostrophes, colons) crisper

    Args:
        image: Input image in RGB format (H, W, 3).

    Returns:
        Enhanced image in RGB format.
    """
    h, w = image.shape[:2]

    # Upscale small images — EasyOCR's CRAFT detector works best at higher res
    min_width = 1024
    if w < min_width:
        scale = min_width / w
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE on the L channel for adaptive contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Unsharp mask — sharpens edges so thin characters like : ' , aren't lost
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return image


def _filter_results(results: list) -> list:
    """Filter out low-confidence and junk detections.

    Removes:
    - Any detection below _MIN_CONFIDENCE
    - Single-punctuation detections below _PUNCT_CONFIDENCE
      (catches EasyOCR misreading commas as ")", periods as "0", etc.)

    Args:
        results: Raw EasyOCR results list [(bbox, text, confidence), ...].

    Returns:
        Filtered results list.
    """
    filtered = []
    for bbox, text, confidence in results:
        # Skip very low confidence
        if confidence < _MIN_CONFIDENCE:
            continue

        # Skip punctuation-only detections with moderate-low confidence
        if _PUNCT_ONLY.match(text.strip()) and confidence < _PUNCT_CONFIDENCE:
            continue

        filtered.append((bbox, text, confidence))

    return filtered


def _group_into_lines(results: list) -> list[list]:
    """Group EasyOCR results into lines based on vertical position.

    Uses the vertical midpoint of each detection and tracks a running
    average midpoint per line to handle slanted text or slight vertical
    drift between words on the same visual line.

    Args:
        results: List of EasyOCR results, each as (bbox, text, confidence).

    Returns:
        A list of lines, where each line is a list of (bbox, text, confidence)
        sorted left-to-right.
    """
    if not results:
        return []

    # Compute vertical midpoint for each detection
    def _mid_y(r):
        ys = [pt[1] for pt in r[0]]
        return (min(ys) + max(ys)) / 2

    def _height(r):
        ys = [pt[1] for pt in r[0]]
        return max(ys) - min(ys)

    # Sort by midpoint Y
    sorted_results = sorted(results, key=_mid_y)

    # Estimate average text height for line-gap threshold
    avg_height = sum(_height(r) for r in sorted_results) / len(sorted_results)
    threshold = avg_height * _LINE_GAP_RATIO

    lines: list[list] = []
    current_line: list = [sorted_results[0]]
    # Track the running midpoint Y of the current line
    current_line_mid_y = _mid_y(sorted_results[0])

    for r in sorted_results[1:]:
        mid_y = _mid_y(r)
        if mid_y - current_line_mid_y > threshold:
            # Flush current line, start new one
            lines.append(sorted(current_line, key=lambda r: min(pt[0] for pt in r[0])))
            current_line = [r]
            current_line_mid_y = mid_y
        else:
            current_line.append(r)
            # Update running midpoint to stay anchored to the group
            current_line_mid_y = sum(_mid_y(x) for x in current_line) / len(current_line)

    if current_line:
        lines.append(sorted(current_line, key=lambda r: min(pt[0] for pt in r[0])))

    return lines


def _draw_boxes(image: np.ndarray, results: list, color=(0, 255, 0), thickness=2) -> None:
    """Draw bounding boxes on the image for each EasyOCR detection.

    Args:
        image: Image array to annotate (modified in-place).
        results: EasyOCR results list.
        color: BGR color tuple for the rectangles.
        thickness: Line thickness in pixels.
    """
    for bbox, _text, _conf in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def process_single_column(image: np.ndarray) -> tuple[str, np.ndarray]:
    """Process an image with single-column layout.

    Pipeline:
    1. Enhance contrast via CLAHE
    2. Run EasyOCR on the enhanced image
    3. Filter out low-confidence/junk detections
    4. Group remaining detections into lines
    5. Draw bounding boxes on the original image

    Args:
        image: Input image in RGB format (H, W, 3). Will be modified in-place
               with bounding box annotations.

    Returns:
        A tuple of (extracted_text, annotated_image).
    """
    reader = load_reader()

    # Enhance contrast for better OCR
    enhanced = _enhance_image(image)
    results = reader.readtext(
        enhanced,
        detail=1,
        paragraph=False,
        contrast_ths=0.1,       # Lower threshold → detect lower-contrast chars like : and '
        adjust_contrast=0.7,    # Auto-adjust contrast for faded text
        text_threshold=0.6,     # Slightly lower → catch small characters
        low_text=0.3,           # Better detection of small/thin text
        link_threshold=0.3,     # Better character linking
        width_ths=0.7,          # Allow wider bounding boxes
    )

    # Filter out junk
    results = _filter_results(results)

    # Draw bounding boxes around each detected text region
    _draw_boxes(image, results)

    # Group detections into lines and build text
    lines = _group_into_lines(results)

    # Compute a GLOBAL avg_char_width across all detected text in the image.
    # Using per-line width is incorrect — a line with both narrow labels and
    # wide values skews the estimate for that line and turns normal inter-word
    # spaces into 2-3 spaces.
    all_char_widths = []
    for line in lines:
        for bbox, text, _ in line:
            w = max(pt[0] for pt in bbox) - min(pt[0] for pt in bbox)
            n = max(len(text), 1)
            all_char_widths.append(w / n)

    global_avg_char_width = max(
        sum(all_char_widths) / len(all_char_widths) if all_char_widths else 10,
        5  # minimum 5px per character
    )

    # A normal inter-word space is roughly 1 char-width in the image.
    # Only insert extra spaces when the gap exceeds 3x that — i.e., it's
    # clearly a structural gap (tab stop, column separator), not just normal
    # word spacing with slight measurement noise.
    structural_gap = global_avg_char_width * 3.0

    extracted_text_lines = []
    for line in lines:
        line_str = ""
        prev_x_end = None

        for bbox, text, _conf in line:
            x_start = min(pt[0] for pt in bbox)
            x_end   = max(pt[0] for pt in bbox)

            if prev_x_end is not None:
                gap = x_start - prev_x_end
                if gap >= structural_gap:
                    # Large structural gap → proportional spaces
                    num_spaces = int(round(gap / global_avg_char_width))
                    num_spaces = max(2, min(num_spaces, 30))
                else:
                    # Normal inter-word gap → single space
                    num_spaces = 1
                line_str += " " * num_spaces

            line_str += text
            prev_x_end = x_end

        extracted_text_lines.append(line_str)

    extracted_text = "\n".join(extracted_text_lines)

    return extracted_text + "\n", image


def process_multi_column(image: np.ndarray, column_splits: list[int] | None = None) -> tuple[str, np.ndarray]:
    """Process an image with multi-column layout.

    Uses column split points (from projection profile analysis) to divide
    the image into vertical strips, then processes each strip independently
    as a single-column layout.

    Args:
        image: Input image in RGB format (H, W, 3). Will be modified in-place.
        column_splits: List of x-coordinates where columns are divided.
            If None, falls back to processing the full image as single-column.

    Returns:
        A tuple of (extracted_text, annotated_image).
    """
    h, w = image.shape[:2]

    if not column_splits:
        # Fallback: no splits detected, treat as single column
        return process_single_column(image)

    # Build column boundaries: [0, split1, split2, ..., width]
    boundaries = [0] + sorted(column_splits) + [w]

    # Draw column separator lines on the image for visual feedback
    for split_x in column_splits:
        cv2.line(image, (split_x, 0), (split_x, h), (0, 0, 255), 2)

    extracted_text = ""
    for i in range(len(boundaries) - 1):
        x_start = boundaries[i]
        x_end = boundaries[i + 1]

        # Extract column ROI (full height)
        column_roi = image[0:h, x_start:x_end].copy()

        # Process each column independently
        column_text, processed_col = process_single_column(column_roi)

        # Copy the annotated column back into the main image
        image[0:h, x_start:x_end] = processed_col

        # Add column separator in text output
        if i > 0:
            extracted_text += "\n--- Column {} ---\n\n".format(i + 1)
        else:
            extracted_text += "--- Column 1 ---\n\n"
        extracted_text += column_text + "\n"

    return extracted_text, image


def process_table(image: np.ndarray) -> tuple[str, np.ndarray]:
    """Process an image with table layout.

    Detects table grid cells via horizontal/vertical line intersection,
    then runs EasyOCR on each cell individually with contrast enhancement.

    Args:
        image: Input image in RGB format (H, W, 3). Will be modified in-place.

    Returns:
        A tuple of (extracted_text, annotated_image) where table cells
        are separated by pipe symbols.
    """
    _, _, thresh = preprocess(image)
    reader = load_reader()

    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 2, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[0] // 2))

    horizontal_lines = cv2.dilate(
        cv2.erode(thresh, horizontal_kernel, iterations=2),
        horizontal_kernel, iterations=2
    )
    vertical_lines = cv2.dilate(
        cv2.erode(thresh, vertical_kernel, iterations=2),
        vertical_kernel, iterations=2
    )

    grid = horizontal_lines & vertical_lines
    contours_cells, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = sorted([cv2.boundingRect(c) for c in contours_cells], key=lambda c: (c[1], c[0]))

    extracted_text = ""
    for x, y, w, h in cells:
        cell_roi = image[y:y + h, x:x + w]
        enhanced_roi = _enhance_image(cell_roi)
        result = reader.readtext(
            enhanced_roi,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.7,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.3,
            width_ths=0.7,
        )
        result = _filter_results(result)
        cell_text = " ".join([res[1] for res in result])
        extracted_text += cell_text + " | "

        # Draw green boxes around detected cells
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return extracted_text, image
